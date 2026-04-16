"""
StreamingConsistencyModule -- Distillation module for ACE-Step Consistency Flow Matching.

Implements the Teacher-Student interaction logic:
1. Slices 4s input into 1s Prefix (Condition) and 3s Prediction (Target).
2. Uses Teacher (8-step) to generate high-precision x0 target.
3. Uses Student (1-step) to predict x0.
4. Enforces consistency and frequency-aware losses.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache, EncoderDecoderCache

# V2 modules
from acestep.training_v2.configs import LoRAConfigV2, LoKRConfigV2, TrainingConfigV2
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_discrete_timesteps
from acestep.training_v2.consistency_loss import compute_latent_frequency_loss

# Union type for adapter configs
AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2]

logger = logging.getLogger(__name__)


class StreamingConsistencyModule(nn.Module):
    """Module for Streaming Consistency Distillation.
    
    Orchestrates Teacher-Student interaction for 1.5-XL-Turbo distillation.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        training_config: TrainingConfigV2,
        device: torch.device,
        dtype: torch.dtype,
        condition_seconds: float = 1.0,
        prediction_seconds: float = 3.0,
        fps: int = 50,  # Frames per second in latent space
        adapter_info: Optional[Dict[str, Any]] = None,
        lycoris_net: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.training_config = training_config
        self.device = device
        self.dtype = dtype
        self.adapter_info = adapter_info or {}
        self.lycoris_net = lycoris_net
        
        # Compatibility attribute for basic training loop
        self.model = student
        
        self.condition_frames = int(condition_seconds * fps)
        self.prediction_frames = int(prediction_seconds * fps)
        self.total_frames = self.condition_frames + self.prediction_frames
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Student configuration
        self.student.train()
        
        # Native 8-step schedule for Turbo (Shift 3.0)
        self.t_schedule = [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6428, 0.5, 0.3, 0.0]
        
    def _get_past_key_values(
        self, 
        model: nn.Module, 
        latents: torch.Tensor, 
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> Any:
        """Forward prefix latents to extract KV-cache."""
        # Note: model is AceStepConditionGenerationModel
        # We need to forward through decoder with use_cache=True
        with torch.no_grad():
            # Setup dummy timestep for prefix (usually 0 or similar as it's 'real' data)
            t_prefix = torch.zeros((latents.shape[0],), device=self.device, dtype=self.dtype)
            
            pkv = EncoderDecoderCache(DynamicCache(), DynamicCache())
            outputs = model.decoder(
                hidden_states=latents,
                timestep=t_prefix,
                timestep_r=t_prefix,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=True,
                past_key_values=pkv,
            )
            return outputs[1]  # Return KV-cache

    def _teacher_solver_8step(
        self,
        xt: torch.Tensor,
        t_start: float,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Run teacher for multiple steps to find x0 target."""
        # Find remaining steps in schedule
        steps = [t for t in self.t_schedule if t <= t_start]
        if not steps:
            steps = [t_start, 0.0]
        elif steps[0] < t_start:
            steps = [t_start] + steps
            
        curr_xt = xt
        curr_pkv = copy.deepcopy(past_key_values)
        
        for i in range(len(steps) - 1):
            t_curr = steps[i]
            t_next = steps[i+1]
            dt = t_curr - t_next
            
            t_tensor = torch.full((xt.shape[0],), t_curr, device=self.device, dtype=self.dtype)
            
            # Simple Euler for teacher steps (or could use Heun if preferred)
            # Turbo model uses Euler in its native generate_audio
            with torch.no_grad():
                outputs = self.teacher.decoder(
                    hidden_states=curr_xt,
                    timestep=t_tensor,
                    timestep_r=t_tensor,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                    use_cache=True,
                    past_key_values=curr_pkv,
                )
                vt = outputs[0]
                # Note: past_key_values are updated but for consistency model distillation 
                # we usually just attend to the fixed prefix and self-attention within prediction
                # In native code, pkv is updated: curr_pkv = outputs[1]
                # However, for 1-step distillation we might want to keep pkv behavior consistent.
                
            if t_next == 0.0:
                # Direct prediction of x0: x0 = xt - t * v
                curr_xt = curr_xt - vt * t_curr
            else:
                curr_xt = curr_xt - vt * dt
                
        return curr_xt

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Consistency distillation training step."""
        # Mixed-precision context
        autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype) if "cuda" in str(self.device) else nullcontext()

        with autocast_ctx:
            # 1. Prepare Data
            target_latents = batch["target_latents"].to(self.device, dtype=self.dtype)
            attention_mask = batch["attention_mask"].to(self.device, dtype=self.dtype)
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype)
            encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype)
            context_latents = batch["context_latents"].to(self.device, dtype=self.dtype)
            
            # Slice into Prefix (1s) and Target (3s)
            # T dimension is 1
            prefix_x0 = target_latents[:, :self.condition_frames, :]
            predict_x0 = target_latents[:, self.condition_frames:self.total_frames, :]
            
            prefix_att_mask = attention_mask[:, :self.condition_frames]
            predict_att_mask = attention_mask[:, self.condition_frames:self.total_frames]
            
            # Context latents might also need slicing as they contain [src_latents, chunk_mask]
            # Assumed shape [B, T, D*2]
            prefix_ctx = context_latents[:, :self.condition_frames, :]
            predict_ctx = context_latents[:, self.condition_frames:self.total_frames, :]

            # 2. Extract Prefix KV-Cache (Conditional Area)
            # Both Teacher and Student share the same past context logic during distillation
            teacher_prefix_pkv = self._get_past_key_values(
                self.teacher, prefix_x0, prefix_att_mask, 
                encoder_hidden_states, encoder_attention_mask, prefix_ctx
            )

            student_prefix_pkv = self._get_past_key_values(
                self.student, prefix_x0, prefix_att_mask, 
                encoder_hidden_states, encoder_attention_mask, prefix_ctx
            )
            
            # 3. Sample Timestep and Noise for Prediction Area
            bsz = predict_x0.shape[0]
            # t = sample_discrete_timesteps(bsz, self.device, self.dtype, self.t_schedule[:-1]) # Sample from schedule (exclude 0.0)
            t_single = sample_discrete_timesteps(1, self.device, self.dtype, self.t_schedule[:-1])
            t = t_single.expand(bsz) # shape: [bsz]
            
            noise = torch.randn_like(predict_x0)
            t_expand = t.view(-1, 1, 1)
            
            # Interpolated xt in prediction area
            predict_xt = t_expand * noise + (1.0 - t_expand) * predict_x0
            
            # 4. Teacher Path: Generate Target x0 (8-step)
            # We use t as the starting point in the ODE trajectory
            with torch.no_grad():
                target_x0 = self._teacher_solver_8step(
                    predict_xt, t[0].item(), # Simplified: assume same t for batch to use t[0] for search
                    predict_att_mask, encoder_hidden_states, encoder_attention_mask, 
                    predict_ctx, teacher_prefix_pkv
                )

            # 5. Student Path: Predict x0 (1-step)
            # The student takes predict_xt and predicts x0 immediately.
            # IMPORTANT: Student only gets 1 step!
            student_outputs = self.student.decoder(
                hidden_states=predict_xt,
                timestep=t,
                timestep_r=t,
                attention_mask=predict_att_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=predict_ctx,
                use_cache=True, # Distillation allows using cache from prefix
                past_key_values=student_prefix_pkv,
            )
            
            # Since student predicts velocity v, we convert to x0
            # x0 = xt - t * v
            v_pred = student_outputs[0]
            pred_x0 = predict_xt - t_expand * v_pred
            
            # 6. Losses
            loss_dict = compute_latent_frequency_loss(
                pred_x0, 
                target_x0, 
                fft_weight=getattr(self.training_config, "fft_weight", 1.0),
                diff_weight=getattr(self.training_config, "diff_weight", 1.0)
            )
            
            return loss_dict
