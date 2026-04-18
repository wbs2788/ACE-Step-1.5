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
        warmup_seconds: float = 1.0,
        max_distill_seconds: float = 12.0,
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
        self.force_input_grads_for_checkpointing = False
        self.training_losses: list[float] = []
        
        self.warmup_frames = int(warmup_seconds * fps)
        self.prediction_frames = int(prediction_seconds * fps)
        self.max_distill_frames = int(max_distill_seconds * fps)
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Student configuration
        self.student.train()
        
        # Native 8-step schedule for Turbo (Shift 3.0)
        self.t_schedule = [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6428, 0.5, 0.3, 0.0]

    def _generate_teacher_chunk(
        self,
        latents_noise: torch.Tensor,
        t_start: float,
        pkv: Any,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Teacher solves ODE for one chunk and returns (x0, updated_pkv)."""
        steps = [t for t in self.t_schedule if t <= t_start]
        if not steps: steps = [t_start, 0.0]
        elif steps[0] < t_start: steps = [t_start] + steps

        curr_xt = latents_noise
        curr_pkv = pkv

        for i in range(len(steps) - 1):
            t_curr, t_next = steps[i], steps[i+1]
            t_tensor = torch.full((latents_noise.shape[0],), t_curr, device=self.device, dtype=self.dtype)
            
            with torch.no_grad():
                outputs = self.teacher.decoder(
                    hidden_states=curr_xt,
                    timestep=t_tensor, timestep_r=t_tensor,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                    use_cache=True,
                    past_key_values=curr_pkv,
                )
                vt, curr_pkv = outputs[0], outputs[1]
                
            dt = t_curr - t_next
            if t_next == 0.0: curr_xt = curr_xt - vt * t_curr
            else: curr_xt = curr_xt - vt * dt

        return curr_xt, curr_pkv

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Sequential autoregressive distillation training step."""
        autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype) if "cuda" in str(self.device) else nullcontext()
        
        with autocast_ctx:
            # 1. Prepare global conditions
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype)
            encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype)
            
            # Identify mode: Real Data vs Prompt-Only (Synthetic)
            target_latents = batch.get("target_latents")
            if target_latents is not None:
                target_latents = target_latents.to(self.device, dtype=self.dtype)
                context_latents = batch["context_latents"].to(self.device, dtype=self.dtype)
                full_att_mask = batch["attention_mask"].to(self.device, dtype=self.dtype)
                bsz = target_latents.shape[0]
            else:
                # Synthetic Mode: Generate from noise using prompt-only conditions
                bsz = encoder_hidden_states.shape[0]
                target_latents = None
                total_frames = self.max_distill_frames + self.warmup_frames
                context_latents = batch.get("context_latents")
                if context_latents is not None:
                    context_latents = context_latents.to(self.device, dtype=self.dtype)
                else:
                    context_latents = torch.randn(
                        bsz,
                        total_frames,
                        128,
                        device=self.device,
                        dtype=self.dtype,
                    )
                full_att_mask = batch.get("attention_mask")
                if full_att_mask is not None:
                    full_att_mask = full_att_mask.to(self.device, dtype=self.dtype)
                else:
                    full_att_mask = torch.ones(
                        bsz,
                        total_frames,
                        device=self.device,
                        dtype=self.dtype,
                    )

            # 2. Warmup Stage: Build initial context (Chunk 0)
            if target_latents is not None:
                # Teacher-forcing from real audio prefix
                prefix_x0 = target_latents[:, :self.warmup_frames, :]
                prefix_ctx = context_latents[:, :self.warmup_frames, :]
                prefix_mask = full_att_mask[:, :self.warmup_frames]
                
                # Extract Teacher PKV
                _, teacher_pkv = self._generate_teacher_chunk(
                    prefix_x0, 0.0, None, 
                    encoder_hidden_states, encoder_attention_mask, prefix_ctx, prefix_mask
                )
            else:
                # Synthetic warmup: Teacher generates 1s from noise
                prefix_noise = torch.randn(bsz, self.warmup_frames, 64, device=self.device, dtype=self.dtype)
                prefix_ctx = context_latents[:, :self.warmup_frames, :]
                prefix_mask = full_att_mask[:, :self.warmup_frames]
                
                _, teacher_pkv = self._generate_teacher_chunk(
                    prefix_noise, 1.0, None,
                    encoder_hidden_states, encoder_attention_mask, prefix_ctx, prefix_mask
                )

            # Student starts with Teacher's warmed context
            student_pkv = copy.deepcopy(teacher_pkv)
            
            # 3. Sequential Distillation Loop
            total_loss_time = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            total_loss_freq = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            total_loss_diff = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            # Predict num_chunks based on prediction_seconds
            step_frames = self.prediction_frames
            num_chunks = max(1, self.max_distill_frames // step_frames)
            max_distill_chunks = getattr(self.training_config, "max_distill_chunks", 3)
            if max_distill_chunks is not None:
                num_chunks = min(max_distill_chunks, num_chunks)
            
            curr_start = self.warmup_frames
            completed_chunks = 0
            for _ in range(num_chunks):
                curr_end = curr_start + self.prediction_frames
                if curr_end > full_att_mask.shape[1]: break
                
                chunk_ctx = context_latents[:, curr_start:curr_end, :]
                chunk_mask = full_att_mask[:, curr_start:curr_end]
                
                # Sample noise and timestep for this segment
                noise = torch.randn(bsz, self.prediction_frames, 64, device=self.device, dtype=self.dtype)
                t_single = sample_discrete_timesteps(1, self.device, self.dtype, self.t_schedule[:-1])
                t = t_single.expand(bsz)
                t_expand = t.view(-1, 1, 1)
                
                # Interpolated xt
                if target_latents is not None:
                    chunk_x0_real = target_latents[:, curr_start:curr_end, :]
                    xt = t_expand * noise + (1.0 - t_expand) * chunk_x0_real
                else:
                    xt = noise # Starting from pure noise in synthetic mode
                
                # A. Teacher Path (8-step)
                with torch.no_grad():
                    target_x0, teacher_pkv = self._generate_teacher_chunk(
                        xt, t[0].item(), teacher_pkv,
                        encoder_hidden_states, encoder_attention_mask, chunk_ctx, chunk_mask
                    )
                
                # B. Student Path (1-step)
                student_xt = xt
                if self.force_input_grads_for_checkpointing:
                    student_xt = student_xt.requires_grad_(True)

                student_outputs = self.student.decoder(
                    hidden_states=student_xt,
                    timestep=t, timestep_r=t,
                    attention_mask=chunk_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=chunk_ctx,
                    use_cache=True,
                    past_key_values=student_pkv, # Updated in-place or returned? (Standard Cache is in-place)
                )
                v_pred = student_outputs[0]
                pred_x0 = student_xt - t_expand * v_pred
                # In-place cache update means student_pkv now contains the state after this chunk.
                # However, for consistency we use Teacher's updated PKV to force the "True" trajectory
                # in the next block. (Self-forcing/Teacher-forcing transition)
                student_pkv = copy.deepcopy(teacher_pkv)
                
                # C. Compute Loss
                losses = compute_latent_frequency_loss(
                    pred_x0, target_x0, 
                    fft_weight=getattr(self.training_config, "fft_weight", 1.0),
                    diff_weight=getattr(self.training_config, "diff_weight", 1.0)
                )
                total_loss_time += losses["loss_time_mse"]
                total_loss_freq += losses["loss_freq_l1"]
                total_loss_diff += losses["loss_diff"]
                
                curr_start = curr_end
                completed_chunks += 1
                
            # Average losses
            divisor = max(1, completed_chunks)
            loss_total = (total_loss_time + total_loss_freq + total_loss_diff) / divisor
            self.training_losses.append(loss_total.detach().float().item())
            return {
                "loss_total": loss_total,
                "loss_time_mse": total_loss_time / divisor,
                "loss_freq_l1": total_loss_freq / divisor,
                "loss_diff": total_loss_diff / divisor,
            }
