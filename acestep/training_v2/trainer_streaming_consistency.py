"""
StreamingConsistencyTrainer -- Distillation orchestrator for ACE-Step 1.5-XL-Turbo.

Loads Teacher and Student models, initializes the StreamingConsistencyModule,
and runs the training loop with chunked KV-cache distillation.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.nn as nn
import wandb

# ACE-Step utilities
from acestep.training.lora_injection import inject_lora_into_dit
from acestep.training.lokr_utils import inject_lokr_into_dit

# V2 infrastructure
from acestep.training_v2.configs import TrainingConfigV2, LoRAConfigV2, LoKRConfigV2
from acestep.training_v2.consistency_prompt_data import PromptBatchConditioner, PromptOnlyDataModule
from acestep.training_v2.model_loader import load_decoder_for_training
from acestep.training_v2.trainer_basic_loop import run_basic_training_loop
from acestep.training_v2.ui import TrainingUpdate

# Local modules
from acestep.training_v2.streaming_consistency_module import StreamingConsistencyModule, AdapterConfig

logger = logging.getLogger(__name__)

class StreamingConsistencyTrainer:
    """Trainer for Streaming Consistency Distillation.
    
    Manages Teacher (frozen) and Student (trainable) modules.
    """

    def __init__(
        self,
        adapter_config: AdapterConfig,
        training_config: TrainingConfigV2,
    ) -> None:
        self.adapter_config = adapter_config
        self.training_config = training_config
        self.adapter_type = training_config.adapter_type
        
        self.module: Optional[StreamingConsistencyModule] = None
        self.is_training = False

    def train(
        self,
        training_state: Optional[Dict[str, Any]] = None,
    ) -> Generator[TrainingUpdate, None, None]:
        """Run the distillation training loop."""
        self.is_training = True
        cfg = self.training_config
        
        try:
            # -- Seed -------------------------------------------------------
            torch.manual_seed(cfg.seed)
            random.seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

            # -- Data -------------------------------------------------------
            num_workers = cfg.num_workers
            if sys.platform == "win32" and num_workers > 0:
                num_workers = 0

            if cfg.data_free:
                data_module = PromptOnlyDataModule(
                    prompt_file=cfg.prompt_file,
                    dataset_json=cfg.dataset_json,
                    batch_size=cfg.batch_size,
                    num_workers=num_workers,
                    pin_memory=cfg.pin_memory,
                )
            else:
                from acestep.training.data_module import PreprocessedDataModule

                data_module = PreprocessedDataModule(
                    tensor_dir=cfg.dataset_dir,
                    batch_size=cfg.batch_size,
                    num_workers=num_workers,
                    pin_memory=cfg.pin_memory,
                )
            data_module.setup("fit")
            
            if len(data_module.train_dataset) == 0:
                yield TrainingUpdate(0, 0.0, "[FAIL] No valid samples found", kind="fail")
                return

            # -- Load Models -----------------------------------------------
            device = torch.device(cfg.device if cfg.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
            precision = cfg.precision if cfg.precision != "auto" else "bf16"
            dtype = torch.bfloat16 if precision == "bf16" else torch.float32

            yield TrainingUpdate(0, 0.0, f"[INFO] Loading Teacher ({cfg.teacher_variant})...", kind="info")
            teacher_model = load_decoder_for_training(
                cfg.checkpoint_dir, variant=cfg.teacher_variant, device=str(device), precision=precision
            )
            
            yield TrainingUpdate(0, 0.0, f"[INFO] Loading Student ({cfg.model_variant})...", kind="info")
            student_model = load_decoder_for_training(
                cfg.checkpoint_dir, variant=cfg.model_variant, device=str(device), precision=precision
            )

            # -- Adapter injection -----------------------------------------------
            adapter_info: Dict[str, Any] = {}
            lycoris_net: Any = None
            
            if self.adapter_type == "lokr":
                # inject_lokr_into_dit returns (model, lycoris_net, adapter_info)
                student_model, lycoris_net, adapter_info = inject_lokr_into_dit(
                    student_model, self.adapter_config
                )
                # Ensure all parameters are on the correct device after LyCORIS injection
                student_model = student_model.to(device)
            else:
                # inject_lora_into_dit returns (model, adapter_info)
                student_model, adapter_info = inject_lora_into_dit(
                    student_model, self.adapter_config
                )
            
            yield TrainingUpdate(
                0, 0.0, 
                f"[OK] {self.adapter_type.upper()} injected: {adapter_info['trainable_params']:,} params", 
                kind="info"
            )

            # -- Build Module -----------------------------------------------
            self.module = StreamingConsistencyModule(
                teacher=teacher_model,
                student=student_model,
                training_config=cfg,
                device=device,
                dtype=dtype,
                condition_seconds=cfg.condition_seconds,
                prediction_seconds=cfg.prediction_seconds,
                warmup_seconds=cfg.warmup_seconds,
                max_distill_seconds=cfg.max_distill_seconds,
                adapter_info=adapter_info,
                lycoris_net=lycoris_net,
            )
            if cfg.use_wandb and wandb.run is None:
                wandb.init(
                    project=os.getenv("WANDB_PROJECT", "acestep-distillation"),
                    config=cfg.to_dict(),
                    name=os.getenv(
                        "WANDB_NAME",
                        f"{cfg.model_variant}-consistency-{cfg.max_iterations or cfg.max_epochs}",
                    ),
                )
            prompt_conditioner: Optional[PromptBatchConditioner] = None
            if cfg.data_free:
                prompt_conditioner = PromptBatchConditioner(
                    teacher_model=teacher_model,
                    checkpoint_dir=cfg.checkpoint_dir,
                    model_variant=cfg.model_variant,
                    device=device,
                    precision=precision,
                    latent_length=self.module.warmup_frames + self.module.max_distill_frames,
                    dtype=dtype,
                )

            # Wrap standard training_step to handle the dict return from consistency_module
            # run_basic_training_loop expects loss = module.training_step(batch)
            orig_step = self.module.training_step
            def wrapped_step(batch):
                if prompt_conditioner is not None and "prompts" in batch:
                    batch = prompt_conditioner.prepare_batch(batch)
                losses = orig_step(batch)
                
                if self.training_config.use_wandb and wandb.run is not None:
                    wandb.log({
                        "train/loss_time_mse": losses["loss_time_mse"].item(),
                        "train/loss_freq_l1": losses["loss_freq_l1"].item(),
                        "train/loss_diff": losses["loss_diff"].item(),
                        "train/loss_total": losses["loss_total"].item(),
                    })
                    
                return losses["loss_total"]
            
            self.module.training_step = wrapped_step
            # Compatibility note: StreamingConsistencyModule now has .model = student

            # -- Run Loop --------------------------------------------------
            yield from run_basic_training_loop(self, data_module, training_state)

        except Exception as exc:
            logger.exception("Consistency training failed")
            yield TrainingUpdate(0, 0.0, f"[FAIL] Training failed: {exc}", kind="fail")
        finally:
            self.is_training = False

    def stop(self) -> None:
        self.is_training = False
