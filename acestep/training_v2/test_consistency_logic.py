import torch
import unittest
from typing import Any, Dict

# Model components
from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
from acestep.models.common.configuration_acestep_v15 import AceStepConfig

# Training components
from acestep.training_v2.streaming_consistency_module import StreamingConsistencyModule
from acestep.training_v2.configs import TrainingConfigV2

class TestConsistencyLogic(unittest.TestCase):
    def setUp(self):
        # Create a tiny config to avoid memory issues and speed up testing
        self.config = AceStepConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=32, # Enough to cover timbre/lyric encoders
            num_attention_heads=4,
            num_key_value_heads=2,
            # Sub-encoder depths
            num_lyric_encoder_hidden_layers=2,
            num_timbre_encoder_hidden_layers=2,
            num_audio_decoder_hidden_layers=2,
            # Dimensions
            fsq_dim=64,
            audio_acoustic_hidden_dim=64,
            text_hidden_dim=64,
        )
        
        self.device = torch.device("cpu")
        self.dtype = torch.float32 # Use float32 on CPU for testing
        
        # Randomly initialize Teacher and Student
        print("Initializing random models...")
        self.teacher = AceStepConditionGenerationModel(self.config).to(self.device).to(self.dtype)
        self.student = AceStepConditionGenerationModel(self.config).to(self.device).to(self.dtype)
        
        self.train_config = TrainingConfigV2(
            batch_size=2,
            fft_weight=1.0,
            diff_weight=1.0,
            condition_seconds=10.0,
            prediction_seconds=30.0, 
            warmup_seconds=10.0,
            max_distill_seconds=40.0, # 1s warmup + 3x 1s chunks
        )
        
        # Instantiate module
        self.module = StreamingConsistencyModule(
            teacher=self.teacher,
            student=self.student,
            training_config=self.train_config,
            device=self.device,
            dtype=self.dtype,
            warmup_seconds=10.0,
            prediction_seconds=30.0,
            max_distill_seconds=40.0,
            fps=25, 
        )

    def test_forward_step_real_data(self):
        """Verifies that the sequential training_step runs with real audio data."""
        bsz = 2
        # Use module's actual frames
        total_frames = self.module.warmup_frames + self.module.prediction_frames
        
        # Mock batch
        batch = {
            "target_latents": torch.randn(bsz, total_frames, 64),
            "attention_mask": torch.ones(bsz, total_frames),
            "encoder_hidden_states": torch.randn(bsz, 20, 64),
            "encoder_attention_mask": torch.ones(bsz, 20),
            "context_latents": torch.randn(bsz, total_frames, 128),
        }
        
        print("Running sequential training_step (Real Data)...")
        losses = self.module.training_step(batch)
        
        # Assertions
        self.assertIn("loss_total", losses)
        self.assertGreater(losses["loss_total"].item(), 0.0)
        print(f"Test Real Data OK. Total Loss: {losses['loss_total'].item():.4f}")

    def test_forward_step_synthetic(self):
        """Verifies that the sequential training_step runs in Synthetic Mode (No Real Audio)."""
        bsz = 2
        
        # Mock batch (NO target_latents)
        batch = {
            "encoder_hidden_states": torch.randn(bsz, 20, 64),
            "encoder_attention_mask": torch.ones(bsz, 20),
        }
        
        print("Running sequential training_step (Synthetic Mode)...")
        losses = self.module.training_step(batch)
        
        # Assertions
        self.assertIn("loss_total", losses)
        self.assertGreater(losses["loss_total"].item(), 0.0)
        print(f"Test Synthetic OK. Total Loss: {losses['loss_total'].item():.4f}")

    def test_teacher_pkv_propagation(self):
        """Verifies that the Teacher can generate a chunk and update PKV."""
        bsz = 1
        noise = torch.randn(bsz, 25, 64)
        enc_hs = torch.randn(bsz, 10, 64)
        enc_mask = torch.ones(bsz, 10)
        ctx = torch.randn(bsz, 25, 128)
        mask = torch.ones(bsz, 25)
        
        # Run solver
        print("Testing Teacher chunk generation and PKV update...")
        x0, next_pkv = self.module._generate_teacher_chunk(
            noise, 1.0, None, enc_hs, enc_mask, ctx, mask
        )
        
        self.assertEqual(x0.shape, noise.shape)
        self.assertIsNotNone(next_pkv)
        print("PKV Propagation test passed.")

if __name__ == "__main__":
    unittest.main()
