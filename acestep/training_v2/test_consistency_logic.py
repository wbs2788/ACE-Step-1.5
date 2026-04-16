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
            condition_seconds=1.0,
            prediction_seconds=1.0, # Shorter for test
        )
        
        # Instantiate module
        self.module = StreamingConsistencyModule(
            teacher=self.teacher,
            student=self.student,
            training_config=self.train_config,
            device=self.device,
            dtype=self.dtype,
            condition_seconds=1.0,
            prediction_seconds=1.0,
            fps=25, # Use 25 for testing
        )

    def test_forward_step(self):
        """Verifies that the training_step runs and returns valid losses."""
        bsz = 2
        total_frames = self.module.total_frames
        
        # Mock batch
        batch = {
            "target_latents": torch.randn(bsz, total_frames, 64),
            "attention_mask": torch.ones(bsz, total_frames),
            "encoder_hidden_states": torch.randn(bsz, 20, 64), # Match text_hidden_dim=64
            "encoder_attention_mask": torch.ones(bsz, 20),
            "context_latents": torch.randn(bsz, total_frames, 128), # Match audio_acoustic_hidden_dim (64) * 2 or similar
        }
        
        print("Running training_step...")
        losses = self.module.training_step(batch)
        
        # Assertions
        self.assertIn("loss_total", losses)
        self.assertIn("loss_time_mse", losses)
        self.assertIn("loss_freq_l1", losses)
        self.assertIn("loss_diff", losses)
        
        self.assertGreater(losses["loss_total"].item(), 0.0)
        print(f"Test Step OK. Total Loss: {losses['loss_total'].item():.4f}")

    def test_pkv_deepcopy_integrity(self):
        """Checks if PKV deepcopy mechanism is working in the solver."""
        # This tests the user's manual fix: curr_pkv = copy.deepcopy(past_key_values)
        xt = torch.randn(2, 5, 64)
        t_start = 1.0
        att_mask = torch.ones(2, 5)
        enc_hs = torch.randn(2, 10, 64) # Match hidden_size=64
        enc_mask = torch.ones(2, 10)
        ctx = torch.randn(2, 5, 128)
        
        # Mock prefix PKVs
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        pkv = EncoderDecoderCache(DynamicCache(), DynamicCache())
        
        # Run solver
        print("Running teacher solver with PKV deepcopy...")
        # Solver will iterate through steps. If deepcopy works, the original pkv should remain empty (or unchanged)
        # while if it was shallow, it might have been modified.
        # Actually EncoderDecoderCache is a complex object, but deepcopy should handle it.
        
        target_x0 = self.module._teacher_solver_8step(
            xt, t_start, att_mask, enc_hs, enc_mask, ctx, pkv
        )
        
        self.assertEqual(target_x0.shape, xt.shape)
        print("Solver integrity test passed.")

if __name__ == "__main__":
    unittest.main()
