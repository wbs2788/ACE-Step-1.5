"""Tests for prompt-only sequential consistency distillation helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from acestep.training_v2.configs import TrainingConfigV2
from acestep.training_v2.consistency_prompt_data import (
    PromptBatchConditioner,
    load_prompt_samples,
)
from acestep.training_v2.streaming_consistency_module import StreamingConsistencyModule


class _DummyTeacher(torch.nn.Module):
    """Minimal teacher stub with an encoder used by the conditioner."""

    def encoder(
        self,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask,
    ):
        del text_attention_mask
        del lyric_hidden_states
        del lyric_attention_mask
        del refer_audio_acoustic_hidden_states_packed
        del refer_audio_order_mask
        mask = torch.ones(
            text_hidden_states.shape[0],
            text_hidden_states.shape[1],
            dtype=text_hidden_states.dtype,
            device=text_hidden_states.device,
        )
        return text_hidden_states + 1.0, mask


class TestPromptOnlyLoading(unittest.TestCase):
    """Verify prompt-only sources are parsed into training samples."""

    def test_load_prompt_samples_from_dataset_json(self):
        """ACE-Step dataset JSON is converted into prompt-only samples."""
        payload = {
            "metadata": {"tag_position": "prepend", "genre_ratio": 0, "custom_tag": ""},
            "samples": [
                {"caption": "bright synthwave intro", "lyrics": "[Instrumental]"},
                {"caption": "slow piano ballad", "lyrics": "hello world"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prompts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            samples = load_prompt_samples(dataset_json=str(path))

        self.assertEqual(len(samples), 2)
        self.assertIn("bright synthwave intro", samples[0]["prompt"])
        self.assertEqual(samples[1]["lyrics"], "hello world")

    def test_load_prompt_samples_from_directory(self):
        """Prompt directories are expanded into sorted prompt files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "2.txt").write_text("second prompt", encoding="utf-8")
            (root / "1.txt").write_text("first prompt", encoding="utf-8")

            samples = load_prompt_samples(prompt_file=str(root))

        self.assertEqual([sample["metadata"]["caption"] for sample in samples], [
            "first prompt",
            "second prompt",
        ])


class TestPromptBatchConditioner(unittest.TestCase):
    """Verify prompt batches are encoded into synthetic distillation inputs."""

    def test_prepare_batch_builds_synthetic_inputs(self):
        """Prompt batches become encoder states plus silence-based context."""
        tokenizer = MagicMock()
        text_encoder = MagicMock()
        text_encoder.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        text_encoder.embed_tokens.side_effect = lambda ids: ids.unsqueeze(-1).repeat(1, 1, 4).float()
        text_encoder.side_effect = lambda ids: SimpleNamespace(
            last_hidden_state=ids.unsqueeze(-1).repeat(1, 1, 4).float()
        )

        with patch(
            "acestep.training_v2.consistency_prompt_conditioner.load_text_encoder",
            return_value=(tokenizer, text_encoder),
        ), patch(
            "acestep.training_v2.consistency_prompt_conditioner.load_silence_latent",
            return_value=torch.zeros(1, 12, 64),
        ), patch(
            "acestep.training_v2.consistency_prompt_conditioner.encode_text",
            return_value=(torch.ones(1, 3, 4), torch.ones(1, 3)),
        ), patch(
            "acestep.training_v2.consistency_prompt_conditioner.encode_lyrics",
            return_value=(torch.ones(1, 5, 4), torch.ones(1, 5)),
        ):
            conditioner = PromptBatchConditioner(
                teacher_model=_DummyTeacher(),
                checkpoint_dir="/tmp/checkpoints",
                model_variant="turbo",
                device=torch.device("cpu"),
                precision="fp32",
                latent_length=12,
                dtype=torch.float32,
            )
            batch = conditioner.prepare_batch(
                {
                    "prompts": ["test prompt", "another prompt"],
                    "lyrics": ["[Instrumental]", "line one"],
                }
            )

        self.assertEqual(batch["encoder_hidden_states"].shape, (2, 3, 4))
        self.assertEqual(batch["context_latents"].shape, (2, 12, 128))
        self.assertEqual(batch["attention_mask"].shape, (2, 12))


class TestSyntheticBatchHandling(unittest.TestCase):
    """Verify synthetic training uses provided prompt-only context tensors."""

    def test_training_step_uses_supplied_prompt_context(self):
        """Synthetic mode should honor batch context instead of random fallback."""
        teacher = MagicMock()
        teacher.parameters.return_value = []
        student = MagicMock()
        student.decoder.return_value = (torch.zeros(1, 4, 64), "student-pkv")

        module = StreamingConsistencyModule(
            teacher=teacher,
            student=student,
            training_config=TrainingConfigV2(batch_size=1),
            device=torch.device("cpu"),
            dtype=torch.float32,
            warmup_seconds=1.0,
            prediction_seconds=1.0,
            max_distill_seconds=1.0,
            fps=4,
        )
        module.force_input_grads_for_checkpointing = True
        module._generate_teacher_chunk = MagicMock(
            side_effect=[
                (torch.zeros(1, 4, 64), "warmup-pkv"),
                (torch.zeros(1, 4, 64), "next-pkv"),
            ]
        )

        batch = {
            "encoder_hidden_states": torch.ones(1, 3, 4),
            "encoder_attention_mask": torch.ones(1, 3),
            "context_latents": torch.full((1, 8, 128), 7.0),
            "attention_mask": torch.ones(1, 8),
        }

        losses = module.training_step(batch)

        prefix_ctx = module._generate_teacher_chunk.call_args_list[0].args[5]
        self.assertTrue(torch.all(prefix_ctx == 7.0))
        self.assertIn("loss_total", losses)
        self.assertTrue(losses["loss_total"].requires_grad)


if __name__ == "__main__":
    unittest.main()
