"""Prompt-only conditioning helpers for synthetic consistency distillation."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from acestep.training.dataset_builder_modules.preprocess_context import build_context_latents
from acestep.training.dataset_builder_modules.preprocess_encoder import run_encoder
from acestep.training.dataset_builder_modules.preprocess_lyrics import encode_lyrics
from acestep.training.dataset_builder_modules.preprocess_text import encode_text
from acestep.training_v2.model_loader import load_silence_latent, load_text_encoder


class PromptBatchConditioner:
    """Encode prompt-only batches into teacher-ready conditioning tensors."""

    def __init__(
        self,
        teacher_model: Any,
        checkpoint_dir: str,
        model_variant: str,
        device: torch.device,
        precision: str,
        latent_length: int,
        dtype: torch.dtype,
    ) -> None:
        self.teacher_model = teacher_model
        self.device = device
        self.dtype = dtype
        self.latent_length = latent_length
        self.tokenizer, self.text_encoder = load_text_encoder(
            checkpoint_dir,
            device=str(device),
            precision=precision,
        )
        self.silence_latent = load_silence_latent(
            checkpoint_dir,
            device=str(device),
            precision=precision,
            variant=model_variant,
        )

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert prompt strings into synthetic distillation conditioning tensors."""
        prompts = batch["prompts"]
        lyrics_list = batch.get("lyrics") or ["[Instrumental]"] * len(prompts)

        encoder_hidden_states: List[torch.Tensor] = []
        encoder_attention_masks: List[torch.Tensor] = []
        context_latents: List[torch.Tensor] = []

        for prompt, lyrics in zip(prompts, lyrics_list):
            text_hs, text_mask = encode_text(
                self.text_encoder,
                self.tokenizer,
                prompt,
                self.device,
                self.dtype,
            )
            lyric_hs, lyric_mask = encode_lyrics(
                self.text_encoder,
                self.tokenizer,
                lyrics,
                self.device,
                self.dtype,
            )
            with torch.no_grad():
                enc_hs, enc_mask = run_encoder(
                    self.teacher_model,
                    text_hs,
                    text_mask,
                    lyric_hs,
                    lyric_mask,
                    self.device,
                    self.dtype,
                    refer_audio_hidden_states_packed=torch.zeros(
                        1,
                        1,
                        64,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    refer_audio_order_mask=torch.zeros(1, device=self.device, dtype=torch.long),
                )
                ctx = build_context_latents(
                    self.silence_latent,
                    self.latent_length,
                    self.device,
                    self.dtype,
                )

            encoder_hidden_states.append(enc_hs.squeeze(0))
            encoder_attention_masks.append(enc_mask.squeeze(0))
            context_latents.append(ctx.squeeze(0))

        attention_mask = torch.ones(
            len(prompts),
            self.latent_length,
            device=self.device,
            dtype=self.dtype,
        )

        return {
            "encoder_hidden_states": torch.stack(encoder_hidden_states),
            "encoder_attention_mask": torch.stack(encoder_attention_masks),
            "context_latents": torch.stack(context_latents),
            "attention_mask": attention_mask,
            "metadata": batch.get("metadata", []),
        }
