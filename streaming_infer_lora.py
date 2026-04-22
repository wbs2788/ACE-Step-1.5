#!/usr/bin/env python3
"""Student-only streaming inference aligned with consistency training."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import soundfile as sf
import torch

from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams
from acestep.llm_inference import LLMHandler


def build_parser() -> argparse.ArgumentParser:
    """Build parser for streaming LoRA inference."""
    parser = argparse.ArgumentParser(
        description="Run streaming student-only inference with a trained consistency LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Base checkpoint root.")
    parser.add_argument("--config-path", default="acestep-v15-xl-turbo", help="Base DiT model name or path.")
    parser.add_argument("--lora-path", default="./output/xl_consistency_v1/final", help="LoRA adapter directory.")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale.")
    parser.add_argument("--caption", required=True, help="Music caption/prompt.")
    parser.add_argument("--lyrics", default="[Instrumental]", help="Lyrics or [Instrumental].")
    parser.add_argument("--save-path", default="./output/streaming_inference_lora.wav", help="Output wav path.")
    parser.add_argument("--device", default="cuda", help="Device string.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="Unused placeholder for parity.")
    parser.add_argument("--warmup-seconds", type=float, default=30.0, help="Warmup chunk duration.")
    parser.add_argument("--prediction-seconds", type=float, default=30.0, help="Prediction chunk duration.")
    parser.add_argument("--max-distill-seconds", type=float, default=150.0, help="Total prediction horizon.")
    parser.add_argument("--max-distill-chunks", type=int, default=5, help="Maximum prediction chunks.")
    parser.add_argument("--use-tiled-decode", action="store_true", help="Use tiled VAE decode.")
    parser.add_argument("--offload-to-cpu", action="store_true", help="Enable CPU offload.")
    return parser


def _build_text_payload(
    dit_handler: AceStepHandler,
    caption: str,
    lyrics: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build text and lyric embeddings using the same handler helpers as training/inference."""
    (
        _text_inputs,
        text_token_ids,
        text_attention_masks,
        lyric_token_ids,
        lyric_attention_masks,
        _non_cover_text_input_ids,
        _non_cover_text_attention_masks,
    ) = dit_handler._prepare_text_conditioning_inputs(
        batch_size=1,
        instructions=["Fill the audio semantic mask based on the given conditions:"],
        captions=[caption],
        lyrics=[lyrics],
        parsed_metas=[""],
        vocal_languages=["unknown"],
        audio_cover_strength=1.0,
    )

    text_token_ids = text_token_ids.to(dit_handler.device)
    text_attention_masks = text_attention_masks.to(dit_handler.device)
    lyric_token_ids = lyric_token_ids.to(dit_handler.device)
    lyric_attention_masks = lyric_attention_masks.to(dit_handler.device)

    with dit_handler._load_model_context("text_encoder"):
        text_hidden_states = dit_handler.infer_text_embeddings(text_token_ids).to(dit_handler.dtype)
        lyric_hidden_states = dit_handler.infer_lyric_embeddings(lyric_token_ids).to(dit_handler.dtype)

    return (
        text_hidden_states,
        text_attention_masks.bool(),
        lyric_hidden_states,
        lyric_attention_masks.bool(),
    )


def _prepare_condition_tensors(
    dit_handler: AceStepHandler,
    text_hidden_states: torch.Tensor,
    text_attention_mask: torch.Tensor,
    lyric_hidden_states: torch.Tensor,
    lyric_attention_mask: torch.Tensor,
    total_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare encoder/context latents for full streaming horizon."""
    dit_handler._ensure_silence_latent_on_device()
    silence_slice = dit_handler._get_silence_latent_slice(total_frames).unsqueeze(0).to(
        device=dit_handler.device, dtype=dit_handler.dtype
    )
    full_mask = torch.ones(1, total_frames, device=dit_handler.device, dtype=dit_handler.dtype)
    chunk_masks = torch.ones(1, total_frames, 64, device=dit_handler.device, dtype=torch.bool)
    is_covers = torch.zeros(1, device=dit_handler.device, dtype=torch.bool)

    refer_audio_latents, refer_audio_order_mask = dit_handler.infer_refer_latent(
        [[torch.zeros(2, 96000, device=dit_handler.device, dtype=dit_handler.dtype)]]
    )
    refer_audio_latents = refer_audio_latents.to(device=dit_handler.device, dtype=dit_handler.dtype)

    with torch.inference_mode():
        encoder_hidden_states, encoder_attention_mask, context_latents = dit_handler.model.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_latents,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=silence_slice,
            attention_mask=full_mask,
            silence_latent=dit_handler.silence_latent,
            src_latents=silence_slice,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=None,
        )
    return encoder_hidden_states, encoder_attention_mask, context_latents


def _student_streaming_rollout(
    dit_handler: AceStepHandler,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    warmup_frames: int,
    prediction_frames: int,
    num_chunks: int,
    seed: int,
) -> torch.Tensor:
    """Run student-only streaming generation with cache rollover."""
    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    pkv = None
    generated_chunks: list[torch.Tensor] = []
    decoder = dit_handler.model.decoder
    dtype = dit_handler.dtype
    device = dit_handler.device

    total_chunks = 1 + num_chunks
    for chunk_idx in range(total_chunks):
        if chunk_idx == 0:
            start = 0
            frames = warmup_frames
        else:
            start = warmup_frames + (chunk_idx - 1) * prediction_frames
            frames = prediction_frames
        end = start + frames

        xt = torch.randn(1, frames, 64, device=device, dtype=dtype)
        t = torch.ones(1, device=device, dtype=dtype)
        t_expand = t.view(-1, 1, 1)
        attention_mask = torch.ones(1, frames, device=device, dtype=dtype)
        ctx = context_latents[:, start:end, :]

        with torch.inference_mode():
            outputs = decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=ctx,
                use_cache=True,
                past_key_values=pkv,
            )
            v_pred, pkv = outputs[0], outputs[1]
            pred_x0 = xt - t_expand * v_pred
        generated_chunks.append(pred_x0)

    return torch.cat(generated_chunks, dim=1)


def main() -> int:
    """Run streaming inference and save a waveform."""
    args = build_parser().parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    lora_path = Path(args.lora_path).resolve()
    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoint_dir)

    dit_handler = AceStepHandler()
    _llm_handler = LLMHandler()

    dit_handler.initialize_service(
        project_root=".",
        config_path=args.config_path,
        device=args.device,
        use_flash_attention=None,
        compile_model=False,
        offload_to_cpu=args.offload_to_cpu,
        offload_dit_to_cpu=False,
    )
    print(dit_handler.load_lora(str(lora_path)))
    print(dit_handler.set_lora_scale(args.lora_scale))
    print(dit_handler.set_use_lora(True))

    params = GenerationParams(
        task_type="text2music",
        caption=args.caption,
        lyrics=args.lyrics,
        instrumental=args.lyrics.strip().lower() in {"[instrumental]", "[inst]"},
        duration=args.warmup_seconds + args.max_distill_seconds,
    )
    print(f"Streaming inference caption: {params.caption}")

    fps = 50
    warmup_frames = int(args.warmup_seconds * fps)
    prediction_frames = int(args.prediction_seconds * fps)
    max_distill_frames = int(args.max_distill_seconds * fps)
    num_chunks = min(args.max_distill_chunks, max(1, max_distill_frames // prediction_frames))
    total_frames = warmup_frames + num_chunks * prediction_frames

    text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = _build_text_payload(
        dit_handler, args.caption, args.lyrics
    )
    encoder_hidden_states, encoder_attention_mask, context_latents = _prepare_condition_tensors(
        dit_handler,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        total_frames,
    )

    start_time = time.time()
    pred_latents = _student_streaming_rollout(
        dit_handler,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
        warmup_frames=warmup_frames,
        prediction_frames=prediction_frames,
        num_chunks=num_chunks,
        seed=args.seed,
    )
    diffusion_time = time.time() - start_time
    print(f"Streaming rollout done in {diffusion_time:.2f}s, latents={tuple(pred_latents.shape)}")

    pred_wavs, _pred_latents_cpu, _time_costs = dit_handler._decode_generate_music_pred_latents(
        pred_latents=pred_latents,
        progress=None,
        use_tiled_decode=args.use_tiled_decode,
        time_costs={"total_time_cost": diffusion_time, "diffusion_per_step_time_cost": 0.0},
    )

    wav = pred_wavs[0].detach().cpu().transpose(0, 1).numpy()
    sf.write(str(save_path), wav, dit_handler.sample_rate)
    print(f"Saved streaming inference audio to {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
