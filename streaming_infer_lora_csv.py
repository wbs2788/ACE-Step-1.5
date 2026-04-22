#!/usr/bin/env python3
"""Batch CSV streaming inference aligned with consistency training."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler

from streaming_infer_lora import (
    _build_text_payload,
    _prepare_condition_tensors,
    _student_streaming_rollout,
)


def build_parser() -> argparse.ArgumentParser:
    """Build parser for CSV batch streaming inference."""
    parser = argparse.ArgumentParser(
        description="Run batch streaming inference from a CSV using a consistency LoRA adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv-path", required=True, help="CSV file containing prompts.")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated audio.")
    parser.add_argument("--id-column", default="caption_id", help="CSV column used as output filename stem.")
    parser.add_argument("--caption-column", default="caption", help="CSV column used as prompt text.")
    parser.add_argument("--lyrics", default="[Instrumental]", help="Lyrics to use for every sample.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Base checkpoint root.")
    parser.add_argument("--config-path", default="acestep-v15-xl-turbo", help="Base DiT model name or path.")
    parser.add_argument(
        "--lora-path",
        default="./output/xl_consistency_v1/final",
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale.")
    parser.add_argument("--device", default="cuda", help="Device string.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--warmup-seconds", type=float, default=30.0, help="Warmup chunk duration.")
    parser.add_argument("--prediction-seconds", type=float, default=30.0, help="Prediction chunk duration.")
    parser.add_argument("--max-distill-seconds", type=float, default=150.0, help="Total prediction horizon.")
    parser.add_argument("--max-distill-chunks", type=int, default=5, help="Maximum prediction chunks.")
    parser.add_argument("--audio-format", default="wav", choices=["wav", "flac"], help="Audio format.")
    parser.add_argument("--use-tiled-decode", action="store_true", help="Use tiled VAE decode.")
    parser.add_argument("--offload-to-cpu", action="store_true", help="Enable CPU offload.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to process.")
    return parser


def init_runtime(args: argparse.Namespace) -> AceStepHandler:
    """Initialize the handler and load the trained LoRA adapter."""
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    lora_path = Path(args.lora_path).resolve()
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
    return dit_handler


def run_one_streaming_sample(
    dit_handler: AceStepHandler,
    caption: str,
    lyrics: str,
    save_path: Path,
    warmup_seconds: float,
    prediction_seconds: float,
    max_distill_seconds: float,
    max_distill_chunks: int,
    seed: int,
    use_tiled_decode: bool,
) -> None:
    """Generate one streaming sample and save it to disk."""
    fps = 50
    warmup_frames = int(warmup_seconds * fps)
    prediction_frames = int(prediction_seconds * fps)
    max_distill_frames = int(max_distill_seconds * fps)
    num_chunks = min(max_distill_chunks, max(1, max_distill_frames // prediction_frames))
    total_frames = warmup_frames + num_chunks * prediction_frames

    text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = (
        _build_text_payload(dit_handler, caption, lyrics)
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
        seed=seed,
    )
    diffusion_time = time.time() - start_time

    pred_wavs, _pred_latents_cpu, _time_costs = dit_handler._decode_generate_music_pred_latents(
        pred_latents=pred_latents,
        progress=None,
        use_tiled_decode=use_tiled_decode,
        time_costs={"total_time_cost": diffusion_time, "diffusion_per_step_time_cost": 0.0},
    )

    wav = pred_wavs[0].detach().cpu().transpose(0, 1).numpy()
    sf.write(str(save_path), wav, dit_handler.sample_rate)


def main() -> int:
    """Run batch CSV streaming inference."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.limit is not None:
        df = df.head(args.limit)

    dit_handler = init_runtime(args)

    extension = args.audio_format
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row[args.id_column])
        caption = str(row[args.caption_column])
        save_path = output_dir / f"{sample_id}.{extension}"
        if save_path.exists():
            continue

        run_one_streaming_sample(
            dit_handler=dit_handler,
            caption=caption,
            lyrics=args.lyrics,
            save_path=save_path,
            warmup_seconds=args.warmup_seconds,
            prediction_seconds=args.prediction_seconds,
            max_distill_seconds=args.max_distill_seconds,
            max_distill_chunks=args.max_distill_chunks,
            seed=args.seed,
            use_tiled_decode=args.use_tiled_decode,
        )
        print(sample_id, save_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
