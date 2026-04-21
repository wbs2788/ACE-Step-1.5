#!/usr/bin/env python3
"""Batch CSV inference for ACE-Step with a trained LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_handler import LLMHandler


def build_parser() -> argparse.ArgumentParser:
    """Build parser for CSV batch inference."""
    parser = argparse.ArgumentParser(
        description="Run batch text-to-music inference from a CSV using a trained LoRA adapter.",
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
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--duration", type=float, default=30.0, help="Target duration in seconds.")
    parser.add_argument("--batch-size", type=int, default=1, help="Outputs per prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Seed; use -1 for random.")
    parser.add_argument("--inference-steps", type=int, default=8, help="Diffusion steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="CFG scale.")
    parser.add_argument("--audio-format", default="wav", choices=["wav", "mp3", "flac"], help="Audio format.")
    parser.add_argument("--thinking", action="store_true", help="Enable LM reasoning.")
    parser.add_argument("--offload-to-cpu", action="store_true", help="Enable CPU offload.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to process.")
    return parser


def init_runtime(args: argparse.Namespace) -> tuple[AceStepHandler, LLMHandler]:
    """Initialize handler and load the trained LoRA adapter."""
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    dit_handler.initialize_service(
        project_root=".",
        config_path=args.config_path,
        device=args.device,
        use_flash_attention=None,
        compile_model=False,
        offload_to_cpu=args.offload_to_cpu,
        offload_dit_to_cpu=False,
    )
    print(dit_handler.load_lora(args.lora_path))
    print(dit_handler.set_lora_scale(args.lora_scale))
    print(dit_handler.set_use_lora(True))
    return dit_handler, llm_handler


def generate_one(
    dit_handler: AceStepHandler,
    llm_handler: LLMHandler,
    caption: str,
    lyrics: str,
    save_dir: str,
    seed: int,
    duration: float,
    batch_size: int,
    inference_steps: int,
    guidance_scale: float,
    audio_format: str,
    thinking: bool,
):
    """Run one generation call."""
    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        instrumental=lyrics.strip().lower() in {"[instrumental]", "[inst]"},
        duration=duration,
        inference_steps=inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
        thinking=thinking,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_lyrics=False,
        use_cot_language=False,
    )
    config = GenerationConfig(
        batch_size=batch_size,
        use_random_seed=seed < 0,
        audio_format=audio_format,
    )
    return generate_music(dit_handler, llm_handler, params, config, save_dir=save_dir)


def main() -> int:
    """Run batch CSV inference."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.limit is not None:
        df = df.head(args.limit)

    dit_handler, llm_handler = init_runtime(args)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row[args.id_column])
        caption = str(row[args.caption_column])

        final_audio_path = output_dir / f"{sample_id}.{args.audio_format}"
        if final_audio_path.exists():
            continue

        temp_save_dir = output_dir / sample_id
        temp_save_dir.mkdir(parents=True, exist_ok=True)

        result = generate_one(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            caption=caption,
            lyrics=args.lyrics,
            save_dir=str(temp_save_dir),
            seed=args.seed,
            duration=args.duration,
            batch_size=args.batch_size,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            audio_format=args.audio_format,
            thinking=args.thinking,
        )

        generated = sorted(temp_save_dir.glob(f"*.{args.audio_format}"))
        if not generated:
            raise FileNotFoundError(f"No generated audio found for {sample_id}: {result}")
        generated[0].replace(final_audio_path)
        for extra in temp_save_dir.iterdir():
            if extra.exists():
                extra.unlink()
        temp_save_dir.rmdir()
        print(sample_id, final_audio_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
