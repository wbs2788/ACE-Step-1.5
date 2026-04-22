#!/usr/bin/env python3
"""Minimal text-to-music inference entry point with LoRA loading."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


def build_parser() -> argparse.ArgumentParser:
    """Build a minimal inference parser for trained LoRA adapters."""
    parser = argparse.ArgumentParser(
        description="Run ACE-Step text-to-music inference with a trained LoRA adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Base checkpoint root.")
    parser.add_argument(
        "--config-path",
        default="acestep-v15-xl-turbo",
        help="Base DiT model name or path used for inference.",
    )
    parser.add_argument(
        "--lora-path",
        default="./output/xl_consistency_v1/final",
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale.")
    parser.add_argument("--caption", required=True, help="Music caption/prompt.")
    parser.add_argument("--lyrics", default="[Instrumental]", help="Lyrics text or [Instrumental].")
    parser.add_argument("--duration", type=float, default=30.0, help="Target duration in seconds.")
    parser.add_argument("--save-dir", default="./output/inference_lora", help="Output directory.")
    parser.add_argument("--device", default="cuda", help="Device string for initialize_service.")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "pt", "mlx"], help="LM backend.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed; set -1 for random.")
    parser.add_argument("--inference-steps", type=int, default=8, help="Diffusion steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="CFG scale.")
    parser.add_argument("--thinking", action="store_true", help="Enable LM reasoning.")
    parser.add_argument("--offload-to-cpu", action="store_true", help="Enable CPU offload.")
    return parser


def main() -> int:
    """Run one inference job with a trained LoRA adapter."""
    args = build_parser().parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    lora_path = Path(args.lora_path).resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoint_dir)

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

    load_msg = dit_handler.load_lora(str(lora_path))
    print(load_msg)
    scale_msg = dit_handler.set_lora_scale(args.lora_scale)
    print(scale_msg)
    use_msg = dit_handler.set_use_lora(True)
    print(use_msg)

    params = GenerationParams(
        task_type="text2music",
        caption=args.caption,
        lyrics=args.lyrics,
        instrumental=args.lyrics.strip().lower() in {"[instrumental]", "[inst]"},
        duration=args.duration,
        inference_steps=args.inference_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        thinking=args.thinking,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_lyrics=False,
        use_cot_language=False,
    )
    config = GenerationConfig(
        batch_size=args.batch_size,
        use_random_seed=args.seed < 0,
        audio_format="wav",
    )

    result = generate_music(dit_handler, llm_handler, params, config, save_dir=str(save_dir))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
