#!/usr/bin/env python3
"""Benchmark standard and streaming inference latency for trained LoRA adapters."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

from streaming_infer_lora import _build_text_payload, _prepare_condition_tensors


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for latency benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark ACE-Step LoRA latency for standard and streaming inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["standard", "streaming"], required=True, help="Benchmark mode.")
    parser.add_argument("--csv-path", required=True, help="CSV file with prompts.")
    parser.add_argument("--output-csv", required=True, help="CSV file to write metrics to.")
    parser.add_argument("--lora-path", default="", help="LoRA adapter directory.")
    parser.add_argument("--experiment-name", required=True, help="Experiment label written to CSV.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Base checkpoint root.")
    parser.add_argument("--config-path", default="acestep-v15-xl-turbo", help="Base DiT model name or path.")
    parser.add_argument("--id-column", default="caption_id", help="CSV id column.")
    parser.add_argument("--caption-column", default="caption", help="CSV caption column.")
    parser.add_argument("--lyrics", default="[Instrumental]", help="Lyrics text or [Instrumental].")
    parser.add_argument("--device", default="cuda", help="Device string.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--limit", type=int, default=10, help="Max prompts to benchmark.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per prompt.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed warmup runs per prompt.")
    parser.add_argument("--audio-format", default="wav", choices=["wav", "mp3", "flac"], help="Audio format.")
    parser.add_argument("--offload-to-cpu", action="store_true", help="Enable CPU offload.")
    parser.add_argument("--thinking", action="store_true", help="Enable LM reasoning for standard inference.")
    parser.add_argument("--duration", type=float, default=30.0, help="Target duration for standard inference.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for standard inference.")
    parser.add_argument("--inference-steps", type=int, default=1, help="Inference steps for standard inference.")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="CFG scale for standard inference.")
    parser.add_argument("--warmup-seconds", type=float, default=30.0, help="Warmup seconds for streaming inference.")
    parser.add_argument(
        "--prediction-seconds", type=float, default=30.0, help="Prediction chunk seconds for streaming inference."
    )
    parser.add_argument(
        "--max-distill-seconds", type=float, default=150.0, help="Total predicted seconds for streaming inference."
    )
    parser.add_argument("--max-distill-chunks", type=int, default=5, help="Max streaming chunks.")
    parser.add_argument("--use-tiled-decode", action="store_true", help="Use tiled VAE decode in streaming mode.")
    return parser


def _sync_cuda_if_needed(device: str) -> None:
    """Synchronize CUDA to make latency measurements accurate."""
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize()


def init_runtime(args: argparse.Namespace) -> tuple[AceStepHandler, LLMHandler]:
    """Initialize ACE-Step runtime and load the requested LoRA."""
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if args.lora_path:
        lora_path = Path(args.lora_path).resolve()
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
    if args.lora_path:
        print(dit_handler.load_lora(str(lora_path)))
        print(dit_handler.set_lora_scale(1.0))
        print(dit_handler.set_use_lora(True))
    else:
        print("[System] No LoRA path provided. Running base model only.")
        dit_handler.set_use_lora(False)
    return dit_handler, llm_handler


def _timed_streaming_rollout(
    dit_handler: AceStepHandler,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    warmup_frames: int,
    prediction_frames: int,
    num_chunks: int,
    seed: int,
) -> tuple[torch.Tensor, list[float]]:
    """Run streaming rollout and return chunk-level timings in seconds."""
    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    pkv = None
    generated_chunks: list[torch.Tensor] = []
    chunk_times: list[float] = []
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

        _sync_cuda_if_needed(str(device))
        chunk_start = time.perf_counter()
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
        _sync_cuda_if_needed(str(device))
        chunk_times.append(time.perf_counter() - chunk_start)
        generated_chunks.append(pred_x0)

    return torch.cat(generated_chunks, dim=1), chunk_times


def benchmark_standard(
    dit_handler: AceStepHandler,
    llm_handler: LLMHandler,
    args: argparse.Namespace,
    sample_id: str,
    caption: str,
) -> list[dict[str, Any]]:
    """Benchmark standard inference for one prompt."""
    rows: list[dict[str, Any]] = []
    params = GenerationParams(
        task_type="text2music",
        caption=caption,
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
        audio_format=args.audio_format,
    )

    for run_idx in range(args.warmup_runs + args.repeats):
        is_warmup = run_idx < args.warmup_runs
        _sync_cuda_if_needed(args.device)
        start = time.perf_counter()
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=None)
        _sync_cuda_if_needed(args.device)
        wall_time = time.perf_counter() - start
        if is_warmup:
            continue

        time_costs = (result.extra_outputs or {}).get("time_costs", {}) if result else {}
        audio_tensor = result.audios[0]["tensor"]
        sample_rate = int(result.audios[0]["sample_rate"])
        audio_duration = audio_tensor.shape[-1] / sample_rate
        rows.append(
            {
                "experiment": args.experiment_name,
                "mode": "standard",
                "sample_id": sample_id,
                "run_idx": run_idx - args.warmup_runs,
                "inference_steps": args.inference_steps,
                "max_distill_chunks": "",
                "wall_time_sec": wall_time,
                "pipeline_total_time_sec": time_costs.get("pipeline_total_time", ""),
                "dit_total_time_sec": time_costs.get("dit_total_time_cost", ""),
                "vae_decode_time_sec": time_costs.get("dit_vae_decode_time_cost", ""),
                "audio_duration_sec": audio_duration,
                "rtf_wall": wall_time / audio_duration,
                "rtf_pipeline": (
                    time_costs.get("pipeline_total_time", 0.0) / audio_duration if time_costs else ""
                ),
            }
        )
    return rows


def benchmark_streaming(
    dit_handler: AceStepHandler,
    args: argparse.Namespace,
    sample_id: str,
    caption: str,
) -> list[dict[str, Any]]:
    """Benchmark streaming inference for one prompt."""
    rows: list[dict[str, Any]] = []
    fps = 50
    warmup_frames = int(args.warmup_seconds * fps)
    prediction_frames = int(args.prediction_seconds * fps)
    max_distill_frames = int(args.max_distill_seconds * fps)
    num_chunks = min(args.max_distill_chunks, max(1, max_distill_frames // prediction_frames))
    total_frames = warmup_frames + num_chunks * prediction_frames

    for run_idx in range(args.warmup_runs + args.repeats):
        is_warmup = run_idx < args.warmup_runs

        _sync_cuda_if_needed(args.device)
        conditioning_start = time.perf_counter()
        text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = _build_text_payload(
            dit_handler, caption, args.lyrics
        )
        encoder_hidden_states, encoder_attention_mask, context_latents = _prepare_condition_tensors(
            dit_handler,
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            total_frames,
        )
        _sync_cuda_if_needed(args.device)
        conditioning_time = time.perf_counter() - conditioning_start

        _sync_cuda_if_needed(args.device)
        diffusion_start = time.perf_counter()
        pred_latents, chunk_times = _timed_streaming_rollout(
            dit_handler=dit_handler,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            warmup_frames=warmup_frames,
            prediction_frames=prediction_frames,
            num_chunks=num_chunks,
            seed=args.seed,
        )
        _sync_cuda_if_needed(args.device)
        diffusion_time = time.perf_counter() - diffusion_start

        _sync_cuda_if_needed(args.device)
        decode_start = time.perf_counter()
        pred_wavs, _pred_latents_cpu, time_costs = dit_handler._decode_generate_music_pred_latents(
            pred_latents=pred_latents,
            progress=None,
            use_tiled_decode=args.use_tiled_decode,
            time_costs={"total_time_cost": diffusion_time, "diffusion_per_step_time_cost": 0.0},
        )
        _sync_cuda_if_needed(args.device)
        decode_time = time.perf_counter() - decode_start
        total_time = conditioning_time + diffusion_time + decode_time

        if is_warmup:
            continue

        audio_duration = pred_wavs[0].shape[-1] / dit_handler.sample_rate
        warmup_chunk_time = chunk_times[0] if chunk_times else 0.0
        prediction_chunk_times = chunk_times[1:] if len(chunk_times) > 1 else []
        avg_prediction_chunk_time = (
            sum(prediction_chunk_times) / len(prediction_chunk_times) if prediction_chunk_times else 0.0
        )
        rows.append(
            {
                "experiment": args.experiment_name,
                "mode": "streaming",
                "sample_id": sample_id,
                "run_idx": run_idx - args.warmup_runs,
                "inference_steps": 1,
                "max_distill_chunks": args.max_distill_chunks,
                "conditioning_time_sec": conditioning_time,
                "diffusion_time_sec": diffusion_time,
                "decode_time_sec": decode_time,
                "total_time_sec": total_time,
                "vae_decode_time_sec": time_costs.get("vae_decode_time_cost", decode_time),
                "audio_duration_sec": audio_duration,
                "rtf_total": total_time / audio_duration,
                "rtf_diffusion": diffusion_time / audio_duration,
                "first_chunk_latency_sec": warmup_chunk_time,
                "avg_prediction_chunk_latency_sec": avg_prediction_chunk_time,
                "prediction_chunk_rtf": (
                    avg_prediction_chunk_time / args.prediction_seconds if args.prediction_seconds > 0 else ""
                ),
            }
        )
    return rows


def main() -> int:
    """Run latency benchmark and write CSV rows."""
    args = build_parser().parse_args()
    df = pd.read_csv(args.csv_path).head(args.limit)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dit_handler, llm_handler = init_runtime(args)

    fieldnames: list[str] = []
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sample_id = str(row[args.id_column])
        caption = str(row[args.caption_column])
        if args.mode == "standard":
            rows.extend(benchmark_standard(dit_handler, llm_handler, args, sample_id, caption))
        else:
            rows.extend(benchmark_streaming(dit_handler, args, sample_id, caption))

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} benchmark rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
