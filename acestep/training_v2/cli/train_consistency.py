"""
consistency subcommand -- Streaming Consistency Distillation (8-step to 1-step).

Distills a Teacher model (frozen) into a Student model (trainable adapter)
using a 1:3 ratio streaming window (Prefix KV-Cache + Prediction Chunk).
"""

from __future__ import annotations

import argparse
import gc
import sys
import torch

from acestep.training_v2.cli.common import build_configs
from acestep.training_v2.trainer_streaming_consistency import StreamingConsistencyTrainer


def _cleanup_gpu() -> None:
    """Release GPU memory so the process can safely reuse it."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, NameError):
        pass


def run_consistency(args: argparse.Namespace) -> int:
    """Execute the consistency distillation training subcommand.

    Returns 0 on success, non-zero on failure.
    """
    # -- UI setup -------------------------------------------------------------
    from acestep.training_v2.ui import set_plain_mode
    from acestep.training_v2.ui.banner import show_banner
    from acestep.training_v2.ui.config_panel import show_config, confirm_start
    from acestep.training_v2.ui.errors import handle_error, show_info
    from acestep.training_v2.ui.progress import track_training
    from acestep.training_v2.ui.summary import show_summary

    if getattr(args, "plain", False):
        set_plain_mode(True)

    # -- Matmul precision -----------------------------------------------------
    torch.set_float32_matmul_precision("medium")

    # -- Build V2 config objects from CLI args --------------------------------
    adapter_cfg, train_cfg = build_configs(args)

    # -- Banner ---------------------------------------------------------------
    if not getattr(args, "_from_wizard", False):
        show_banner(
            subcommand="consistency",
            device=train_cfg.device,
            precision=train_cfg.precision,
        )

    # -- Config summary & confirmation ----------------------------------------
    show_config(adapter_cfg, train_cfg, subcommand="consistency")
    skip_confirm = getattr(args, "yes", False)
    if not confirm_start(skip=skip_confirm):
        return 0

    trainer = None
    try:
        # -- Distillation Training ---------------------------------------------
        try:
            trainer = StreamingConsistencyTrainer(adapter_cfg, train_cfg)

            stats = track_training(
                training_iter=trainer.train(),
                max_epochs=train_cfg.max_epochs,
                device=train_cfg.device,
            )

            # -- Summary ------------------------------------------------------
            show_summary(
                stats=stats,
                output_dir=train_cfg.output_dir,
                log_dir=str(train_cfg.effective_log_dir),
            )
        except KeyboardInterrupt:
            show_info("Distillation interrupted by user (Ctrl+C)")
            return 130
        except Exception as exc:
            handle_error(exc, context="Consistency Distillation", show_traceback=True)
            return 1

        return 0
    finally:
        # Explicitly release GPU memory
        del trainer
        _cleanup_gpu()


def main() -> int:
    """Standalone CLI entry point for the ``consistency`` training subcommand.

    Returns:
        Integer exit code: 0 on success, non-zero on failure.
    """
    from acestep.training_v2.cli.args import build_root_parser
    from acestep.training_v2.cli.validation import validate_paths

    # We use the root parser but enforce the consistency subcommand logic
    parser = build_root_parser()
    args = parser.parse_args()
    
    # If called directly from this script, subcommand might be None if not passed
    if args.subcommand != "consistency":
        args.subcommand = "consistency"

    if not args.data_free and not args.dataset_dir:
        parser.error("--dataset-dir is required unless --data-free is enabled")
    if args.data_free and not (args.prompt_file or args.dataset_json):
        parser.error("--data-free requires --prompt-file or --dataset-json")
    if not args.output_dir:
        parser.error("--output-dir is required for training")

    if not validate_paths(args):
        return 1

    return run_consistency(args)


if __name__ == "__main__":
    sys.exit(main())
