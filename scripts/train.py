#!/usr/bin/env python3
"""Occupy CUDA GPU memory and compute with configurable load."""

import argparse
import time

import torch


DEFAULT_MATMUL_SIZE = 4096
DEFAULT_RESERVE_RATIO = 0.7


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GPU load script."""
    parser = argparse.ArgumentParser(
        description="Reserve CUDA VRAM and keep the GPU busy with matmul workloads."
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--reserve-ratio",
        type=float,
        default=DEFAULT_RESERVE_RATIO,
        help="Fraction of total VRAM to reserve with a tensor.",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=DEFAULT_MATMUL_SIZE,
        help="Square matrix size used for repeated matmul.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Seconds to sleep between iterations.",
    )
    return parser.parse_args()


def resolve_target_bytes(total_bytes: int, reserve_ratio: float) -> int:
    """Clamp reserve ratio and convert it to a target VRAM size in bytes."""
    safe_ratio = min(max(reserve_ratio, 0.05), 0.95)
    return int(total_bytes * safe_ratio)


def reserve_memory(device: str, target_bytes: int) -> torch.Tensor | None:
    """Allocate a float16 tensor close to the requested VRAM target."""
    element_size = torch.tensor([], dtype=torch.float16).element_size()
    element_count = max(target_bytes // element_size, 1)
    try:
        return torch.empty(element_count, device=device, dtype=torch.float16)
    except RuntimeError as exc:
        print(f"warning: reserve allocation failed, continuing with compute only: {exc}")
        torch.cuda.empty_cache()
        return None


def print_status(device_index: int, reserved: torch.Tensor | None) -> None:
    """Print the current device and memory reservation status."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    reserved_gb = 0.0 if reserved is None else reserved.numel() * reserved.element_size() / 1024**3
    print(f"device: cuda:{device_index}")
    print(f"total vram: {total_bytes / 1024**3:.2f} GB")
    print(f"free vram after setup: {free_bytes / 1024**3:.2f} GB")
    print(f"reserved tensor: {reserved_gb:.2f} GB")
    print("running matmul loop, press Ctrl+C to stop")


def main() -> None:
    """Run the CUDA load loop until interrupted."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
    target_bytes = min(resolve_target_bytes(total_bytes, args.reserve_ratio), free_bytes)
    reserved = reserve_memory(device, target_bytes)

    size = max(args.matmul_size, 512)
    left = torch.randn((size, size), device=device, dtype=torch.float16)
    right = torch.randn((size, size), device=device, dtype=torch.float16)

    print_status(args.device, reserved)

    iterations = 0
    started_at = time.time()
    try:
        while True:
            result = left @ right
            result = torch.relu(result)
            left.copy_(result)
            iterations += 1
            if iterations % 25 == 0:
                torch.cuda.synchronize(args.device)
                elapsed = time.time() - started_at
                print(f"iterations: {iterations}, elapsed: {elapsed:.1f}s")
            if args.sleep > 0:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        torch.cuda.synchronize(args.device)
        elapsed = time.time() - started_at
        print(f"stopped after {iterations} iterations in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
