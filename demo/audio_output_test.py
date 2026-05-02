#!/usr/bin/env python3
"""Minimal realtime audio output test for sounddevice devices."""

from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd


def list_output_devices() -> None:
    """Print available output devices with host API names."""
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device["max_output_channels"] <= 0:
            continue
        hostapi = hostapis[device["hostapi"]]["name"]
        default_sr = int(device["default_samplerate"])
        print(
            f"{index:>3} | {hostapi:<20} | {device['max_output_channels']} ch | "
            f"{default_sr} Hz | {device['name']}"
        )


def choose_output_device(prefer_name: str) -> int | None:
    """Choose an output device, preferring non-MME devices matching ``prefer_name``."""
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    candidates = []
    for index, device in enumerate(devices):
        if device["max_output_channels"] <= 0:
            continue
        hostapi = hostapis[device["hostapi"]]["name"]
        name = device["name"]
        if prefer_name.lower() in name.lower():
            priority = 0 if hostapi in {"Windows DirectSound", "Windows WDM-KS"} else 1
            candidates.append((priority, index, name, hostapi))

    if candidates:
        candidates.sort()
        return candidates[0][1]

    default_output = sd.default.device[1]
    return default_output if default_output is not None and default_output >= 0 else None


def make_test_block(sample_rate: int, seconds: float) -> np.ndarray:
    """Create a stereo test signal with tone bursts and silence gaps."""
    total = int(sample_rate * seconds)
    t = np.arange(total, dtype=np.float32) / sample_rate
    tone = 0.18 * np.sin(2 * np.pi * 440.0 * t)
    tone += 0.08 * np.sin(2 * np.pi * 880.0 * t)

    gate = ((t % 1.0) < 0.65).astype(np.float32)
    fade_len = min(int(sample_rate * 0.02), total // 2)
    envelope = np.ones(total, dtype=np.float32)
    envelope[:fade_len] = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    envelope[-fade_len:] = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    mono = tone * gate * envelope
    stereo = np.column_stack([mono, mono])
    return np.ascontiguousarray(stereo.astype(np.float32))


def play_test(device: int | None, sample_rate: int, seconds: float) -> None:
    """Play the test signal through a sounddevice output stream."""
    signal = make_test_block(sample_rate, seconds)
    info = sd.query_devices(device, "output") if device is not None else sd.query_devices(kind="output")
    print(f"Using output device: {device} | {info['name']}")
    print(f"Playing {seconds:.1f}s test tone at {sample_rate} Hz...")

    with sd.OutputStream(
        device=device,
        samplerate=sample_rate,
        channels=2,
        dtype="float32",
        blocksize=1024,
        latency="low",
    ) as stream:
        block_size = 1024
        for start in range(0, len(signal), block_size):
            stream.write(signal[start:start + block_size])
    time.sleep(0.1)
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description="Play a realtime sounddevice test tone.")
    parser.add_argument("--list-devices", action="store_true", help="List output devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Output device index")
    parser.add_argument("--prefer-name", default="FiiO", help="Device name substring for auto selection")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--seconds", type=float, default=5.0, help="Playback duration")
    return parser


def main() -> int:
    """Run the audio output test."""
    args = build_parser().parse_args()
    if args.list_devices:
        list_output_devices()
        return 0

    device = args.device
    if device is None:
        device = choose_output_device(args.prefer_name)
    play_test(device=device, sample_rate=args.sample_rate, seconds=args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
