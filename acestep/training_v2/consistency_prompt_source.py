"""Prompt-only sample loading for streaming consistency distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader, Dataset

from acestep.training_v2.preprocess_discovery import load_dataset_metadata, select_genre_indices
from acestep.training_v2.preprocess_prompt import build_simple_prompt


def _normalize_prompt_entry(entry: Any) -> Optional[Dict[str, Any]]:
    """Normalize one prompt entry into a common internal representation."""
    if isinstance(entry, str):
        prompt = entry.strip()
        if not prompt:
            return None
        return {
            "prompt": prompt,
            "lyrics": "[Instrumental]",
            "metadata": {"caption": prompt},
        }

    if not isinstance(entry, dict):
        return None

    prompt = (entry.get("prompt") or entry.get("caption") or entry.get("text") or "").strip()
    if not prompt:
        prompt = build_simple_prompt(entry).strip()
    if not prompt:
        return None

    return {
        "prompt": prompt,
        "lyrics": entry.get("lyrics", "[Instrumental]") or "[Instrumental]",
        "metadata": entry,
    }


def _load_prompt_entries_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load prompt entries from JSON or ACE-Step dataset JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        return [sample for entry in raw if (sample := _normalize_prompt_entry(entry)) is not None]

    if not isinstance(raw, dict):
        return []

    samples = raw.get("samples")
    if isinstance(samples, list):
        dataset_meta = load_dataset_metadata(str(path))
        genre_indices = select_genre_indices(len(samples), dataset_meta["genre_ratio"])
        normalized: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            prompt = build_simple_prompt(
                sample,
                tag_position=dataset_meta["tag_position"],
                use_genre=idx in genre_indices,
            ).strip()
            if not prompt:
                continue
            normalized.append(
                {
                    "prompt": prompt,
                    "lyrics": sample.get("lyrics", "[Instrumental]") or "[Instrumental]",
                    "metadata": sample,
                }
            )
        return normalized

    sample = _normalize_prompt_entry(raw)
    return [sample] if sample is not None else []


def load_prompt_samples(
    prompt_file: Optional[str] = None,
    dataset_json: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load prompt-only training samples from the configured source."""
    source = prompt_file or dataset_json
    if not source:
        return []

    path = Path(source)
    if path.is_dir():
        samples: List[Dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in {".txt", ".json", ".jsonl"}:
                samples.extend(load_prompt_samples(prompt_file=str(child)))
        return samples

    if not path.is_file():
        return []

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [
            sample
            for line in path.read_text(encoding="utf-8").splitlines()
            if (sample := _normalize_prompt_entry(line)) is not None
        ]

    if suffix == ".jsonl":
        samples: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            sample = _normalize_prompt_entry(json.loads(line))
            if sample is not None:
                samples.append(sample)
        return samples

    if suffix == ".json":
        return _load_prompt_entries_from_json(path)

    raise ValueError(f"Unsupported prompt source format: {path}")


class PromptOnlyDataset(Dataset):
    """Dataset that yields prompt-only samples for synthetic distillation."""

    def __init__(
        self,
        prompt_file: Optional[str] = None,
        dataset_json: Optional[str] = None,
    ) -> None:
        self.samples = load_prompt_samples(prompt_file=prompt_file, dataset_json=dataset_json)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_prompt_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate prompt-only samples into a compact batch dict."""
    return {
        "prompts": [sample["prompt"] for sample in batch],
        "lyrics": [sample.get("lyrics", "[Instrumental]") for sample in batch],
        "metadata": [sample.get("metadata", {}) for sample in batch],
    }


class PromptOnlyDataModule:
    """Minimal data module for prompt-only consistency distillation."""

    def __init__(
        self,
        prompt_file: Optional[str],
        dataset_json: Optional[str],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        self.prompt_file = prompt_file
        self.dataset_json = dataset_json
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset: Optional[PromptOnlyDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the prompt-only training dataset."""
        if stage == "fit" or stage is None:
            self.train_dataset = PromptOnlyDataset(
                prompt_file=self.prompt_file,
                dataset_json=self.dataset_json,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the prompt-only training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_prompt_batch,
            drop_last=False,
        )
