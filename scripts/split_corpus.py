#!/usr/bin/env python3
"""Split data/pubmed_sample.jsonl into disjoint train / held-out files.

Default: 45 000 train, 5 000 held-out  (requires data/pubmed_sample.jsonl).
Shuffle is deterministic (seed 42 on line indices) — same split every run.

Run: uv run python scripts/split_corpus.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_jsonl(
    src: Path,
    train_out: Path,
    heldout_out: Path,
    heldout_n: int = 5000,
    seed: int = 42,
) -> tuple[int, int]:
    """Read src, shuffle with seed, write first (total - heldout_n) to train, rest to heldout."""
    if not src.is_file():
        raise FileNotFoundError(f"source JSONL not found: {src}")
    lines = src.read_bytes().splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    n_total = len(lines)
    n_heldout = min(heldout_n, n_total)
    n_train = n_total - n_heldout
    train_out.parent.mkdir(parents=True, exist_ok=True)
    heldout_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_bytes(b"\n".join(lines[:n_train]) + b"\n")
    heldout_out.write_bytes(b"\n".join(lines[n_train:]) + b"\n")
    return n_train, n_heldout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "pubmed_sample.jsonl")
    parser.add_argument("--train-out", type=Path, default=ROOT / "data" / "pubmed_train.jsonl")
    parser.add_argument("--heldout-out", type=Path, default=ROOT / "data" / "pubmed_heldout.jsonl")
    parser.add_argument("--heldout-n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    n_train, n_heldout = split_jsonl(
        args.src, args.train_out, args.heldout_out, args.heldout_n, args.seed
    )
    print(f"train: {n_train} docs  →  {args.train_out}")
    print(f"heldout: {n_heldout} docs  →  {args.heldout_out}")
    print("Train with: uv run python scripts/train_medical_tokenizer.py --corpus", args.train_out)


if __name__ == "__main__":
    main()
