#!/usr/bin/env python3
"""Train a class-scale byte-level BPE on the authored medical corpus.

Run: uv run python scripts/train_medical_tokenizer.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "medical_corpus.txt"
DEFAULT_OUT = ROOT / "artifacts" / "medical-bpe" / "tokenizer.json"
SPECIAL_TOKENS = ["<|endoftext|>", "<pad>"]


def find_root() -> Path:
    """Repo root (directory that contains data/medical_corpus.txt)."""
    return ROOT


def iter_corpus_batches(path: Path, batch_size: int = 64) -> Iterator[list[str]]:
    """Yield batches of non-empty lines for train_from_iterator.

    Plain text: one document per line. JSONL: uses the ``text`` field when present.
    """
    batch: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            text = raw
            if raw.startswith("{"):
                try:
                    import json

                    obj = json.loads(raw)
                    if isinstance(obj, dict) and obj.get("text"):
                        text = str(obj["text"]).strip()
                except json.JSONDecodeError:
                    text = raw
            if not text:
                continue
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def count_lines(path: Path) -> int:
    """Count non-empty lines (progress length for the trainer)."""
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def build_byte_level_bpe() -> Tokenizer:
    """Empty byte-level BPE with GPT-style pretokenizer/decoder."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    return tokenizer


def train_byte_level_bpe(
    corpus: Path,
    output: Path,
    vocab_size: int = 16000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train BPE and save tokenizer.json. Returns the trained Tokenizer."""
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")
    tokenizer = build_byte_level_bpe()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    length = count_lines(corpus)
    tokenizer.train_from_iterator(
        iter_corpus_batches(corpus),
        trainer=trainer,
        length=length,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="UTF-8 text, one document per line",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    tokenizer = train_byte_level_bpe(
        args.corpus.expanduser().resolve(),
        args.out.expanduser().resolve(),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"saved {args.out}  vocab={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
