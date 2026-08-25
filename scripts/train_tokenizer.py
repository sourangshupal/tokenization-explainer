#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on the FreeLaw corpus.

Usage:
    uv run python scripts/train_tokenizer.py [--vocab-size N] [--min-freq N]

Reads:  data/corpus/legal_shard_*.txt
Writes: models/legal-bpe-50k/tokenizer.json

Training is CPU-bound (Rust, all cores). GPUs are not used here.
Estimated time: 1–2 hours for 10 shards (~5 GB / ~500M tokens).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
MODEL_DIR = ROOT / "models" / "legal-bpe-50k"

SPECIAL_TOKENS = ["<|endoftext|>"]


def train(vocab_size: int, min_frequency: int, corpus_dir: Path, model_dir: Path) -> None:
    try:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import ByteLevel as ByteLevelProcessor
        from tokenizers.trainers import BpeTrainer
    except ImportError:
        log.error("tokenizers not installed — run: uv sync")
        sys.exit(1)

    shards = sorted(corpus_dir.glob("legal_shard_*.txt"))
    if not shards:
        log.error(
            "No corpus shards found in %s. Run scripts/build_corpus.py first.", corpus_dir
        )
        sys.exit(1)

    total_mb = sum(p.stat().st_size for p in shards) / 1_048_576
    log.info(
        "Found %d shard(s) — %.1f MB total. Training vocab_size=%d, min_freq=%d.",
        len(shards),
        total_mb,
        vocab_size,
        min_frequency,
    )

    tok = Tokenizer(BPE(unk_token=None))

    # ByteLevel pre-tokenizer encodes each byte as a printable character, giving
    # 256 base tokens and guaranteeing zero UNK. The internal regex (use_regex=True,
    # the default) already applies GPT-2-style whitespace/word-boundary splitting
    # before BPE merges, so no separate Split pre-tokenizer is needed.
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        # Force all 256 byte-level characters into the base vocabulary.
        # Without this, bytes absent from the training corpus are dropped,
        # breaking round-trips and producing UNK for unseen characters.
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    shard_paths = [str(p) for p in shards]
    log.info("Starting BPE merge training… (this is CPU-bound, ~1–2 h for 5 GB)")
    t0 = time.monotonic()
    tok.train(files=shard_paths, trainer=trainer)
    elapsed = time.monotonic() - t0
    log.info("Training complete in %.1f min.", elapsed / 60)

    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "tokenizer.json"
    tok.save(str(out_path))
    log.info("Saved tokenizer → %s  (vocab size: %d)", out_path, tok.get_vocab_size())

    # Smoke test: round-trip a legal sentence.
    sample = (
        "The court hereby grants the motion for summary judgment notwithstanding "
        "the defendant's objection pursuant to 28 U.S.C. § 1331."
    )
    enc = tok.encode(sample)
    decoded = tok.decode(enc.ids)
    assert decoded == sample, f"Round-trip failed!\n  in:  {sample!r}\n  out: {decoded!r}"
    log.info("Smoke test passed. '%s' → %d tokens.", sample[:60], len(enc.ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train byte-level BPE tokenizer on FreeLaw")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50_257,
        help="Target vocabulary size (default 50,257 — matches GPT-2)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum pair frequency to merge (default 2)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPUS_DIR,
        help="Directory containing legal_shard_*.txt files",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Output directory for tokenizer.json",
    )
    args = parser.parse_args()
    train(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        corpus_dir=args.corpus_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
