#!/usr/bin/env python3
"""
Wrap the trained tokenizer.json into a HuggingFace PreTrainedTokenizerFast
and call save_pretrained() so the output directory is Hub-ready.

Usage:
    uv run python scripts/wrap_tokenizer.py

Reads:  models/legal-bpe-50k/tokenizer.json
Writes: models/legal-bpe-50k/
    tokenizer.json            (already there, unchanged)
    tokenizer_config.json     (new)
    special_tokens_map.json   (new)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "legal-bpe-50k"

EOT = "<|endoftext|>"


def wrap(model_dir: Path) -> None:
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast  # type: ignore[import-untyped]
    except ImportError:
        log.error("transformers not installed — run: uv sync")
        sys.exit(1)

    tok_file = model_dir / "tokenizer.json"
    if not tok_file.exists():
        log.error(
            "tokenizer.json not found at %s. Run scripts/train_tokenizer.py first.",
            tok_file,
        )
        sys.exit(1)

    log.info("Wrapping %s into PreTrainedTokenizerFast…", tok_file)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file),
        bos_token=EOT,
        eos_token=EOT,
        unk_token=None,    # byte-level BPE: UNK is impossible
        pad_token=EOT,
        model_max_length=int(1e30),  # no hard limit — caller decides
    )

    fast_tok.save_pretrained(str(model_dir))
    log.info("save_pretrained complete → %s", model_dir)
    log.info("Files written: %s", sorted(p.name for p in model_dir.iterdir()))

    # Verify round-trip load.
    reloaded = AutoTokenizer.from_pretrained(str(model_dir))
    sample = "The court grants certiorari notwithstanding counsel's objection."
    ids = reloaded.encode(sample)
    decoded = reloaded.decode(ids, skip_special_tokens=False)
    assert decoded == sample, f"Reload round-trip failed: {decoded!r}"
    log.info(
        "Reload verification passed. vocab_size=%d, sample tokens=%d.",
        reloaded.vocab_size,
        len(ids),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Wrap tokenizer.json → HF save_pretrained dir")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory containing tokenizer.json (output of train_tokenizer.py)",
    )
    args = parser.parse_args()
    wrap(args.model_dir)


if __name__ == "__main__":
    main()
