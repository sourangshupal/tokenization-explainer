#!/usr/bin/env python3
"""Wrap tokenizer.json as a HuggingFace PreTrainedTokenizerFast directory.

Run: uv run python scripts/wrap_medical_tokenizer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "artifacts" / "medical-bpe" / "tokenizer.json"
DEFAULT_OUT = ROOT / "artifacts" / "medical-bpe-hf"


def wrap_tokenizer(
    tokenizer_json: Path,
    output_dir: Path,
    eos_token: str = "<|endoftext|>",
    pad_token: str = "<pad>",
) -> PreTrainedTokenizerFast:
    """Load tokenizer.json and save_pretrained to output_dir."""
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_json}")
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        bos_token=None,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tok.save_pretrained(output_dir)
    return hf_tok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    hf_tok = wrap_tokenizer(
        args.tokenizer_json.expanduser().resolve(),
        args.out.expanduser().resolve(),
    )
    print(f"saved {args.out}  size={len(hf_tok)}")


if __name__ == "__main__":
    main()
