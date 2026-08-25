#!/usr/bin/env python3
"""
Push the trained tokenizer to the Hugging Face Hub.

Usage:
    uv run python scripts/push_hub.py --repo YOUR_USERNAME/legal-bpe-50k

The script:
  1. Logs in (reads ~/.huggingface token or prompts).
  2. Reloads the tokenizer from models/legal-bpe-50k/.
  3. Pushes all three Hub files:
       tokenizer.json, tokenizer_config.json, special_tokens_map.json
  4. Prints the canonical one-liner for users to load it.

After push anyone can load with:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("YOUR_USERNAME/legal-bpe-50k")
"""

from __future__ import annotations

import argparse
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


def push(repo_id: str, model_dir: Path, private: bool) -> None:
    try:
        from huggingface_hub import login  # type: ignore[import-untyped]
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
    except ImportError:
        log.error("huggingface_hub / transformers not installed — run: uv sync")
        sys.exit(1)

    config = model_dir / "tokenizer_config.json"
    if not config.exists():
        log.error(
            "tokenizer_config.json not found in %s. "
            "Run scripts/wrap_tokenizer.py first.",
            model_dir,
        )
        sys.exit(1)

    log.info("Logging in to Hugging Face Hub…")
    login()  # reads ~/.huggingface/token or prompts for browser auth

    log.info("Loading tokenizer from %s…", model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    log.info("Vocab size: %d", tok.vocab_size)

    log.info("Pushing to Hub → %s (private=%s)…", repo_id, private)
    tok.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Add legal-bpe-50k: byte-level BPE trained on FreeLaw (pile-of-law)",
    )

    url = f"https://huggingface.co/{repo_id}"
    log.info("Done! Tokenizer live at %s", url)
    print("\n" + "=" * 60)
    print("Load your tokenizer with:")
    print(f"  from transformers import AutoTokenizer")
    print(f'  tok = AutoTokenizer.from_pretrained("{repo_id}")')
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push tokenizer to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help='Hub repo id, e.g. "myname/legal-bpe-50k"',
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Local save_pretrained directory (default: models/legal-bpe-50k/)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create a private repository (default: public)",
    )
    args = parser.parse_args()
    push(repo_id=args.repo, model_dir=args.model_dir, private=args.private)


if __name__ == "__main__":
    main()
