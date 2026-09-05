#!/usr/bin/env python3
"""Push a trained tokenizer directory to the HuggingFace Hub.

Reads HF_TOKEN from .env (or environment).  Creates the repo if it does not
exist, then uploads every file in the tokenizer directory.

Run:
    uv run python scripts/push_to_hub.py \\
        --tokenizer-dir artifacts/medical-bpe-pubmed-hf \\
        --repo-id YOUR_HF_USERNAME/my-medical-tokenizer

Example:
    uv run python scripts/push_to_hub.py \\
        --tokenizer-dir artifacts/medical-bpe-pubmed-hf \\
        --repo-id johndoe/medical-bpe-16k
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_token() -> str:
    """Return HF token from .env or environment variable HF_TOKEN."""
    # Try python-dotenv first (graceful if not installed)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass  # no dotenv — rely on env var set by user

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print(
            "ERROR: HF_TOKEN not found.\n"
            "  1. Copy .env.example → .env\n"
            "  2. Paste your token from https://huggingface.co/settings/tokens\n"
            "  3. Re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def push(tokenizer_dir: Path, repo_id: str, private: bool) -> None:
    """Create repo (if needed) and upload all files from tokenizer_dir."""
    from huggingface_hub import HfApi, create_repo

    token = load_token()
    api = HfApi(token=token)

    # Validate dir
    if not tokenizer_dir.is_dir():
        print(
            f"ERROR: tokenizer directory not found: {tokenizer_dir}\n"
            "  Did you run wrap_medical_tokenizer.py first?",
            file=sys.stderr,
        )
        sys.exit(1)

    files = list(tokenizer_dir.iterdir())
    if not files:
        print(f"ERROR: {tokenizer_dir} is empty.", file=sys.stderr)
        sys.exit(1)

    # Create repo
    visibility = "private" if private else "public"
    print(f"creating/verifying repo  {repo_id}  ({visibility}) ...")
    create_repo(repo_id, repo_type="model", private=private, token=token, exist_ok=True)

    # Upload tokenizer files
    print(f"uploading {len(files)} file(s) from {tokenizer_dir} ...")
    api.upload_folder(
        folder_path=str(tokenizer_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add tokenizer",
        token=token,
    )

    print(f"\n✅  done — https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=ROOT / "artifacts" / "medical-bpe-pubmed-hf",
        help="Path to the HuggingFace-format tokenizer directory (default: artifacts/medical-bpe-pubmed-hf)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo in the form USERNAME/REPO-NAME  e.g. johndoe/medical-bpe-16k",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create a private repo (default: public)",
    )
    args = parser.parse_args()
    push(args.tokenizer_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()
