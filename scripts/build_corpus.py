#!/usr/bin/env python3
"""
Download and prepare the FreeLaw corpus for BPE tokenizer training.

Usage:
    uv run python scripts/build_corpus.py [--max-shards N] [--shard-size-mb N]

Output:
    data/corpus/legal_shard_0000.txt  (gitignored)
    data/corpus/legal_shard_0001.txt
    ...
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"

# Patterns that flag pure boilerplate (court headers, page markers).
_BOILERPLATE = re.compile(
    r"^(IN THE UNITED STATES|UNITED STATES DISTRICT|UNITED STATES COURT OF APPEALS"
    r"|SUPREME COURT OF|BEFORE THE|Page \d+|\*{3,}|_{3,}|-{3,})",
    re.IGNORECASE | re.MULTILINE,
)

_BLANK_LINE_RUN = re.compile(r"\n{3,}")


def _clean(text: str) -> str | None:
    """Minimal cleaning pass. Returns None if document is unusable."""
    text = text.strip()
    if len(text) < 200:
        return None
    # Collapse runs of blank lines.
    text = _BLANK_LINE_RUN.sub("\n\n", text)
    # Drop documents that are >50% boilerplate header lines.
    lines = text.splitlines()
    boilerplate_lines = sum(1 for ln in lines if _BOILERPLATE.match(ln))
    if lines and boilerplate_lines / len(lines) > 0.5:
        return None
    return text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def build_corpus(max_shards: int, shard_size_bytes: int, streaming: bool = True) -> None:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        log.error("datasets not installed — run: uv sync")
        sys.exit(1)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading pile-of-law FreeLaw split (streaming=%s)…", streaming)
    ds = load_dataset(
        "pile-of-law/pile-of-law",
        "FreeLaw",
        split="train",
        streaming=streaming,
        trust_remote_code=True,
    )

    seen: set[str] = set()
    shard_idx = 0
    shard_bytes = 0
    docs_written = 0
    docs_skipped = 0

    shard_path = CORPUS_DIR / f"legal_shard_{shard_idx:04d}.txt"
    shard_fh = shard_path.open("w", encoding="utf-8")
    log.info("Writing shard 0 → %s", shard_path)

    with tqdm(desc="documents", unit="doc") as pbar:
        for row in ds:
            text_raw: str = row.get("text") or row.get("contents") or ""
            text = _clean(text_raw)
            if text is None:
                docs_skipped += 1
                pbar.update(1)
                continue

            digest = _sha256(text[:2000])  # cheap prefix hash for dedup
            if digest in seen:
                docs_skipped += 1
                pbar.update(1)
                continue
            seen.add(digest)

            encoded = text.encode("utf-8")
            shard_fh.write(text)
            shard_fh.write("\n\n")
            shard_bytes += len(encoded)
            docs_written += 1
            pbar.update(1)
            pbar.set_postfix(shards=shard_idx + 1, mb=f"{shard_bytes // 1_048_576}")

            if shard_bytes >= shard_size_bytes:
                shard_fh.close()
                log.info(
                    "Shard %d complete: %.1f MB, %d docs",
                    shard_idx,
                    shard_bytes / 1_048_576,
                    docs_written,
                )
                shard_idx += 1
                if shard_idx >= max_shards:
                    log.info("Reached max_shards=%d — stopping.", max_shards)
                    break
                shard_path = CORPUS_DIR / f"legal_shard_{shard_idx:04d}.txt"
                shard_fh = shard_path.open("w", encoding="utf-8")
                log.info("Writing shard %d → %s", shard_idx, shard_path)
                seen.clear()  # reset dedup per shard to keep memory bounded
                shard_bytes = 0
                docs_written = 0

    if not shard_fh.closed:
        shard_fh.close()

    total_shards = shard_idx + 1
    log.info(
        "Done. %d shards written to %s. %d docs skipped (short/dup/boilerplate).",
        total_shards,
        CORPUS_DIR,
        docs_skipped,
    )
    shards = sorted(CORPUS_DIR.glob("legal_shard_*.txt"))
    total_mb = sum(p.stat().st_size for p in shards) / 1_048_576
    log.info("Total corpus size: %.1f MB across %d files.", total_mb, len(shards))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FreeLaw training corpus")
    parser.add_argument(
        "--max-shards",
        type=int,
        default=10,
        help="Maximum number of shards to write (default 10, ~5 GB total)",
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=500,
        help="Target shard size in MB (default 500)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download full dataset before iterating (slower startup)",
    )
    args = parser.parse_args()
    build_corpus(
        max_shards=args.max_shards,
        shard_size_bytes=args.shard_size_mb * 1_048_576,
        streaming=not args.no_streaming,
    )


if __name__ == "__main__":
    main()
