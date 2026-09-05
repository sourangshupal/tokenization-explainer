#!/usr/bin/env python3
"""Download wikitext-103 paragraphs as the general-English control corpus (gitignored).

Streams Salesforce/wikitext (parquet — no dataset-script issue) and writes:
  data/general_train.jsonl  (default 45 000 paragraphs)
  data/general_heldout.jsonl (default  5 000 paragraphs)

Run: uv run python scripts/download_general_sample.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
HF_DATASET_ID = "Salesforce/wikitext"
HF_CONFIG = "wikitext-103-raw-v1"
MIN_CHARS = 200  # skip stubs / section headings


def stream_wikitext(total: int) -> Iterator[dict[str, str]]:
    """Yield {text: ...} dicts from wikitext-103 train split.

    Uses streaming=False (full Parquet download) to avoid [Errno 9] Bad file
    descriptor errors that occur when fsspec tries to seek on a non-seekable
    network stream in some environments.  The Parquet files are cached by
    datasets so subsequent runs are instant.
    """
    from datasets import load_dataset

    print(f"downloading {HF_DATASET_ID}/{HF_CONFIG} train split (cached after first run)...", flush=True)
    ds = load_dataset(HF_DATASET_ID, HF_CONFIG, split="train", streaming=False)
    yielded = 0
    scanned = 0
    for row in ds:
        scanned += 1
        text = (row.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            continue
        yielded += 1
        if yielded % 1000 == 0:
            print(f"kept {yielded}/{total} paragraphs (scanned {scanned})", flush=True)
        yield {"text": text}
        if yielded >= total:
            return
    if yielded < total:
        print(f"warning: only got {yielded} paragraphs (asked for {total})", flush=True)


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-n", type=int, default=45000)
    parser.add_argument("--heldout-n", type=int, default=5000)
    parser.add_argument("--train-out", type=Path, default=ROOT / "data" / "general_train.jsonl")
    parser.add_argument("--heldout-out", type=Path, default=ROOT / "data" / "general_heldout.jsonl")
    args = parser.parse_args()

    total = args.train_n + args.heldout_n
    print(f"downloading {total} paragraphs from wikitext-103", flush=True)

    try:
        records = list(stream_wikitext(total))
    except Exception as exc:  # noqa: BLE001
        print(f"offline or error: {exc}\nSkipping general-corpus download.", flush=True)
        return

    train_recs = records[: args.train_n]
    heldout_recs = records[args.train_n :]
    write_jsonl(args.train_out, train_recs)
    write_jsonl(args.heldout_out, heldout_recs)
    print(f"train: {len(train_recs)} → {args.train_out}")
    print(f"heldout: {len(heldout_recs)} → {args.heldout_out}")
    print("Train general BPE with: uv run python scripts/train_medical_tokenizer.py "
          f"--corpus {args.train_out} --out artifacts/general-bpe/tokenizer.json")


if __name__ == "__main__":
    main()
