#!/usr/bin/env python3
"""Download PubMed title+abstract JSONL (gitignored).

HF `ncbi/pubmed` uses a dataset script (blocked on current `datasets`).
This loader streams parquet JSONL from Hugging Face, then falls back to NCBI FTP XML.

Run: uv run python scripts/download_pubmed_sample.py --max-docs 50000
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterator
from xml.etree.ElementTree import iterparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "pubmed_sample.jsonl"
HF_PARQUET_ID = "slinusc/PubMedAbstractsSubset"
NCBI_XML = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n{n:04d}.xml.gz"


def _as_text(value: Any) -> str:
    """Flatten nested title/abstract fields to a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(part for item in value if (part := _as_text(item)))
    if isinstance(value, dict):
        if "AbstractText" in value:
            return _as_text(value["AbstractText"])
        if "#text" in value:
            return _as_text(value["#text"])
        return " ".join(part for v in value.values() if (part := _as_text(v)))
    return str(value).strip()


def join_title_abstract(title: str, abstract: str) -> str | None:
    """Return title+abstract if either is non-empty."""
    text = f"{title}\n{abstract}".strip()
    return text if text else None


def stream_hf_parquet(max_docs: int) -> Iterator[dict[str, str]]:
    """Download title/abstract rows from a parquet Hub dataset.

    Uses streaming=False (full Parquet download) to avoid [Errno 9] Bad file
    descriptor errors caused by fsspec seek() on non-seekable network streams.
    Dataset is cached locally after the first run.
    """
    from datasets import load_dataset

    print(f"downloading {HF_PARQUET_ID} (cached after first run)...", flush=True)
    dataset = load_dataset(HF_PARQUET_ID, split="train", streaming=False)
    yielded = 0
    for row in dataset:
        rec = dict(row)
        text = join_title_abstract(_as_text(rec.get("title")), _as_text(rec.get("abstract")))
        if not text:
            continue
        yielded += 1
        if yielded % 1000 == 0:
            print(f"kept {yielded}/{max_docs}", flush=True)
        yield {"text": text}
        if yielded >= max_docs:
            return


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def texts_from_pubmed_xml_gz(path: Path) -> Iterator[str]:
    """Yield title+abstract strings from one NCBI baseline XML.gz."""
    with gzip.open(path, "rb") as handle:
        for _event, elem in iterparse(handle, events=("end",)):
            if _local_tag(elem.tag) != "PubmedArticle":
                continue
            title = ""
            abstract_parts: list[str] = []
            for child in elem.iter():
                name = _local_tag(child.tag)
                if name == "ArticleTitle" and child.text:
                    title = (child.text or "").strip()
                elif name == "AbstractText":
                    chunk = "".join(child.itertext()).strip()
                    if chunk:
                        abstract_parts.append(chunk)
            text = join_title_abstract(title, " ".join(abstract_parts))
            elem.clear()
            if text:
                yield text


def stream_ncbi_ftp(max_docs: int, cache_dir: Path) -> Iterator[dict[str, str]]:
    """Download NCBI baseline XML.gz files until max_docs abstracts are kept."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    yielded = 0
    for file_i in range(1, 80):
        url = NCBI_XML.format(n=file_i)
        dest = cache_dir / f"pubmed26n{file_i:04d}.xml.gz"
        if not dest.is_file():
            print(f"download {url}", flush=True)
            urllib.request.urlretrieve(url, dest)
        print(f"parse {dest.name}", flush=True)
        for text in texts_from_pubmed_xml_gz(dest):
            yielded += 1
            if yielded % 1000 == 0:
                print(f"kept {yielded}/{max_docs}", flush=True)
            yield {"text": text}
            if yielded >= max_docs:
                return
    raise RuntimeError(f"only got {yielded} abstracts from NCBI files 1–79")


def stream_pubmed(max_docs: int) -> Iterator[dict[str, str]]:
    """Prefer Hub parquet; fall back to NCBI FTP XML."""
    try:
        yield from stream_hf_parquet(max_docs)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"Hub parquet failed ({exc!r}); falling back to NCBI FTP", flush=True)
    cache = ROOT / "data" / ".pubmed_xml_cache"
    yield from stream_ncbi_ftp(max_docs, cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    print(f"streaming up to {args.max_docs} abstracts → {args.out}", flush=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for rec in stream_pubmed(args.max_docs):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n >= args.max_docs:
                break
    print(f"wrote {n} docs → {args.out}", flush=True)
    print("Train with: uv run python scripts/train_medical_tokenizer.py --corpus", args.out)


if __name__ == "__main__":
    main()
