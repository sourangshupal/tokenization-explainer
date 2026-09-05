#!/usr/bin/env python3
"""Compare custom medical BPE vs tiktoken cl100k_base, o200k_base, and an optional
same-size general-domain BPE control.

Quick run (probes only):
  uv run python scripts/compare_tokenizers.py --no-qwen

With held-out eval (run after split + train):
  uv run python scripts/compare_tokenizers.py --no-qwen \\
      --tokenizer-json artifacts/medical-bpe-pubmed/tokenizer.json \\
      --general-bpe    artifacts/general-bpe/tokenizer.json \\
      --heldout-medical data/pubmed_heldout.jsonl \\
      --heldout-general data/general_heldout.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DEFAULT_PROBES = ROOT / "data" / "medical_probes.txt"
DEFAULT_CONTROL = ROOT / "data" / "medical_control.txt"
QWEN_ID = "Qwen/Qwen3-0.6B"
FALLBACK_JSON = ROOT / "models" / "pretrained" / "medical_bpe_tiny" / "tokenizer.json"
TRAINED_JSON = ROOT / "artifacts" / "medical-bpe-pubmed" / "tokenizer.json"

# Medical domain terms used for single-token-rate metric
DOMAIN_TERMS = [
    "empagliflozin", "acetylcholinesterase", "BRCA1", "metformin", "levothyroxine",
    "oseltamivir", "vancomycin", "troponin", "hemoglobin", "creatinine",
    "myocardial", "pneumonia", "lymphocyte", "epinephrine", "corticosteroid",
    "hypertension", "cardiomyopathy", "anticoagulation", "thrombocytopenia", "septicemia",
]

console = Console()

EncodeFn = Callable[[str], list[str]]


# ── I/O helpers ─────────────────────────────────────────────────────────────


def read_lines(path: Path) -> list[str]:
    """Non-empty stripped lines."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_jsonl_texts(path: Path, max_docs: int = 10_000) -> list[str]:
    """Read JSONL file, return non-empty text strings up to max_docs."""
    texts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
            except json.JSONDecodeError:
                text = line
            if text.strip():
                texts.append(text.strip())
            if len(texts) >= max_docs:
                break
    return texts


# ── Metric helpers ───────────────────────────────────────────────────────────


def n_words(text: str) -> int:
    """Whitespace word count, at least 1."""
    return max(1, len(text.split()))


def fertility(pieces_per_text: Sequence[list[str]], texts: Sequence[str]) -> float:
    """Mean tokens / whitespace-words."""
    token_n = sum(len(p) for p in pieces_per_text)
    word_n = sum(n_words(t) for t in texts)
    return token_n / word_n


def tokens_per_100_chars(pieces_per_text: Sequence[list[str]], texts: Sequence[str]) -> float:
    """Mean tokens per 100 characters."""
    token_n = sum(len(p) for p in pieces_per_text)
    char_n = sum(len(t) for t in texts)
    if char_n == 0:
        return 0.0
    return 100.0 * token_n / char_n


def mean_piece_count(encode: EncodeFn, texts: Sequence[str]) -> float:
    """Average pieces per string."""
    if not texts:
        return 0.0
    return sum(len(encode(t)) for t in texts) / len(texts)


def single_token_rate(encode: EncodeFn, terms: Sequence[str]) -> float:
    """Fraction of terms encoded as exactly one piece."""
    if not terms:
        return 0.0
    return sum(1 for t in terms if len(encode(t)) == 1) / len(terms)


def winner(counts: dict[str, int]) -> str:
    """Name of tokenizer with fewest pieces; 'tie' if equal first."""
    best = min(counts.values())
    names = [n for n, v in counts.items() if v == best]
    return names[0] if len(names) == 1 else "tie"


# ── Tokenizer loaders ────────────────────────────────────────────────────────


def load_tiktoken_encode(encoding: str = "cl100k_base") -> EncodeFn:
    """Encode fn for tiktoken: cl100k_base or o200k_base."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding)

    def _encode(text: str) -> list[str]:
        ids = enc.encode(text)
        return [enc.decode([i]) for i in ids]

    return _encode


def load_hf_encode(model_id: str) -> EncodeFn:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def _encode(text: str) -> list[str]:
        return list(tok.tokenize(text))

    return _encode


def load_custom_encode(tokenizer_json: Path) -> EncodeFn:
    """Encode fn from a raw tokenizers.Tokenizer tokenizer.json."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def _encode(text: str) -> list[str]:
        return list(tokenizer.encode(text).tokens)

    return _encode


def try_load_qwen(model_id: str = QWEN_ID) -> EncodeFn | None:
    try:
        return load_hf_encode(model_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]skip Qwen ({model_id}): {exc}[/]")
        return None


def resolve_custom_json(explicit: Path | None = None) -> Path:
    """Prefer explicitly given path, then trained artifact, then tiny fallback."""
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"custom tokenizer.json not found: {explicit}")
        return explicit
    if TRAINED_JSON.is_file():
        return TRAINED_JSON
    if FALLBACK_JSON.is_file():
        return FALLBACK_JSON
    raise FileNotFoundError(
        "No custom tokenizer.json. Train with scripts/train_medical_tokenizer.py "
        "or keep models/pretrained/medical_bpe_tiny/tokenizer.json in the tree."
    )


def custom_decode(tokenizer_json: Path) -> Callable[[list[str]], str]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def _decode(pieces: list[str]) -> str:
        ids = [tokenizer.token_to_id(p) for p in pieces]
        if any(i is None for i in ids):
            return ""
        return tokenizer.decode(ids)  # type: ignore[arg-type]

    return _decode


# ── Print helpers ────────────────────────────────────────────────────────────


def summarize(
    name: str, encode: EncodeFn, texts: Sequence[str]
) -> tuple[list[list[str]], float, float]:
    pieces = [encode(t) for t in texts]
    return pieces, fertility(pieces, texts), tokens_per_100_chars(pieces, texts)


def print_probe_table(probes: Sequence[str], columns: dict[str, EncodeFn]) -> None:
    table = Table(title="Probe piece counts", box=box.SIMPLE_HEAD, show_header=True)
    table.add_column("probe", overflow="fold")
    for name in columns:
        table.add_column(name, justify="right")
    table.add_column("winner")
    for probe in probes:
        counts = {name: len(columns[name](probe)) for name in columns}
        table.add_row(probe, *[str(counts[n]) for n in columns], winner(counts))
    console.print(table)


def print_set_metrics(
    title: str, texts: Sequence[str], columns: dict[str, EncodeFn]
) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAD, show_header=True)
    table.add_column("tokenizer")
    table.add_column("fertility", justify="right")
    table.add_column("tok/100ch", justify="right")
    table.add_column("mean pieces", justify="right")
    for name, encode in columns.items():
        pieces, fert, t100 = summarize(name, encode, texts)
        mean_n = sum(len(p) for p in pieces) / max(1, len(pieces))
        table.add_row(name, f"{fert:.3f}", f"{t100:.2f}", f"{mean_n:.2f}")
    console.print(table)


def print_single_token_table(terms: Sequence[str], columns: dict[str, EncodeFn]) -> None:
    """Single-token-rate per tokenizer for a list of domain terms."""
    table = Table(title="Single-token rate on domain terms", box=box.SIMPLE_HEAD)
    table.add_column("tokenizer")
    table.add_column("single-tok rate", justify="right")
    table.add_column("(terms encoded as 1 piece)", justify="right")
    total = len(terms)
    for name, encode in columns.items():
        n_single = sum(1 for t in terms if len(encode(t)) == 1)
        rate = n_single / total if total else 0.0
        table.add_row(name, f"{rate:.1%}", f"{n_single}/{total}")
    console.print(table)


def print_heldout_fertility(
    title: str,
    path: Path,
    columns: dict[str, EncodeFn],
    max_docs: int = 5000,
) -> dict[str, float]:
    """Read a JSONL held-out file, report fertility for each tokenizer. Returns fertility map."""
    if not path.is_file():
        console.print(f"[yellow]skip held-out eval: {path} not found[/]")
        return {}
    texts = iter_jsonl_texts(path, max_docs)
    console.print(f"[dim]held-out: {len(texts)} docs from {path}[/]")
    table = Table(title=title, box=box.SIMPLE_HEAD)
    table.add_column("tokenizer")
    table.add_column("fertility", justify="right")
    table.add_column("tok/100ch", justify="right")
    table.add_column("mean pieces", justify="right")
    results: dict[str, float] = {}
    for name, encode in columns.items():
        pieces, fert, t100 = summarize(name, encode, texts)
        mean_n = sum(len(p) for p in pieces) / max(1, len(pieces))
        table.add_row(name, f"{fert:.3f}", f"{t100:.2f}", f"{mean_n:.2f}")
        results[name] = fert
    console.print(table)
    return results


# ── Column builders ──────────────────────────────────────────────────────────


def build_columns(
    custom_json: Path,
    include_qwen: bool,
    general_bpe_json: Path | None = None,
) -> dict[str, EncodeFn]:
    """Ordered tokenizer encode functions."""
    columns: dict[str, EncodeFn] = {
        "cl100k": load_tiktoken_encode("cl100k_base"),
        "o200k": load_tiktoken_encode("o200k_base"),
        "custom-med": load_custom_encode(custom_json),
    }
    if general_bpe_json is not None and general_bpe_json.is_file():
        columns["general-bpe"] = load_custom_encode(general_bpe_json)
    elif general_bpe_json is not None:
        console.print(f"[yellow]general-bpe not found at {general_bpe_json}, skipping[/]")
    if include_qwen:
        qwen = try_load_qwen()
        if qwen is not None:
            columns["qwen"] = qwen
    return columns


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--tokenizer-json", type=Path, default=None,
                        help="Path to custom-medical tokenizer.json")
    parser.add_argument("--general-bpe", type=Path, default=None,
                        help="Path to same-size general-domain BPE tokenizer.json (fairness control)")
    parser.add_argument("--heldout-medical", type=Path, default=None,
                        help="JSONL held-out medical abstracts for fertility eval")
    parser.add_argument("--heldout-general", type=Path, default=None,
                        help="JSONL held-out general text for cross-domain fertility eval")
    parser.add_argument("--no-qwen", action="store_true")
    args = parser.parse_args()

    custom_json = resolve_custom_json(args.tokenizer_json)
    console.print(f"[dim]custom-med tokenizer: {custom_json}[/]")
    columns = build_columns(
        custom_json, include_qwen=not args.no_qwen, general_bpe_json=args.general_bpe
    )

    probes = read_lines(args.probes)
    control = read_lines(args.control)

    # ── 1. Curated probes (illustration) ─────────────────────────────────────
    print_probe_table(probes, columns)
    print_set_metrics("Medical probes (illustration)", probes, columns)
    print_set_metrics(
        "General-English control (custom-med is not required to win)", control, columns
    )

    # ── 2. Single-token-rate on domain terms ─────────────────────────────────
    print_single_token_table(DOMAIN_TERMS, columns)

    # ── 3. Held-out fertility (the fair eval) ─────────────────────────────────
    if args.heldout_medical:
        med_results = print_heldout_fertility(
            "Held-out PubMed abstracts — FAIR eval (custom-med should win)",
            args.heldout_medical,
            columns,
        )
        if med_results:
            best = min(med_results, key=med_results.get)  # type: ignore[arg-type]
            console.print(
                f"\n[bold green]Medical held-out winner: {best} "
                f"(fertility {med_results[best]:.3f})[/]\n"
            )

    if args.heldout_general:
        gen_results = print_heldout_fertility(
            "Held-out general English — custom-med expected to lose",
            args.heldout_general,
            columns,
        )
        if gen_results:
            best = min(gen_results, key=gen_results.get)  # type: ignore[arg-type]
            console.print(
                f"[bold cyan]General held-out winner: {best} "
                f"(fertility {gen_results[best]:.3f})[/]\n"
            )


if __name__ == "__main__":
    main()
