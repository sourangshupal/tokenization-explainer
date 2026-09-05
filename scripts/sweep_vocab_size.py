#!/usr/bin/env python3
"""Vocab-size sweep on one medical corpus.

Train (or load) byte-level BPE at 16k / 32k / 50k / 64k / 100k. Measure average
tokens per held-out document. Pick the smallest vocab near the minimum.

Instructor:

  uv run python scripts/sweep_vocab_size.py \\
      --corpus data/pubmed_train.jsonl \\
      --heldout data/pubmed_heldout.jsonl \\
      --no-qwen
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_tokenizers import (  # noqa: E402
    DOMAIN_TERMS,
    EncodeFn,
    fertility,
    iter_jsonl_texts,
    load_custom_encode,
    load_tiktoken_encode,
    read_lines,
    single_token_rate,
    tokens_per_100_chars,
    try_load_qwen,
)
from train_medical_tokenizer import train_byte_level_bpe  # noqa: E402

console = Console()

DEFAULT_SIZES: tuple[int, ...] = (16_000, 32_000, 50_000, 64_000, 100_000)
D_MODEL = 1024
KNEE_THRESHOLD = 0.02

# Lesson table (large-corpus example). Used when sweep artifacts are missing.
WORKED_EXAMPLE: tuple[tuple[int, float], ...] = (
    (16_000, 1250.0),
    (32_000, 1050.0),
    (50_000, 980.0),
    (64_000, 960.0),
    (100_000, 950.0),
)

BAND_SMALL = "16k-32k"
BAND_MEDIUM = "32k-64k"
BAND_LARGE = "50k-100k"


@dataclass
class SweepRow:
    """One vocab size evaluated on a held-out set."""

    requested: int
    actual: int
    mean_tokens_per_doc: float
    fertility: float = 0.0
    tok_per_100ch: float = 0.0
    single_tok_rate: float = 0.0
    vs_previous_pct: float | None = None
    vs_base_pct: float | None = None
    name: str = ""
    path: Path | None = None
    saturated: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            self.name = size_label(self.requested)
        self.saturated = self.actual < self.requested


# ── Pure helpers (notebook + tests import these) ─────────────────────────────


def size_label(vocab_size: int) -> str:
    """16000 → '16k'."""
    if vocab_size % 1000 == 0:
        return f"{vocab_size // 1000}k"
    return str(vocab_size)


def delta_pct(current: float, previous: float) -> float | None:
    """Percent change from previous to current. Negative = fewer tokens."""
    if previous == 0:
        return None
    return 100.0 * (current - previous) / previous


def recommend_vocab(
    rows: Sequence[tuple[int, float]],
    threshold: float = KNEE_THRESHOLD,
) -> int:
    """Smallest vocab whose avg tokens/doc is within threshold of the best (lowest)."""
    if not rows:
        raise ValueError("rows must be non-empty")
    best = min(avg for _, avg in rows)
    near = [(size, avg) for size, avg in rows if avg <= best * (1.0 + threshold)]
    return min(size for size, _ in near)


def corpus_band(n_tokens: int) -> str:
    """Recommended vocab band from estimated train-set token count."""
    if n_tokens < 1_000_000_000:
        return BAND_SMALL
    if n_tokens < 50_000_000_000:
        return BAND_MEDIUM
    return BAND_LARGE


def embedding_params(vocab_size: int, d_model: int = D_MODEL) -> int:
    """Input-embedding (or output softmax) parameter count: vocab × d_model."""
    return vocab_size * d_model


def sweep_json_path(root: Path, vocab_size: int) -> Path:
    """artifacts/medical-bpe-pubmed-{16k,...}/tokenizer.json"""
    return root / "artifacts" / f"medical-bpe-pubmed-{size_label(vocab_size)}" / "tokenizer.json"


def alias_16k_path(root: Path) -> Path:
    """Existing lab artifact used as the 16k sweep alias."""
    return root / "artifacts" / "medical-bpe-pubmed" / "tokenizer.json"


def resolve_existing_json(root: Path, vocab_size: int) -> Path | None:
    """Prefer labeled sweep dir; 16k may fall back to the lab alias."""
    labeled = sweep_json_path(root, vocab_size)
    if labeled.is_file():
        return labeled
    if vocab_size == 16_000:
        alias = alias_16k_path(root)
        if alias.is_file():
            return alias
    return None


def train_or_load(
    corpus: Path,
    output: Path,
    vocab_size: int,
    skip_existing: bool = True,
    min_frequency: int = 2,
) -> Path:
    """Train BPE unless skip_existing and tokenizer.json already exists."""
    output = Path(output)
    if skip_existing and output.is_file():
        return output
    train_byte_level_bpe(
        corpus, output, vocab_size=vocab_size, min_frequency=min_frequency
    )
    return output


def evaluate_tokenizer(
    tokenizer_json: Path,
    texts: Sequence[str],
    requested: int,
    terms: Sequence[str] = DOMAIN_TERMS,
) -> SweepRow:
    """Mean tokens/doc, fertility, tok/100ch, single-token rate, actual vocab."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    encode = load_custom_encode(tokenizer_json)
    return evaluate_encode(
        name=size_label(requested),
        encode=encode,
        texts=texts,
        requested=requested,
        actual=tokenizer.get_vocab_size(),
        terms=terms,
        path=tokenizer_json,
    )


def evaluate_encode(
    name: str,
    encode: EncodeFn,
    texts: Sequence[str],
    requested: int,
    actual: int,
    terms: Sequence[str] = DOMAIN_TERMS,
    path: Path | None = None,
) -> SweepRow:
    """Evaluate any encode fn (custom BPE or production baseline)."""
    pieces = [encode(t) for t in texts]
    n_docs = max(1, len(pieces))
    mean_n = sum(len(p) for p in pieces) / n_docs
    return SweepRow(
        requested=requested,
        actual=actual,
        mean_tokens_per_doc=mean_n,
        fertility=fertility(pieces, texts) if texts else 0.0,
        tok_per_100ch=tokens_per_100_chars(pieces, texts) if texts else 0.0,
        single_tok_rate=single_token_rate(encode, terms),
        name=name,
        path=path,
    )


def attach_deltas(rows: Sequence[SweepRow]) -> list[SweepRow]:
    """Fill vs-previous and vs-first-row percent columns."""
    out = list(rows)
    if not out:
        return out
    base = out[0].mean_tokens_per_doc
    prev: float | None = None
    for i, row in enumerate(out):
        row.vs_previous_pct = None if prev is None else delta_pct(row.mean_tokens_per_doc, prev)
        row.vs_base_pct = None if i == 0 else delta_pct(row.mean_tokens_per_doc, base)
        prev = row.mean_tokens_per_doc
    return out


def estimate_corpus_tokens(path: Path, max_docs: int = 50_000) -> int:
    """Whitespace-word proxy for train-set size (not model tokens)."""
    if not path.is_file():
        return 0
    if path.suffix == ".jsonl":
        texts = iter_jsonl_texts(path, max_docs=max_docs)
    else:
        texts = read_lines(path)
    return sum(max(1, len(t.split())) for t in texts)


def load_heldout_texts(path: Path, max_docs: int) -> list[str]:
    """JSONL abstracts or plain-text lines."""
    if path.suffix == ".jsonl":
        return iter_jsonl_texts(path, max_docs=max_docs)
    return read_lines(path)


def worked_example_rows() -> list[SweepRow]:
    """Lesson numbers so class can discuss flattening without trained artifacts."""
    rows = [
        SweepRow(
            requested=size,
            actual=size,
            mean_tokens_per_doc=avg,
            name=size_label(size),
        )
        for size, avg in WORKED_EXAMPLE
    ]
    return attach_deltas(rows)


def load_available_sweep_rows(
    root: Path,
    texts: Sequence[str],
    sizes: Sequence[int] = DEFAULT_SIZES,
) -> tuple[list[SweepRow], bool]:
    """Load trained artifacts. If none exist, return the worked example.

    Returns (rows, used_example).
    """
    if not texts:
        return worked_example_rows(), True
    rows: list[SweepRow] = []
    for size in sizes:
        json_path = resolve_existing_json(root, size)
        if json_path is None:
            continue
        rows.append(evaluate_tokenizer(json_path, texts, requested=size))
    if not rows:
        return worked_example_rows(), True
    return attach_deltas(rows), False


# ── Display ──────────────────────────────────────────────────────────────────


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}%"


def print_sweep_table(rows: Sequence[SweepRow], title: str) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAD)
    table.add_column("vocab")
    table.add_column("requested", justify="right")
    table.add_column("actual", justify="right")
    table.add_column("avg tokens/doc", justify="right")
    table.add_column("vs previous", justify="right")
    table.add_column("vs 16k", justify="right")
    table.add_column("fertility", justify="right")
    table.add_column("tok/100ch", justify="right")
    table.add_column("single-tok", justify="right")
    for row in rows:
        actual = str(row.actual)
        if row.saturated:
            actual = f"{row.actual}*"
        table.add_row(
            row.name,
            f"{row.requested:,}",
            actual,
            f"{row.mean_tokens_per_doc:,.1f}",
            _fmt_pct(row.vs_previous_pct),
            _fmt_pct(row.vs_base_pct),
            f"{row.fertility:.3f}",
            f"{row.tok_per_100ch:.2f}",
            f"{row.single_tok_rate:.0%}",
        )
    console.print(table)


def print_embedding_table(sizes: Sequence[int], d_model: int) -> None:
    base = embedding_params(sizes[0], d_model) if sizes else 1
    table = Table(
        title=f"Embedding cost  (params ≈ vocab × d_model={d_model})",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("vocab")
    table.add_column("embedding params", justify="right")
    table.add_column("vs smallest", justify="right")
    for size in sizes:
        n = embedding_params(size, d_model)
        table.add_row(size_label(size), f"{n:,}", f"{n / base:.2f}×")
    console.print(table)


def print_footer(
    rows: Sequence[SweepRow],
    n_train_tokens: int,
    d_model: int,
) -> None:
    pairs = [(r.requested, r.mean_tokens_per_doc) for r in rows]
    knee = recommend_vocab(pairs)
    band = corpus_band(n_train_tokens) if n_train_tokens else BAND_SMALL
    console.print(
        f"\n[bold]Knee / rule of thumb:[/] smallest vocab within "
        f"{KNEE_THRESHOLD:.0%} of best avg tokens/doc → [green]{size_label(knee)}[/]"
    )
    if n_train_tokens:
        console.print(
            f"[bold]Corpus-size rule:[/] ~{n_train_tokens:,} whitespace-word tokens "
            f"→ band [cyan]{band}[/]"
        )
        if band == BAND_SMALL and knee > 32_000:
            console.print(
                "[yellow]Mismatch:[/] small corpus (<1B) usually wants 16k–32k. "
                "A bigger knee is extra embedding rows for little compression."
            )
    saturated = [r for r in rows if r.saturated]
    if saturated:
        labels = ", ".join(r.name for r in saturated)
        console.print(
            f"[yellow]Saturation:[/] actual vocab < requested at {labels}. "
            "BPE vocab_size is a max; min_frequency=2 starved the rest of the budget."
        )
    print_embedding_table([r.requested for r in rows], d_model)
    console.print(
        "[dim]Rule of thumb: choose the smallest vocabulary that achieves "
        "near-minimum tokenization length. Extra vocab = extra parameters "
        "in both the input embedding and the output softmax.[/]\n"
    )


def print_baseline_table(rows: Sequence[SweepRow]) -> None:
    if not rows:
        return
    table = Table(
        title="Production baselines (not trained on this corpus)",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("tokenizer")
    table.add_column("vocab", justify="right")
    table.add_column("avg tokens/doc", justify="right")
    table.add_column("fertility", justify="right")
    for row in rows:
        table.add_row(
            row.name,
            f"{row.actual:,}",
            f"{row.mean_tokens_per_doc:,.1f}",
            f"{row.fertility:.3f}",
        )
    console.print(table)


def tiktoken_vocab_size(encoding: str) -> int:
    import tiktoken

    return tiktoken.get_encoding(encoding).n_vocab


def collect_baselines(
    texts: Sequence[str],
    include_qwen: bool,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for name, enc_name, approx in (
        ("cl100k", "cl100k_base", 100_256),
        ("o200k", "o200k_base", 200_019),
    ):
        encode = load_tiktoken_encode(enc_name)
        actual = tiktoken_vocab_size(enc_name)
        rows.append(
            evaluate_encode(name, encode, texts, requested=approx, actual=actual)
        )
    if include_qwen:
        qwen = try_load_qwen()
        if qwen is not None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
            rows.append(
                evaluate_encode(
                    "qwen", qwen, texts, requested=len(tok), actual=len(tok)
                )
            )
    return rows


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_sizes(raw: str) -> list[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("need at least one vocab size")
    return sizes


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "pubmed_train.jsonl",
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=ROOT / "data" / "pubmed_heldout.jsonl",
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=list(DEFAULT_SIZES),
        help="comma-separated vocab sizes (default: 16000,32000,50000,64000,100000)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse tokenizer.json if present (default: true)",
    )
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    corpus = args.corpus.expanduser().resolve()
    heldout = args.heldout.expanduser().resolve()
    sizes: list[int] = list(args.sizes)

    json_paths: dict[int, Path] = {}
    for size in sizes:
        labeled = sweep_json_path(root, size)
        existing = resolve_existing_json(root, size) if args.skip_existing else None
        if existing is not None:
            console.print(f"[dim]reuse {size_label(size)}: {existing}[/]")
            json_paths[size] = existing
            continue
        if not corpus.is_file():
            raise FileNotFoundError(
                f"corpus not found: {corpus}. Train with "
                "scripts/train_medical_tokenizer.py or pass --corpus."
            )
        console.print(f"[cyan]train {size_label(size)} → {labeled}[/]")
        json_paths[size] = train_or_load(
            corpus=corpus,
            output=labeled,
            vocab_size=size,
            skip_existing=False,
            min_frequency=args.min_frequency,
        )

    if not heldout.is_file():
        console.print(f"[yellow]held-out not found: {heldout} — using worked example[/]")
        rows = worked_example_rows()
        print_sweep_table(
            rows,
            "Worked example (large-corpus lesson numbers — not this PubMed run)",
        )
        print_footer(rows, n_train_tokens=0, d_model=args.d_model)
        return

    texts = load_heldout_texts(heldout, max_docs=args.max_docs)
    console.print(f"[dim]held-out: {len(texts)} docs from {heldout}[/]")

    rows = attach_deltas(
        [
            evaluate_tokenizer(json_paths[size], texts, requested=size)
            for size in sizes
        ]
    )
    print_sweep_table(rows, "Vocab-size sweep — avg tokens / held-out document")
    print_baseline_table(collect_baselines(texts, include_qwen=not args.no_qwen))

    n_train = estimate_corpus_tokens(corpus) if corpus.is_file() else 0
    print_footer(rows, n_train_tokens=n_train, d_model=args.d_model)


if __name__ == "__main__":
    main()
