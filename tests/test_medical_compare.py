"""Medical custom-vs-general tokenizer lab tests. Network: tiktoken encodings only.

Fair eval design:
- Golden assert uses held-out PubMed abstracts (pubmed_heldout.jsonl) — skipped if absent.
- Curated probes test is marked as illustration only (can fail on fallback tokenizer).
- General-BPE cross-domain sanity test: general-bpe < custom-med on general held-out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_tokenizers import (  # noqa: E402
    DOMAIN_TERMS,
    fertility,
    iter_jsonl_texts,
    load_custom_encode,
    load_tiktoken_encode,
    mean_piece_count,
    read_lines,
    single_token_rate,
)
from train_medical_tokenizer import train_byte_level_bpe  # noqa: E402
from wrap_medical_tokenizer import wrap_tokenizer  # noqa: E402

THEORY_FILES = [
    "00-overview.md",
    "01-why-tokenization-matters.md",
    "02-general-purpose-tokenizers.md",
    "03-custom-domain-tokenizers.md",
    "04-metrics.md",
    "05-why-custom-wins-in-healthcare.md",
    "06-pretrained-model-trap.md",
    "07-optional-lora-sft.md",
    "08-vocab-size-tradeoff.md",
]

ROUND_TRIP_STRINGS = [
    "empagliflozin 10 mg daily",
    "serum creatinine 1.4 mg/dL",
    "levothyroxine 75 μg daily",
    "β-lactam allergy",
    "Naïve CD4+ T cells",
    "ICD-10-CM E11.65",
]

# Held-out files created by scripts/split_corpus.py and scripts/download_general_sample.py
PUBMED_HELDOUT = ROOT / "data" / "pubmed_heldout.jsonl"
GENERAL_HELDOUT = ROOT / "data" / "general_heldout.jsonl"
MED_BPE_JSON = ROOT / "artifacts" / "medical-bpe-pubmed" / "tokenizer.json"
GENERAL_BPE_JSON = ROOT / "artifacts" / "general-bpe" / "tokenizer.json"


@pytest.fixture(scope="session")
def trained_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a session-scoped BPE on the authored medical corpus (fast, no network)."""
    out = tmp_path_factory.mktemp("bpe") / "tokenizer.json"
    train_byte_level_bpe(
        ROOT / "data" / "medical_corpus.txt",
        out,
        vocab_size=8000,
        min_frequency=2,
    )
    return out


# ── Structural tests (always run) ───────────────────────────────────────────


def test_pretrained_fallback_exists() -> None:
    path = ROOT / "models" / "pretrained" / "medical_bpe_tiny" / "tokenizer.json"
    assert path.is_file()
    assert path.stat().st_size > 1000
    lines = read_lines(ROOT / "data" / "medical_corpus.txt")
    assert len(lines) >= 300


def test_probes_include_required_examples() -> None:
    text = (ROOT / "data" / "medical_probes.txt").read_text(encoding="utf-8")
    required = [
        "empagliflozin 10 mg daily",
        "acetylcholinesterase inhibitor",
        "BRCA1 pathogenic variant",
        "ICD-10-CM E11.65",
        "serum creatinine 1.4 mg/dL",
        "β-lactam allergy",
    ]
    for probe in required:
        assert probe in text, probe


def test_theory_files_exist() -> None:
    theory = ROOT / "docs" / "theory"
    for name in THEORY_FILES:
        path = theory / name
        assert path.is_file(), path
        assert path.stat().st_size > 80, name


def test_round_trip(trained_json: Path) -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(trained_json))
    for text in ROUND_TRIP_STRINGS:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        assert decoded == text, (text, decoded)


# ── Golden assert: held-out fertility (fair eval) ────────────────────────────


@pytest.mark.skipif(
    not PUBMED_HELDOUT.is_file() or not MED_BPE_JSON.is_file(),
    reason="pubmed_heldout.jsonl or medical-bpe-pubmed/tokenizer.json not found; "
           "run scripts/split_corpus.py and scripts/train_medical_tokenizer.py first",
)
def test_custom_med_beats_cl100k_on_heldout_abstracts() -> None:
    """Fair eval: custom-med fertility < cl100k on 1000 held-out PubMed abstracts."""
    texts = iter_jsonl_texts(PUBMED_HELDOUT, max_docs=1000)
    assert len(texts) >= 100, f"only {len(texts)} heldout texts — split may have failed"

    custom = load_custom_encode(MED_BPE_JSON)
    cl100k = load_tiktoken_encode("cl100k_base")

    custom_pieces = [custom(t) for t in texts]
    cl100k_pieces = [cl100k(t) for t in texts]

    custom_fert = fertility(custom_pieces, texts)
    cl100k_fert = fertility(cl100k_pieces, texts)

    assert custom_fert < cl100k_fert, (
        f"custom-med fertility {custom_fert:.4f} >= cl100k fertility {cl100k_fert:.4f} "
        f"on {len(texts)} held-out abstracts — domain BPE should win on same-domain text"
    )


@pytest.mark.skipif(
    not PUBMED_HELDOUT.is_file() or not MED_BPE_JSON.is_file(),
    reason="held-out files not found",
)
def test_custom_med_beats_o200k_on_heldout_abstracts() -> None:
    """Fair eval: custom-med fertility < o200k on held-out PubMed abstracts."""
    texts = iter_jsonl_texts(PUBMED_HELDOUT, max_docs=1000)
    custom = load_custom_encode(MED_BPE_JSON)
    o200k = load_tiktoken_encode("o200k_base")
    custom_pieces = [custom(t) for t in texts]
    o200k_pieces = [o200k(t) for t in texts]
    custom_fert = fertility(custom_pieces, texts)
    o200k_fert = fertility(o200k_pieces, texts)
    assert custom_fert < o200k_fert, (
        f"custom-med {custom_fert:.4f} >= o200k {o200k_fert:.4f}"
    )


@pytest.mark.skipif(
    not PUBMED_HELDOUT.is_file() or not MED_BPE_JSON.is_file() or not GENERAL_BPE_JSON.is_file(),
    reason="held-out files or general-bpe not found",
)
def test_custom_med_beats_general_bpe_on_heldout_abstracts() -> None:
    """Fairness control: custom-med beats same-size general BPE on medical text.
    Proves domain, not vocab size, drives the improvement."""
    texts = iter_jsonl_texts(PUBMED_HELDOUT, max_docs=1000)
    custom = load_custom_encode(MED_BPE_JSON)
    general = load_custom_encode(GENERAL_BPE_JSON)
    custom_pieces = [custom(t) for t in texts]
    general_pieces = [general(t) for t in texts]
    custom_fert = fertility(custom_pieces, texts)
    general_fert = fertility(general_pieces, texts)
    assert custom_fert < general_fert, (
        f"custom-med {custom_fert:.4f} >= general-bpe {general_fert:.4f} — "
        f"same vocab size, domain should be the differentiator"
    )


@pytest.mark.skipif(
    not GENERAL_HELDOUT.is_file() or not MED_BPE_JSON.is_file() or not GENERAL_BPE_JSON.is_file(),
    reason="general held-out or tokenizer files not found",
)
def test_general_bpe_beats_custom_med_on_general_text() -> None:
    """Cross-domain sanity: general-bpe fertility < custom-med on general English held-out.
    Custom-med MUST lose out-of-domain — that is the expected, honest result."""
    texts = iter_jsonl_texts(GENERAL_HELDOUT, max_docs=1000)
    custom = load_custom_encode(MED_BPE_JSON)
    general = load_custom_encode(GENERAL_BPE_JSON)
    custom_pieces = [custom(t) for t in texts]
    general_pieces = [general(t) for t in texts]
    custom_fert = fertility(custom_pieces, texts)
    general_fert = fertility(general_pieces, texts)
    assert general_fert < custom_fert, (
        f"general-bpe {general_fert:.4f} >= custom-med {custom_fert:.4f} on general text — "
        f"specialization should cost out-of-domain performance"
    )


# ── Single-token rate ─────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not MED_BPE_JSON.is_file(),
    reason="medical-bpe-pubmed not found",
)
def test_custom_med_single_token_rate_beats_cl100k() -> None:
    """custom-med should encode more medical terms as single tokens than cl100k."""
    custom = load_custom_encode(MED_BPE_JSON)
    cl100k = load_tiktoken_encode("cl100k_base")
    custom_rate = single_token_rate(custom, DOMAIN_TERMS)
    cl100k_rate = single_token_rate(cl100k, DOMAIN_TERMS)
    assert custom_rate >= cl100k_rate, (
        f"custom-med single-token rate {custom_rate:.2%} < cl100k {cl100k_rate:.2%}"
    )


# ── Illustration test (leakage note kept for teaching) ───────────────────────


def test_custom_mean_pieces_illustration_probes(trained_json: Path) -> None:
    """Illustration: custom trained on medical_corpus beats cl100k on same-domain probes.
    NOTE: this is an illustration of in-domain fit, not a fair eval — the probes overlap
    with phrases in medical_corpus.txt. The fair eval is test_custom_med_beats_cl100k_on_heldout_abstracts.
    """
    probes = read_lines(ROOT / "data" / "medical_probes.txt")
    custom = load_custom_encode(trained_json)
    cl100k = load_tiktoken_encode()
    custom_mean = mean_piece_count(custom, probes)
    general_mean = mean_piece_count(cl100k, probes)
    assert custom_mean < general_mean, (custom_mean, general_mean)


def test_control_has_no_required_win(trained_json: Path) -> None:
    """Control file exists; custom is allowed to lose. No assert on winner."""
    control = read_lines(ROOT / "data" / "medical_control.txt")
    assert len(control) >= 20
    custom = load_custom_encode(trained_json)
    cl100k = load_tiktoken_encode()
    _ = mean_piece_count(custom, control)
    _ = mean_piece_count(cl100k, control)


def test_wrap_tokenizer(trained_json: Path, tmp_path: Path) -> None:
    out = tmp_path / "hf"
    hf_tok = wrap_tokenizer(trained_json, out)
    assert (out / "tokenizer.json").is_file()
    pieces = hf_tok.tokenize("empagliflozin 10 mg daily")
    assert pieces
    assert len(hf_tok) >= 256
