"""Vocab-size sweep helpers and tiny-corpus train/eval.

CI never trains 50k–100k. Tiny authored corpus + small vocabs only.
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
    fertility,
    load_custom_encode,
    mean_piece_count,
    read_lines,
)
from sweep_vocab_size import (  # noqa: E402
    WORKED_EXAMPLE,
    corpus_band,
    delta_pct,
    embedding_params,
    evaluate_tokenizer,
    load_available_sweep_rows,
    recommend_vocab,
    size_label,
    sweep_json_path,
    train_or_load,
)
from train_medical_tokenizer import train_byte_level_bpe  # noqa: E402

TINY_CORPUS = ROOT / "data" / "medical_corpus.txt"


def test_delta_pct_worked_example_64k_to_100k_is_about_one_percent() -> None:
    """960 → 950 is ~1% fewer tokens — the lesson's flattening jump."""
    pct = delta_pct(950.0, 960.0)
    assert pct is not None
    assert pct == pytest.approx(-1.04166, abs=0.01)


def test_delta_pct_none_when_previous_is_zero() -> None:
    assert delta_pct(10.0, 0.0) is None


def test_recommend_vocab_picks_smallest_within_two_percent() -> None:
    """Worked example: 100k is best (950) but 64k (960) is within 2% → pick 64k."""
    rows = [(size, avg) for size, avg in WORKED_EXAMPLE]
    assert recommend_vocab(rows, threshold=0.02) == 64000


def test_recommend_vocab_picks_best_when_gap_exceeds_threshold() -> None:
    rows = [(16_000, 1250.0), (32_000, 1050.0)]
    assert recommend_vocab(rows, threshold=0.02) == 32_000


def test_corpus_band_small_corpus_is_16k_32k() -> None:
    assert corpus_band(10_000_000) == "16k-32k"


def test_corpus_band_medium_and_large() -> None:
    assert corpus_band(1_000_000_000) == "32k-64k"
    assert corpus_band(5_000_000_000) == "32k-64k"
    assert corpus_band(100_000_000_000) == "50k-100k"


def test_embedding_params_100k_at_d_model_1024() -> None:
    assert embedding_params(100_000, 1024) == 102_400_000
    assert embedding_params(16_000, 1024) * 6.25 == embedding_params(100_000, 1024)


def test_size_label() -> None:
    assert size_label(16_000) == "16k"
    assert size_label(32_000) == "32k"
    assert size_label(50_000) == "50k"
    assert size_label(100_000) == "100k"


def test_sweep_json_path_uses_labeled_dir() -> None:
    path = sweep_json_path(ROOT, 32_000)
    assert path == ROOT / "artifacts" / "medical-bpe-pubmed-32k" / "tokenizer.json"


def test_reports_requested_and_actual_vocab(tmp_path: Path) -> None:
    out = tmp_path / "tok.json"
    tok = train_byte_level_bpe(TINY_CORPUS, out, vocab_size=512, min_frequency=2)
    texts = read_lines(TINY_CORPUS)[:40]
    row = evaluate_tokenizer(out, texts, requested=512)
    assert row.requested == 512
    assert row.actual == tok.get_vocab_size()
    assert row.actual <= 512
    assert row.mean_tokens_per_doc > 0


def test_larger_vocab_mean_pieces_does_not_increase(tmp_path: Path) -> None:
    """On the same tiny corpus, bigger requested vocab → fewer or equal mean pieces."""
    texts = read_lines(TINY_CORPUS)[:80]
    means: list[float] = []
    for size in (512, 1024, 2048):
        out = tmp_path / f"v{size}" / "tokenizer.json"
        train_byte_level_bpe(TINY_CORPUS, out, vocab_size=size, min_frequency=2)
        encode = load_custom_encode(out)
        means.append(mean_piece_count(encode, texts))
        pieces = [encode(t) for t in texts]
        assert fertility(pieces, texts) > 0
    assert means[1] <= means[0] + 1e-9
    assert means[2] <= means[1] + 1e-9


def test_train_or_load_skips_existing(tmp_path: Path) -> None:
    corpus = TINY_CORPUS
    first = train_or_load(
        corpus=corpus,
        output=tmp_path / "a" / "tokenizer.json",
        vocab_size=512,
        skip_existing=False,
    )
    second = train_or_load(
        corpus=corpus,
        output=tmp_path / "a" / "tokenizer.json",
        vocab_size=512,
        skip_existing=True,
    )
    assert first == second
    assert (tmp_path / "a" / "tokenizer.json").is_file()


def test_empty_texts_uses_worked_example() -> None:
    rows, used_example = load_available_sweep_rows(ROOT, texts=[], sizes=(16_000, 32_000))
    assert used_example is True
    assert [r.requested for r in rows] == [16_000, 32_000, 50_000, 64_000, 100_000]
    assert recommend_vocab([(r.requested, r.mean_tokens_per_doc) for r in rows]) == 64_000
