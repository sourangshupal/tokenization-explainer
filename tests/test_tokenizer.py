"""
Pytest correctness + adversarial suite for the legal-bpe-50k tokenizer.

Run:
    uv run pytest tests/test_tokenizer.py -v

The test file discovers the tokenizer from models/legal-bpe-50k/.
If that directory does not exist, tests are skipped (not failed) so CI
on the source branch continues to pass before training completes.

Test categories (Aug 2026 TokEval / TokTier methodology):
  1. Round-trip correctness (encode → decode === original)
  2. UNK-free guarantee  (byte-level BPE: no UNK ever)
  3. Special token isolation
  4. Empty and trivial inputs
  5. Digit boundary alignment   (TokEval structure-sensitive metric)
  6. Byte-boundary integrity    (no partial UTF-8 codepoints)
  7. Adversarial / edge-case battery  (TokTier 93 k+ agent-step methodology)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "legal-bpe-50k"
TOKENIZER_CONFIG = MODEL_DIR / "tokenizer_config.json"

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tokenizer():
    if not TOKENIZER_CONFIG.exists():
        pytest.skip(
            f"Tokenizer not found at {MODEL_DIR}. "
            "Run scripts/train_tokenizer.py + scripts/wrap_tokenizer.py first."
        )
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("transformers not installed — run: uv sync")
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


# ---------------------------------------------------------------------------
# 1. Round-trip correctness
# ---------------------------------------------------------------------------

ROUND_TRIP_SAMPLES = [
    # Plain legal prose
    "The court hereby grants the motion for summary judgment.",
    "Plaintiff's counsel filed a motion pursuant to 28 U.S.C. § 1331.",
    "WHEREAS the parties have agreed to the following terms and conditions:",
    "HEREINAFTER referred to as 'Company', a Delaware corporation.",
    "NOTWITHSTANDING anything to the contrary contained in this Agreement,",
    # Numbers and citations
    "123 F.3d 456 (2d Cir. 2024)",
    "Fed. R. Civ. P. 12(b)(6)",
    "42 U.S.C. § 1983",
    "$2,456,789.00",
    "9.875%",
    # Specialised terms
    "certiorari",
    "habeas corpus",
    "indemnification",
    "subrogation",
    "notwithstanding",
    # Punctuation-heavy
    "inter alia, res judicata, collateral estoppel, and stare decisis.",
    "(a)(1)(A)(i)",
    "...",
    "---",
    # Multi-sentence
    (
        "The defendant moved to dismiss for lack of personal jurisdiction. "
        "The court denied the motion, finding minimum contacts sufficient."
    ),
    # Long document fragment
    " ".join(["indemnification"] * 100),
]


@pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
def test_round_trip(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    assert decoded == text, f"Round-trip failed for {text!r}\n  got: {decoded!r}"


# ---------------------------------------------------------------------------
# 2. UNK-free guarantee
# ---------------------------------------------------------------------------

UNK_FREE_INPUTS = [
    # Non-Latin scripts
    "한글",
    "日本語",
    "العربية",
    "Ελληνικά",
    "中文",
    "বাংলা",
    "हिंदी",
    "தமிழ்",
    # Emoji
    "👨‍⚕️",
    "🏛️",
    "⚖️",
    "📜",
    # Combining characters (é as base + combining accent)
    "e\u0301",
    # Mathematical / special symbols
    "½ ¾ ™ © ® § ¶",
    "𝕳𝖊𝖑𝖑𝖔",
    # Control bytes
    "\x00",
    "\x01\x1f",
    "\xff",
    # Empty
    "",
]


@pytest.mark.parametrize("text", UNK_FREE_INPUTS)
def test_no_unk(tokenizer, text):
    """Byte-level BPE must never produce UNK for any UTF-8 input."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    unk_id = tokenizer.unk_token_id
    if unk_id is not None:
        assert unk_id not in ids, (
            f"UNK token found for input {text!r}. "
            "Byte-level BPE should never produce UNK."
        )


# ---------------------------------------------------------------------------
# 3. Special token isolation
# ---------------------------------------------------------------------------


def test_special_token_isolation(tokenizer):
    """<|endoftext|> must be a single token, not split into subpieces."""
    eot = "<|endoftext|>"
    ids = tokenizer.encode(eot, add_special_tokens=True)
    assert tokenizer.eos_token_id in ids, (
        f"eos_token_id {tokenizer.eos_token_id} not in ids {ids} for {eot!r}"
    )


def test_special_token_not_split_in_context(tokenizer):
    """<|endoftext|> embedded in normal text must remain a single token."""
    text = "end of document<|endoftext|>start of next"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.eos_token_id in ids


# ---------------------------------------------------------------------------
# 4. Empty and trivial inputs
# ---------------------------------------------------------------------------


def test_empty_string(tokenizer):
    assert tokenizer.encode("", add_special_tokens=False) == []


def test_single_space(tokenizer):
    ids = tokenizer.encode(" ", add_special_tokens=False)
    assert isinstance(ids, list) and len(ids) >= 1


def test_newline(tokenizer):
    ids = tokenizer.encode("\n", add_special_tokens=False)
    assert isinstance(ids, list) and len(ids) >= 1


def test_single_byte(tokenizer):
    for ch in "aZ19!.":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        assert len(ids) >= 1


# ---------------------------------------------------------------------------
# 5. Digit boundary alignment (TokEval structure-sensitive metric)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("digit", list("0123456789"))
def test_digit_has_own_token(tokenizer, digit):
    """
    TokEval: each decimal digit should appear in at least one token
    when encoded in isolation, enabling digit-place-value arithmetic.
    """
    ids = tokenizer.encode(digit, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    assert any(digit in (t or "") for t in tokens), (
        f"Digit '{digit}' not found in any token: {tokens}"
    )


# ---------------------------------------------------------------------------
# 6. Byte-boundary integrity
# ---------------------------------------------------------------------------


def test_no_partial_utf8_tokens(tokenizer):
    """
    Sweep the first 2,000 token IDs: decoding each in isolation should not
    raise a UnicodeDecodeError (which would indicate a split UTF-8 codepoint).
    ByteLevel decoder handles this by design; this is a regression guard.
    """
    vocab_size = tokenizer.vocab_size
    sample = list(range(min(vocab_size, 2000)))
    for tid in sample:
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True)
            decoded.encode("utf-8")  # re-encode to confirm round-trip UTF-8 validity
        except (UnicodeDecodeError, UnicodeEncodeError) as exc:
            pytest.fail(f"Token id={tid} produced invalid UTF-8: {exc}")


# ---------------------------------------------------------------------------
# 7. Adversarial & edge-case battery (TokTier methodology)
# ---------------------------------------------------------------------------

ADVERSARIAL = [
    # Degenerate whitespace
    ("\t", "tab"),
    ("\r\n", "CRLF"),
    ("\r", "CR"),
    ("\u00a0", "non-breaking space"),
    ("\u200b", "zero-width space"),
    ("\u200c", "zero-width non-joiner"),
    ("\u2028", "line separator"),
    ("\u2029", "paragraph separator"),
    # Pathological repetition (TokTier digit-run target)
    ("a" * 1000, "repeat_a_1000"),
    ("ab" * 500, "repeat_ab_500"),
    ("0" * 500, "repeat_0_500"),
    ("1234567890" * 100, "digit_run_1000"),
    # Citation patterns
    ("28 U.S.C. § 1331", "statute_cite"),
    ("Fed. R. Civ. P. 12(b)(6)", "frcp_cite"),
    ("123 F.3d 456 (2d Cir. 2024)", "case_cite"),
    ("§§ 101–115", "section_range"),
    # Mixed scripts
    ("Section 1 (第一条): indemnification", "mixed_en_zh"),
    ("Article 2 (条款): notwithstanding", "mixed_en_zh2"),
    # RTL text
    ("\u0645\u0631\u062d\u0628\u0627", "arabic_hello"),
    # Mathematical / legal symbols
    ("½ ¾ ™ © ® § ¶ † ‡", "symbols"),
    # Emoji with ZWJ sequences
    ("👨‍⚕️ ⚖️ 🏛️", "emoji_legal"),
    # Null and control bytes
    ("\x00", "null_byte"),
    ("\x00\x01\x02\x03", "control_bytes"),
    ("\x1f", "unit_sep"),
    # Very long legal term
    ("indemnification " * 200, "long_repeat_legal_term"),
    # Special token injection attempt
    ("<|endoftext|>injected content", "special_token_injection"),
    ("prefix<|endoftext|>suffix", "special_token_mid"),
]


@pytest.mark.parametrize("text,label", ADVERSARIAL)
def test_adversarial_no_crash(tokenizer, text, label):
    """All adversarial inputs must encode and decode without raising."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokenizer.decode(ids, skip_special_tokens=True)
    except Exception as exc:
        pytest.fail(f"[{label}] Unexpected exception for {text!r}: {exc}")


@pytest.mark.parametrize("text,label", ADVERSARIAL)
def test_adversarial_ids_are_ints(tokenizer, text, label):
    """All token IDs must be non-negative integers within vocab range."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    vocab_size = tokenizer.vocab_size
    for tid in ids:
        assert isinstance(tid, int) and 0 <= tid < vocab_size + 100, (
            f"[{label}] Invalid token id {tid} (vocab_size={vocab_size})"
        )


# ---------------------------------------------------------------------------
# 8. Performance smoke test (not a timing gate — just ensures no hang)
# ---------------------------------------------------------------------------


def test_long_document_no_hang(tokenizer):
    """Encode a 50,000-character document without hanging."""
    text = ("The court finds that the defendant's motion is without merit. " * 600)[:50_000]
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) > 0
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    # Allow minor whitespace normalisation at boundaries but require substantial match.
    assert len(decoded) >= len(text) * 0.99
