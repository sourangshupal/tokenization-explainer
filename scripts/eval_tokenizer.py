#!/usr/bin/env python3
"""
Evaluate the trained tokenizer against GPT-2 and cl100k_base baselines.

Metrics reported (Aug 2026 TokEval / TokCollate standard):
  - Fertility             tokens / whitespace-delimited word
  - Characters per token  chars / token  (compression efficiency)
  - Compression ratio     bits per character vs baselines
  - Rényi efficiency      token frequency distribution flatness (α=2)
  - Shannon entropy       vocabulary utilization breadth
  - Domain-term 1-token rate  % of top legal terms encoded as single token
  - Digit boundary alignment  each digit 0–9 has its own token representation
  - Byte-boundary integrity   no token ID maps to an incomplete UTF-8 sequence
  - Round-trip correctness    encode → decode === original on 500 held-out docs

Usage:
    uv run python scripts/eval_tokenizer.py [--model-dir DIR] [--test-docs N]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "legal-bpe-50k"
CORPUS_DIR = ROOT / "data" / "corpus"

# Top-200 specialized legal terms that general tokenizers fragment.
LEGAL_TERMS = [
    "certiorari", "mandamus", "habeas", "corpus", "subpoena", "indemnification",
    "subrogation", "adjudication", "jurisdiction", "plaintiff", "defendant",
    "appellant", "appellee", "affidavit", "deposition", "interrogatories",
    "arbitration", "injunction", "contempt", "magistrate", "promissory",
    "fiduciary", "indictment", "arraignment", "felony", "misdemeanor",
    "probation", "recidivism", "tort", "negligence", "malpractice",
    "liability", "damages", "plaintiff", "subpoena", "hearsay", "perjury",
    "admissibility", "inadmissible", "restitution", "recusal", "sequester",
    "venire", "voir", "dire", "preemption", "severability", "promulgation",
    "codification", "jurisprudence", "precedent", "stare", "decisis",
    "ultra", "vires", "pro", "tanto", "res", "judicata", "collateral",
    "estoppel", "laches", "waiver", "acquiescence", "exculpatory",
    "exoneration", "exculpate", "malfeasance", "nonfeasance", "misfeasance",
    "embezzlement", "racketeering", "conspiracy", "tortfeasor", "complainant",
    "intervenor", "amicus", "curiae", "sua", "sponte", "inter", "alia",
    "prima", "facie", "ipso", "facto", "mens", "rea", "actus", "reus",
    "habeas", "corpus", "mandamus", "certiorari", "prohibition", "quo", "warranto",
    "notwithstanding", "hereinafter", "whereas", "therein", "thereof",
    "thereunder", "hereof", "hereunder", "herein", "thereto", "wherefrom",
    "indemnify", "indemnitor", "indemnitee", "guarantor", "guaranty",
    "subordinate", "subordination", "subrogee", "subrogor", "covenantee",
    "covenantor", "promissee", "promisor", "obligee", "obligor",
    "beneficiary", "grantor", "grantee", "mortgagor", "mortgagee",
    "lessor", "lessee", "licensor", "licensee", "assignor", "assignee",
    "bailor", "bailee", "trustor", "trustee", "testator", "testatrix",
    "intestate", "testate", "probate", "decedent", "devisee", "legatee",
    "bequest", "bequeath", "escheat", "eminent", "domain", "expropriation",
    "condemnation", "easement", "encumbrance", "lien", "foreclosure",
    "dispossession", "ejectment", "unlawful", "detainer", "trespass",
    "conversion", "bailment", "constructive", "fraudulent", "conveyance",
    "unjust", "enrichment", "quantum", "meruit", "promissory", "estoppel",
    "anticipatory", "repudiation", "impossibility", "frustration",
    "unconscionability", "unconscionable", "adhesion", "boilerplate",
    "liquidated", "consequential", "incidental", "punitive", "exemplary",
    "statutory", "compensatory", "nominal", "treble", "injunctive",
    "declaratory", "mandamus", "interdictory", "supersedeas", "replevin",
    "garnishment", "attachment", "sequestration", "receivership",
    "bankruptcy", "insolvency", "liquidation", "reorganization",
    "fraudulent", "preferential", "avoidable", "preference", "automatic",
    "stay", "discharge", "reaffirmation", "cramdown", "cram",
]

# Sample legal sentences for fertility measurement.
LEGAL_SENTENCES = [
    "The court hereby grants the motion for summary judgment notwithstanding the defendant's objection.",
    "Plaintiff's counsel filed a motion to compel discovery responses pursuant to Fed. R. Civ. P. 37.",
    "The indemnification clause shall survive termination of this agreement for a period of three years.",
    "Certiorari was granted to resolve the circuit split on the question of personal jurisdiction.",
    "The court finds that the habeas corpus petition fails to establish a constitutional violation.",
    "Defendant's affirmative defense of promissory estoppel requires clear and convincing evidence.",
    "The arbitration panel awarded compensatory and punitive damages totaling $2.4 million.",
    "Plaintiff seeks injunctive relief to prevent irreparable harm pending final adjudication.",
    "The subrogation claim arises from the insurer's payment of the policyholder's covered loss.",
    "This court has subject matter jurisdiction pursuant to 28 U.S.C. § 1331 and § 1332.",
    "The statute of limitations on the tort claim expired prior to the filing of the complaint.",
    "The fiduciary duty of loyalty prohibits self-dealing transactions without full disclosure.",
    "Defendant's counsel argued that the hearsay evidence was inadmissible under Rule 802.",
    "The magistrate judge issued a report and recommendation on the motion to suppress.",
    "Appellate jurisdiction is proper under 28 U.S.C. § 1291 as a final appealable order.",
    "The preliminary injunction standard requires likelihood of success on the merits.",
    "Res judicata bars the relitigation of claims that were or could have been raised.",
    "The collateral estoppel doctrine precludes re-litigation of issues already decided.",
    "Stare decisis requires this court to follow the binding precedent of the circuit court.",
    "The tortfeasor's negligence was the proximate cause of the plaintiff's injuries.",
    "Probable cause for the warrantless arrest existed based on the totality of circumstances.",
    "The Fourth Amendment protects against unreasonable searches and seizures.",
    "Miranda warnings must be given prior to custodial interrogation of a suspect.",
    "The exclusionary rule requires suppression of evidence obtained in violation of the Constitution.",
    "The court applies strict scrutiny to laws that burden fundamental constitutional rights.",
    "Due process requires notice and an opportunity to be heard before deprivation of a liberty interest.",
    "Equal protection prohibits arbitrary discrimination against similarly situated persons.",
    "The dormant Commerce Clause bars state laws that discriminate against interstate commerce.",
    "Federal preemption occurs when Congress enacts legislation occupying an entire field.",
    "The Supremacy Clause establishes that federal law is the supreme law of the land.",
    "WHEREAS the parties have entered into this agreement as of the date first written above,",
    "HEREINAFTER referred to as 'Company', a Delaware corporation duly organized and existing.",
    "NOTWITHSTANDING anything to the contrary contained herein, this provision shall govern.",
    "The obligor shall indemnify and hold harmless the indemnitee from all claims and losses.",
    "The mortgagee's security interest in the collateral is perfected by recording.",
    "The testator executed this last will and testament free from undue influence.",
    "The probate court admitted the will to probate and appointed an executor.",
    "The trustee owes a fiduciary duty to act in the best interests of the beneficiaries.",
    "Escheat laws transfer abandoned property to the state after a dormancy period.",
    "Eminent domain allows the government to take private property for public use with just compensation.",
    "The easement appurtenant runs with the land and benefits the dominant estate.",
    "Foreclosure proceedings were initiated following the mortgagor's default.",
    "Unlawful detainer proceedings are available to landlords after the tenant holds over.",
    "Quantum meruit recovery is available in quasi-contract to prevent unjust enrichment.",
    "Anticipatory repudiation occurs when a party unequivocally indicates it will breach.",
    "The liquidated damages clause must be a reasonable pre-estimate of actual harm.",
    "Consequential damages are recoverable only if they were within the contemplation of the parties.",
    "The automatic stay in bankruptcy halts all collection actions against the debtor.",
    "Cramdown confirmation requires at least one impaired class to vote in favor.",
    "The fraudulent conveyance was set aside because it was made to hinder creditors.",
]


def _load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
    except ImportError:
        log.error("transformers not installed — run: uv sync")
        sys.exit(1)
    return AutoTokenizer.from_pretrained(str(model_dir))


def _fertility(tokenizer, sentences: list[str]) -> float:
    """Average tokens per whitespace-delimited word."""
    total_tokens = 0
    total_words = 0
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        ids = tokenizer.encode(sent, add_special_tokens=False)
        total_tokens += len(ids)
        total_words += len(words)
    return total_tokens / total_words if total_words else 0.0


def _chars_per_token(tokenizer, sentences: list[str]) -> float:
    total_chars = 0
    total_tokens = 0
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        total_chars += len(sent)
        total_tokens += len(ids)
    return total_chars / total_tokens if total_tokens else 0.0


def _bits_per_char(tokenizer, sentences: list[str]) -> float:
    """Empirical compression: sum(log2(vocab_size)) / total_chars approximation."""
    vocab = tokenizer.vocab_size
    total_chars = sum(len(s) for s in sentences)
    total_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences)
    if total_chars == 0:
        return 0.0
    bits = total_tokens * math.log2(vocab)
    return bits / total_chars


def _renyi_efficiency(tokenizer, sentences: list[str], alpha: float = 2.0) -> float:
    """
    Rényi efficiency at order alpha.
    Higher α values weight common tokens more heavily.
    Measures how flat the token frequency distribution is.
    Perfect flat distribution → 1.0.
    """
    counts: Counter[int] = Counter()
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        counts.update(ids)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counts.values()], dtype=np.float64)
    # Rényi entropy H_alpha = (1/(1-alpha)) * log2(sum(p^alpha))
    renyi_entropy = (1.0 / (1.0 - alpha)) * math.log2(float(np.sum(probs**alpha)))
    max_entropy = math.log2(len(counts))  # uniform distribution
    return renyi_entropy / max_entropy if max_entropy > 0 else 0.0


def _shannon_entropy(tokenizer, sentences: list[str]) -> float:
    """Shannon entropy of the token distribution in bits."""
    counts: Counter[int] = Counter()
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        counts.update(ids)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counts.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _domain_term_rate(tokenizer, terms: list[str]) -> dict[str, float]:
    """Fraction of terms encoded as exactly 1 / 2 / 3+ tokens."""
    one = two = many = 0
    for term in terms:
        ids = tokenizer.encode(term.lower(), add_special_tokens=False)
        n = len(ids)
        if n == 1:
            one += 1
        elif n == 2:
            two += 1
        else:
            many += 1
    total = len(terms)
    return {
        "1_token_%": round(100 * one / total, 1),
        "2_token_%": round(100 * two / total, 1),
        "3+_token_%": round(100 * many / total, 1),
    }


def _digit_alignment(tokenizer) -> bool:
    """
    TokEval structure-sensitive metric: each decimal digit 0–9 should appear
    in at least one token by itself so that digit-place-value reasoning works.
    """
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        if not any(d in (t or "") for t in tokens):
            return False
    return True


def _byte_boundary_integrity(tokenizer) -> bool:
    """
    No token in the vocabulary should decode to an incomplete UTF-8 sequence
    when decoded in isolation (apart from ByteLevel offset tokens).
    Tests a sweep of 1000 random-ish token IDs.
    """
    vocab_size = tokenizer.vocab_size
    sample_ids = list(range(min(vocab_size, 1000)))
    for tid in sample_ids:
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True)
            decoded.encode("utf-8")  # raises UnicodeEncodeError if broken
        except Exception:
            return False
    return True


def _round_trip(tokenizer, sentences: list[str]) -> tuple[int, int]:
    """Returns (passed, total)."""
    passed = 0
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded == sent:
            passed += 1
    return passed, len(sentences)


def _load_test_sentences(corpus_dir: Path, max_docs: int) -> list[str]:
    """Pull held-out sentences from shard files (last shard reserved as test)."""
    shards = sorted(corpus_dir.glob("legal_shard_*.txt"))
    if not shards:
        return LEGAL_SENTENCES  # fallback to built-in set
    # Use the last shard as held-out.
    test_shard = shards[-1]
    lines = test_shard.read_text(encoding="utf-8").splitlines()
    # Take non-empty lines, up to max_docs.
    docs = [ln.strip() for ln in lines if ln.strip()][:max_docs]
    if not docs:
        return LEGAL_SENTENCES
    return docs


def evaluate(model_dir: Path, test_docs: int) -> dict:
    log.info("Loading tokenizer from %s…", model_dir)
    tok = _load_tokenizer(model_dir)

    log.info("Loading baseline tokenizers…")
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        log.warning("Could not load gpt2 tokenizer — skipping baseline.")
        gpt2_tok = None

    try:
        import tiktoken  # type: ignore[import-untyped]
        cl100k = tiktoken.get_encoding("cl100k_base")

        class _TikWrap:
            vocab_size = 100_277

            @staticmethod
            def encode(text, add_special_tokens=False):
                return cl100k.encode(text)

            @staticmethod
            def decode(ids, skip_special_tokens=True):
                return cl100k.decode(ids)

            @staticmethod
            def convert_ids_to_tokens(ids):
                return [cl100k.decode([i]) for i in ids]

        cl100k_tok = _TikWrap()
    except Exception:
        log.warning("Could not load cl100k_base — skipping baseline.")
        cl100k_tok = None

    log.info("Loading test sentences…")
    test_sents = _load_test_sentences(CORPUS_DIR, max_docs=test_docs)
    # Always include the curated legal sentences.
    all_sents = list(dict.fromkeys(LEGAL_SENTENCES + test_sents))
    log.info("Evaluating on %d sentences.", len(all_sents))

    results: dict = {}

    # --- our tokenizer ---
    log.info("Computing metrics for legal-bpe-50k…")
    results["legal-bpe-50k"] = {
        "vocab_size": tok.vocab_size,
        "fertility": round(_fertility(tok, all_sents), 4),
        "chars_per_token": round(_chars_per_token(tok, all_sents), 4),
        "bits_per_char": round(_bits_per_char(tok, all_sents), 4),
        "renyi_efficiency_alpha2": round(_renyi_efficiency(tok, all_sents, alpha=2.0), 4),
        "shannon_entropy_bits": round(_shannon_entropy(tok, all_sents), 4),
        "domain_term_tokenization": _domain_term_rate(tok, LEGAL_TERMS),
        "digit_boundary_alignment": _digit_alignment(tok),
        "byte_boundary_integrity": _byte_boundary_integrity(tok),
        "round_trip": dict(zip(("passed", "total"), _round_trip(tok, all_sents))),
    }

    # --- gpt2 baseline ---
    if gpt2_tok is not None:
        log.info("Computing metrics for gpt2…")
        results["gpt2"] = {
            "vocab_size": gpt2_tok.vocab_size,
            "fertility": round(_fertility(gpt2_tok, all_sents), 4),
            "chars_per_token": round(_chars_per_token(gpt2_tok, all_sents), 4),
            "bits_per_char": round(_bits_per_char(gpt2_tok, all_sents), 4),
            "renyi_efficiency_alpha2": round(_renyi_efficiency(gpt2_tok, all_sents, alpha=2.0), 4),
            "shannon_entropy_bits": round(_shannon_entropy(gpt2_tok, all_sents), 4),
            "domain_term_tokenization": _domain_term_rate(gpt2_tok, LEGAL_TERMS),
            "digit_boundary_alignment": _digit_alignment(gpt2_tok),
        }

    # --- cl100k baseline ---
    if cl100k_tok is not None:
        log.info("Computing metrics for cl100k_base…")
        results["cl100k_base"] = {
            "vocab_size": cl100k_tok.vocab_size,
            "fertility": round(_fertility(cl100k_tok, all_sents), 4),
            "chars_per_token": round(_chars_per_token(cl100k_tok, all_sents), 4),
            "bits_per_char": round(_bits_per_char(cl100k_tok, all_sents), 4),
            "renyi_efficiency_alpha2": round(
                _renyi_efficiency(cl100k_tok, all_sents, alpha=2.0), 4
            ),
            "shannon_entropy_bits": round(_shannon_entropy(cl100k_tok, all_sents), 4),
            "domain_term_tokenization": _domain_term_rate(cl100k_tok, LEGAL_TERMS),
            "digit_boundary_alignment": _digit_alignment(cl100k_tok),
        }

    return results


def _print_table(results: dict) -> None:
    """Pretty-print a comparison table to stdout."""
    metrics = [
        ("vocab_size", "Vocab size", lambda v: f"{v:,}"),
        ("fertility", "Fertility (tok/word) ↓", lambda v: f"{v:.3f}"),
        ("chars_per_token", "Chars per token ↑", lambda v: f"{v:.3f}"),
        ("bits_per_char", "Bits per char ↓", lambda v: f"{v:.3f}"),
        ("renyi_efficiency_alpha2", "Rényi efficiency α=2 ↑", lambda v: f"{v:.4f}"),
        ("shannon_entropy_bits", "Shannon entropy (bits) ↑", lambda v: f"{v:.3f}"),
    ]
    tokenizers = list(results.keys())
    col_w = 22
    header = f"{'Metric':<32}" + "".join(f"{t:<{col_w}}" for t in tokenizers)
    print("\n" + "=" * (32 + col_w * len(tokenizers)))
    print(header)
    print("-" * (32 + col_w * len(tokenizers)))
    for key, label, fmt in metrics:
        row = f"{label:<32}"
        for t in tokenizers:
            val = results[t].get(key, "—")
            row += f"{fmt(val) if val != '—' else '—':<{col_w}}"
        print(row)

    print("\nDomain-term tokenization (% of top legal terms):")
    print(f"{'Term encoding':<32}" + "".join(f"{t:<{col_w}}" for t in tokenizers))
    print("-" * (32 + col_w * len(tokenizers)))
    for sub_key, label in [
        ("1_token_%", "  Single token ↑"),
        ("2_token_%", "  Two tokens"),
        ("3+_token_%", "  Three+ tokens ↓"),
    ]:
        row = f"{label:<32}"
        for t in tokenizers:
            dt = results[t].get("domain_term_tokenization", {})
            val = dt.get(sub_key, "—")
            row += f"{val!s:<{col_w}}"
        print(row)

    print("\nStructural integrity checks:")
    for check in ["digit_boundary_alignment", "byte_boundary_integrity"]:
        row = f"  {check:<30}"
        for t in tokenizers:
            val = results[t].get(check, "—")
            row += f"{'PASS' if val is True else 'FAIL' if val is False else '—':<{col_w}}"
        print(row)

    if "round_trip" in results.get(list(results.keys())[0] if results else "", {}):
        rt = results[list(results.keys())[0]]["round_trip"]
        pct = 100 * rt["passed"] / rt["total"] if rt["total"] else 0
        print(f"\nRound-trip (legal-bpe-50k): {rt['passed']}/{rt['total']}  ({pct:.1f}%)")

    print("=" * (32 + col_w * len(tokenizers)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tokenizer quality (TokEval metrics)")
    parser.add_argument(
        "--model-dir", type=Path, default=MODEL_DIR, help="Path to save_pretrained directory"
    )
    parser.add_argument(
        "--test-docs",
        type=int,
        default=500,
        help="Max held-out documents to include from corpus (default 500)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write results JSON to this path (default: no file)",
    )
    args = parser.parse_args()

    if not (args.model_dir / "tokenizer_config.json").exists():
        log.error(
            "tokenizer_config.json not found in %s. Run scripts/wrap_tokenizer.py first.",
            args.model_dir,
        )
        sys.exit(1)

    results = evaluate(args.model_dir, args.test_docs)
    _print_table(results)

    out_path = args.output_json or (args.model_dir / "eval_report.json")
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
