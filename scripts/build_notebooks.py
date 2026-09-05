#!/usr/bin/env python3
"""Generate the medical custom-vs-general notebook. Run: uv run python scripts/build_notebooks.py"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def write(name: str, cells: list[nbf.NotebookNode], title: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = cells
    path = NB_DIR / name
    nbf.write(nb, path)
    print(f"wrote {path} ({len(cells)} cells) {title}")


def custom_vs_general_notebook() -> list[nbf.NotebookNode]:
    return [

        # ── 0. Title ──────────────────────────────────────────────────────────
        md("""
# Lab 04 — Custom medical BPE vs general-purpose tokenizers

**The big question:** Does training a tokenizer on medical text actually produce fewer pieces
on medical strings than a production general-purpose tokenizer?  And can we prove it *fairly*?

Theory to read first (or alongside):
- [00 overview](../docs/theory/00-overview.md)
- [01 why tokenization matters](../docs/theory/01-why-tokenization-matters.md)
- [02 general-purpose tokenizers](../docs/theory/02-general-purpose-tokenizers.md)
- [03 custom domain tokenizers](../docs/theory/03-custom-domain-tokenizers.md)
- [04 metrics](../docs/theory/04-metrics.md) ← start here for the fairness design
- [05 why custom wins in healthcare](../docs/theory/05-why-custom-wins-in-healthcare.md)
- [06 the pretrained-model trap](../docs/theory/06-pretrained-model-trap.md) ← **required**
- [08 vocab-size trade-off](../docs/theory/08-vocab-size-tradeoff.md) ← experiment after the 4-way compare

> **Reminder:** custom tokenizer = better *segmentation* on medical text.
> It is **not** a drop-in replacement for a pretrained LLM's tokenizer.
"""),

        # ── 1. What is a token? (beginner anchor) ─────────────────────────────
        md("""
## What is a token?

A **token** is the smallest unit a language model sees.  Models don't read characters or
words — they read token IDs, and each ID indexes one row of an embedding table.

```
"empagliflozin 10 mg daily"
                             ↓ general tokenizer (cl100k)
 emp | ag | l | if | lo | zin |  10 |  mg |  daily    ← 9 pieces

                             ↓ medical tokenizer (custom-med)
 em | pa | gl | if | lo | z | in | Ġ10 | Ġmg | Ġdaily ← 10 pieces (tie here; wins on others)
```

**Fertility** = tokens ÷ whitespace words.  Lower fertility = denser encoding = fewer tokens
needed to say the same thing = more text fits in the model's context window.

> **Note on `Ġ`:** Byte-level BPE marks a leading space as `Ġ` (capital G with cedilla).
> `Ġ10` means " 10" (space + 10).  It is not an error — it is how GPT-style tokenizers
> encode word boundaries.
"""),

        # ── 2. Setup ──────────────────────────────────────────────────────────
        md("""
## Setup

Run this cell first.  It finds the repo root, imports helpers, and checks which data files
and trained artifacts are present.
"""),
        code(r'''
from __future__ import annotations
from pathlib import Path
import sys

from IPython.display import display, Markdown

# ── Find repo root ─────────────────────────────────────────────────────────
def find_root() -> Path:
    here = Path.cwd()
    for candidate in [here, here.parent]:
        if (candidate / "data" / "medical_corpus.txt").exists():
            return candidate
    raise FileNotFoundError("Run from the repo root or the notebooks/ folder.")

ROOT = find_root()
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Display helpers (token chips, tables, charts) ──────────────────────────
from lab_display import (
    ascii_bar,               # single-series ASCII bars (vocab-size experiment)
    ascii_fertility_chart,   # ASCII bar chart comparing fertility across domains
    compare_table,           # side-by-side Rich table
    ok,                      # green success panel
    show_pieces,             # print token chips in a panel
    side_by_side_segs,       # show all tokenizers on one probe
    vocab_membership,        # which terms exist as single tokens
)

# ── Metric + tokenizer loading functions ──────────────────────────────────
from compare_tokenizers import (
    DOMAIN_TERMS,            # 20 medical terms for single-token rate
    build_columns,           # load all tokenizers into a dict
    fertility,               # n_tokens / n_words
    iter_jsonl_texts,        # read JSONL held-out file
    load_custom_encode,      # load a tokenizer.json
    load_tiktoken_encode,    # load cl100k or o200k
    mean_piece_count,        # mean pieces per string
    read_lines,              # read a plain-text list file
    single_token_rate,       # fraction of terms encoded as 1 piece
    tokens_per_100_chars,    # tokens per 100 chars (word-count-independent fertility)
    winner,                  # which tokenizer used fewest pieces
)
from sweep_vocab_size import (
    D_MODEL,                 # 1024 — embedding-cost stand-in
    attach_deltas,
    corpus_band,
    embedding_params,
    estimate_corpus_tokens,
    load_available_sweep_rows,
    recommend_vocab,
    size_label,
)
from train_medical_tokenizer import train_byte_level_bpe

# ── Paths ──────────────────────────────────────────────────────────────────
PROBES          = ROOT / "data" / "medical_probes.txt"
CONTROL         = ROOT / "data" / "medical_control.txt"
PUBMED_HELDOUT  = ROOT / "data" / "pubmed_heldout.jsonl"
GENERAL_HELDOUT = ROOT / "data" / "general_heldout.jsonl"
MED_BPE_JSON    = ROOT / "artifacts" / "medical-bpe-pubmed" / "tokenizer.json"
MED_BPE_32K     = ROOT / "artifacts" / "medical-bpe-pubmed-32k" / "tokenizer.json"
PUBMED_TRAIN    = ROOT / "data" / "pubmed_train.jsonl"
TINY_CORPUS     = ROOT / "data" / "medical_corpus.txt"
GENERAL_BPE_JSON= ROOT / "artifacts" / "general-bpe" / "tokenizer.json"
FALLBACK_JSON   = ROOT / "models" / "pretrained" / "medical_bpe_tiny" / "tokenizer.json"
SWEEP_SIZES     = (16_000, 32_000, 50_000, 64_000, 100_000)

# ── Status check ───────────────────────────────────────────────────────────
for label, path in [
    ("custom-med 16k  ", MED_BPE_JSON),
    ("custom-med 32k  ", MED_BPE_32K),
    ("custom-med 50k  ", ROOT / "artifacts" / "medical-bpe-pubmed-50k" / "tokenizer.json"),
    ("custom-med 64k  ", ROOT / "artifacts" / "medical-bpe-pubmed-64k" / "tokenizer.json"),
    ("custom-med 100k ", ROOT / "artifacts" / "medical-bpe-pubmed-100k" / "tokenizer.json"),
    ("general-bpe 16k ", GENERAL_BPE_JSON),
    ("PubMed train    ", PUBMED_TRAIN),
    ("PubMed heldout  ", PUBMED_HELDOUT),
    ("General heldout ", GENERAL_HELDOUT),
]:
    status = "✓ found" if path.is_file() else "✗ not found (some cells will skip)"
    print(f"  {label}: {status}")
'''),

        # ── 3. Load tokenizers ────────────────────────────────────────────────
        md("""
## Load tokenizers

We compare **four** tokenizers:

| name | vocab size | training data | purpose |
|------|-----------|--------------|---------|
| `cl100k` | ~100k | general web (GPT-4) | real-world general baseline |
| `o200k` | ~200k | general web (GPT-4o) | stronger general baseline |
| `custom-med` | **16k** | **PubMed abstracts** | the domain tokenizer |
| `general-bpe` | **16k** | wikitext-103 | **fairness control** — same size as custom-med |

The `general-bpe` column is critical: if `custom-med` beats it on medical text,
the win comes from **domain**, not from having a smaller or different vocabulary.
"""),
        code(r'''
# ── Pick custom-med source ─────────────────────────────────────────────────
if MED_BPE_JSON.is_file():
    custom_med_json = MED_BPE_JSON
    med_source = "trained on 45k PubMed abstracts"
elif FALLBACK_JSON.is_file():
    custom_med_json = FALLBACK_JSON
    med_source = "fallback (tiny authored corpus — limited coverage)"
else:
    raise FileNotFoundError(
        "No custom-med tokenizer found.\n"
        "Run:  uv run python scripts/split_corpus.py\n"
        "Then: uv run python scripts/train_medical_tokenizer.py "
        "--corpus data/pubmed_train.jsonl --out artifacts/medical-bpe-pubmed/tokenizer.json"
    )

# ── Build column dict: {name → encode_fn} ─────────────────────────────────
columns = build_columns(
    custom_med_json,
    include_qwen=False,
    general_bpe_json=GENERAL_BPE_JSON if GENERAL_BPE_JSON.is_file() else None,
)

ok(
    f"Loaded {len(columns)} tokenizers: {list(columns.keys())}\n"
    f"custom-med source: {med_source}"
)
'''),

        # ── 4. Predict, then reveal (headline probe) ──────────────────────────
        md("""
## Step 1 — Predict, then reveal

Before running the next cell, **write down** your guesses:

- How many pieces does `cl100k` (GPT-4 tokenizer) split `empagliflozin 10 mg daily` into?
- How many pieces does the custom medical tokenizer use?

**empagliflozin** is an SGLT2-inhibitor — a type 2 diabetes drug.  It is a long, rare word on
the general web but appears in hundreds of thousands of PubMed abstracts.  That frequency
difference is exactly what BPE exploits.

> Run the cell and see if your prediction was right.
"""),
        code(r'''
PROBE = "empagliflozin 10 mg daily"

# Show all tokenizers on the same probe — colored chip for each token
side_by_side_segs(PROBE, columns)

# Commentary
cl_n  = len(columns["cl100k"](PROBE))
med_n = len(columns["custom-med"](PROBE))

if med_n < cl_n:
    ok(f"custom-med wins: {med_n} pieces vs cl100k {cl_n} pieces  (−{cl_n - med_n} tokens)")
else:
    # At 16k vocab, empagliflozin may tie — vocab budget goes to higher-freq terms first
    print(
        f"custom-med: {med_n} pieces  |  cl100k: {cl_n} pieces\n"
        f"→ Tie or loss on this probe is expected at 16k vocab — see 03-custom-domain-tokenizers.md\n"
        f"  for why: vocab budget goes to higher-frequency medical terms first."
    )
'''),

        # ── 5. Side-by-side segmentation (5 key probes) ───────────────────────
        md("""
## Step 2 — Side-by-side segmentation

Each colored box = one token.  **Fewer boxes = better compression for that string.**

These 5 probes are chosen because they appear frequently in PubMed but rarely on the general web.
Notice how `general-bpe` (same 16k vocab as custom-med) fragments even more than `cl100k` —
proof that vocab *size* is not the issue; it is training *domain*.

> `Ġ` = leading space (byte-level BPE convention, normal and expected).
"""),
        code(r'''
SHOW_PROBES = [
    "acetylcholinesterase inhibitor",    # compound clinical noun — best win: 3 vs 7
    "ST-elevation myocardial infarction",# STEMI — very common in cardiology abstracts
    "hemoglobin A1c 8.2%",               # HbA1c — in every diabetes abstract
    "serum creatinine 1.4 mg/dL",        # lab + unit — metabolic panels
    "metformin 1000 mg twice daily",     # drug + dose pattern
]

for probe in SHOW_PROBES:
    side_by_side_segs(probe, columns)
'''),

        # ── 6. All 20 probes ──────────────────────────────────────────────────
        md("""
## Step 3 — All 20 curated probes

The full probe list with winner column.

**These are illustration examples, not the pass bar.** Some probes tie or lose —
that is honest and explained in [05](../docs/theory/05-why-custom-wins-in-healthcare.md).
The real proof is the held-out fertility eval in Step 5.
"""),
        code(r'''
probes = read_lines(PROBES)
headers = ["probe"] + list(columns) + ["winner"]
table_rows = []
for probe in probes:
    ns = {name: len(enc(probe)) for name, enc in columns.items()}
    table_rows.append([probe, *[ns[n] for n in columns], winner(ns)])

compare_table(
    headers, table_rows,
    title="20 curated probes — piece counts per tokenizer",
    caption="'winner' = fewest pieces. Ties are honest: some terms are already in the general vocab.",
)
'''),

        # ── 7. Single-token rate ──────────────────────────────────────────────
        md("""
## Step 4 — Single-token rate: does the term exist in the vocabulary?

If a tokenizer encodes a term as **exactly 1 piece**, that term is a single entry in its
vocabulary — one atomic concept, one embedding row.

If it needs 5 pieces, the model must reassemble the concept from fragments every time it
appears — harder to learn, more tokens wasted.

Watch: `cl100k` and `o200k` encode **zero** of these 20 medical terms as a single token.
"""),
        code(r'''
from rich.console import Console
from rich import box
from rich.table import Table

c = Console(force_jupyter=True, width=110)

table = Table(
    title="Single-token rate on 20 medical domain terms",
    caption="0% means every term is fragmented into 2+ pieces (no single-token concept exists)",
    box=box.SIMPLE_HEAD,
)
table.add_column("tokenizer")
table.add_column("single-token rate", justify="right")
table.add_column("terms encoded as 1 piece", justify="right")
table.add_column("which terms (if any)")

for name, encode in columns.items():
    singles = [t for t in DOMAIN_TERMS if len(encode(t)) == 1]
    rate = len(singles) / len(DOMAIN_TERMS)
    table.add_row(
        name,
        f"{rate:.1%}",
        f"{len(singles)}/{len(DOMAIN_TERMS)}",
        ", ".join(singles) if singles else "none",
    )

c.print(table)
'''),

        # ── 8. Vocab membership (mechanism) ───────────────────────────────────
        md("""
## Step 4b — Vocab membership: the mechanism

This table asks a simple question for each medical term and each tokenizer:
**"Is this word a single token in your vocabulary?"**

If ✓ — the tokenizer has seen that word often enough during training to give it its own slot.
If ✗ with a number — the tokenizer has to split it into that many fragments.

This is *why* custom-med works: PubMed abstracts mention these terms thousands of times,
so BPE merges them into whole tokens.  The general web doesn't, so cl100k/o200k/general-bpe never get the merges.
"""),
        code(r'''
KEY_TERMS = [
    # Drug names
    "empagliflozin", "metformin", "vancomycin", "levothyroxine", "oseltamivir",
    # Lab / clinical terms
    "creatinine", "troponin", "hemoglobin", "lymphocyte", "myocardial",
    # Disease / pathology terms
    "thrombocytopenia", "cardiomyopathy", "corticosteroid", "pneumonia", "hypertension",
]

vocab_membership(KEY_TERMS, columns)
'''),

        # ── 9. Held-out fertility (the fair eval) ─────────────────────────────
        md("""
## Step 5 — Held-out fertility: the fair eval

> **This is the most important cell in the lab.**

Previous steps used curated probes — strings we hand-picked and that overlap with the
training data.  That is illustration, not proof.

Here we evaluate on **5000 PubMed abstracts that no tokenizer has seen** (held-out split).

**Reading the numbers:**
- Fertility of **1.40** means on average each whitespace word becomes 1.40 tokens.
- Lower is better.  Difference of 0.08 over a 200-word abstract = ~16 fewer tokens.
- Over a 1000-token context window that compounds to real capacity gains.
"""),
        code(r'''
# ── Medical held-out (5 000 PubMed abstracts, never seen during training) ────
if PUBMED_HELDOUT.is_file():
    med_texts = iter_jsonl_texts(PUBMED_HELDOUT, max_docs=2000)
    print(f"Evaluating on {len(med_texts)} held-out PubMed abstracts ...\n")

    rows = []
    med_fertilities = {}

    for name, encode in columns.items():
        pieces = [encode(t) for t in med_texts]
        fert   = fertility(pieces, med_texts)
        t100   = tokens_per_100_chars(pieces, med_texts)
        mean_p = sum(len(p) for p in pieces) / max(1, len(pieces))
        rows.append([name, f"{fert:.3f}", f"{t100:.2f}", f"{mean_p:.0f}"])
        med_fertilities[name] = fert

    compare_table(
        ["tokenizer", "fertility ↓", "tok/100ch ↓", "mean tokens/abstract ↓"],
        rows,
        title="FAIR EVAL — 2000 held-out PubMed abstracts",
        caption=(
            "↓ lower is better.  "
            "Fertility = tokens ÷ whitespace-words.  "
            "custom-med should have the lowest value."
        ),
    )

    best = min(med_fertilities, key=med_fertilities.get)
    second = sorted(med_fertilities, key=med_fertilities.get)[1]
    gap = med_fertilities[second] - med_fertilities[best]
    ok(
        f"Medical held-out winner: {best}  (fertility {med_fertilities[best]:.3f})\n"
        f"Gap vs next best ({second}): {gap:.3f}  "
        f"≈ {gap / med_fertilities[second] * 100:.1f}% fewer tokens on every abstract"
    )
else:
    print("⚠ pubmed_heldout.jsonl not found.")
    print("  Run:  uv run python scripts/split_corpus.py")
    print("  Then re-run this cell.")
'''),

        # ── 10. General held-out ──────────────────────────────────────────────
        md("""
## Step 5b — General held-out: the cross-domain check

A good domain tokenizer should **lose** on general English.  If it wins everywhere,
it was not actually specialised — the medical corpus leaked too much general prose.

custom-med spending its 16k merges on medicine means it has fewer merges left for
everyday English.  That is the expected and honest result.
"""),
        code(r'''
if GENERAL_HELDOUT.is_file():
    gen_texts = iter_jsonl_texts(GENERAL_HELDOUT, max_docs=2000)
    print(f"Evaluating on {len(gen_texts)} held-out wikitext paragraphs ...\n")

    rows_gen = []
    gen_fertilities = {}

    for name, encode in columns.items():
        pieces = [encode(t) for t in gen_texts]
        fert   = fertility(pieces, gen_texts)
        t100   = tokens_per_100_chars(pieces, gen_texts)
        mean_p = sum(len(p) for p in pieces) / max(1, len(pieces))
        rows_gen.append([name, f"{fert:.3f}", f"{t100:.2f}", f"{mean_p:.0f}"])
        gen_fertilities[name] = fert

    compare_table(
        ["tokenizer", "fertility ↓", "tok/100ch ↓", "mean tokens/paragraph ↓"],
        rows_gen,
        title="Cross-domain check — 2000 held-out wikitext paragraphs",
        caption=(
            "custom-med is expected to lose here — that is a pass, not a fail."
        ),
    )

    best_gen = min(gen_fertilities, key=gen_fertilities.get)
    print(f"\nGeneral held-out winner: {best_gen}  (fertility {gen_fertilities[best_gen]:.3f})")
    print(f"custom-med fertility:    {gen_fertilities.get('custom-med', '—')}")
    print(f"→ custom-med loses general English: expected.  It spent vocab on medicine.")
else:
    print("⚠ general_heldout.jsonl not found.")
    print("  Run:  uv run python scripts/download_general_sample.py")
    print("  Then re-run this cell.")
'''),

        # ── 11. 2×2 chart ────────────────────────────────────────────────────
        md("""
## Step 6 — The 2×2 payoff chart

This is the full picture:

- **Short bar on medical + long bar on general** = good domain tokenizer (custom-med)
- **Short bar on both** = large general vocab tokenizer (cl100k, o200k)
- **Long bar on both** = small general-domain BPE (general-bpe) — same size as custom-med
  but wrong domain, loses everywhere against the 100k+ tokenizers

Lower bar = lower fertility = fewer tokens = better.
"""),
        code(r'''
# Use held-out sets if computed above; fall back to curated probes/control
chart_data: dict[str, dict[str, float]] = {}

med_fert_available = "med_fertilities" in dir() and bool(med_fertilities)
gen_fert_available = "gen_fertilities" in dir() and bool(gen_fertilities)

if med_fert_available and gen_fert_available:
    for name in columns:
        chart_data[name] = {
            "medical (held-out)": med_fertilities.get(name, 0.0),
            "general (held-out)": gen_fertilities.get(name, 0.0),
        }
else:
    # Offline fallback
    probes  = read_lines(PROBES)
    control = read_lines(CONTROL)
    for name, encode in columns.items():
        p_pieces = [encode(t) for t in probes]
        c_pieces = [encode(t) for t in control]
        chart_data[name] = {
            "medical (probes)":  fertility(p_pieces, probes),
            "general (control)": fertility(c_pieces, control),
        }

ascii_fertility_chart(chart_data, title="Fertility by tokenizer and domain  (lower bar = better)")
'''),

        # ── 12. Control spot-check ────────────────────────────────────────────
        md("""
## Step 7 — Control: ordinary English spot-check

Quick check on `data/medical_control.txt` (40 general English sentences).
custom-med is expected to lose or tie here.
"""),
        code(r'''
control = read_lines(CONTROL)
rows_ctrl = []
for name, encode in columns.items():
    pieces = [encode(t) for t in control]
    fert   = fertility(pieces, control)
    t100   = tokens_per_100_chars(pieces, control)
    rows_ctrl.append([name, f"{fert:.3f}", f"{t100:.2f}"])

compare_table(
    ["tokenizer", "fertility", "tok/100ch"],
    rows_ctrl,
    title="General-English control (40 sentences)",
    caption="custom-med losing here = healthy specialisation, not a bug.",
)

# Visual: show chips for one control sentence
sample = control[0]
print(f'\nControl sentence: "{sample[:70]}..."\n')
for name, encode in columns.items():
        show_pieces(encode(sample), title=f"{name}")
'''),

        # ── 12b. Vocab-size experiment ───────────────────────────────────────
        md("""
## Experiment — which vocab size? (read [08](../docs/theory/08-vocab-size-tradeoff.md) first)

Domain already won. Next question: **given the same PubMed train set, which vocab budget?**

There is no single optimal size. Best size is a trade-off between **compression**
(fewer tokens) and **vocabulary size** (more embedding rows).

> 🔬 **Diagram:** [diminishing returns + embedding cost](../docs/diagrams/06-vocab-size-knee.md)

### Step 0 — Predict (write answers before running the next cells)

1. Will 100k always beat 16k on avg tokens/doc? Why or why not?
2. This train set is 45k abstracts (tens of millions of tokens, **not** billions).
   Circle a band: **16k–32k** / 32k–64k / 50k–100k.

Corpus-size rule:

| Train-set scale | Recommended vocab |
|-----------------|-------------------|
| Small (<1B tokens) | **16k–32k** |
| Medium (1–50B) | 32k–64k |
| Very large (100B+) | 50k–100k |

Modern LLM default is 32k–64k (50k often a sweet spot). **That default is for large
pretraining corpora, not this lab.**
"""),
        md("""
### Step 1 — Same corpus, five sizes

Instructor trained 16k / 32k / 50k / 64k / 100k on `data/pubmed_train.jsonl`
(`scripts/sweep_vocab_size.py --skip-existing`). This cell does **not** train 100k live.

Tiny demo below trains 512 vs 1024 on the authored corpus so you see the API once.
"""),
        code(r'''
demo_dir = ROOT / "artifacts" / "_demo_vocab"
for size in (512, 1024):
    out = demo_dir / f"v{size}" / "tokenizer.json"
    if out.is_file():
        print(f"reuse demo {size}: {out}")
        continue
    train_byte_level_bpe(TINY_CORPUS, out, vocab_size=size, min_frequency=2)
    print(f"trained demo {size}: {out}")

from tokenizers import Tokenizer
for size in (512, 1024):
    path = demo_dir / f"v{size}" / "tokenizer.json"
    tok = Tokenizer.from_file(str(path))
    print(f"  requested {size:>4d}  actual {tok.get_vocab_size()}")
'''),
        md("""
### Steps 2–3 — Measure avg tokens/doc and read diminishing returns

Headline metric: **average tokens per held-out document**. Smaller = better compression.

If sweep artifacts are missing, the table falls back to the **worked example**
(1250 / 1050 / 980 / 960 / 950). Caption will say so — those numbers are from a
*large* corpus. Your PubMed knee should sit left of that.
"""),
        code(r'''
sweep_texts: list[str] = []
if PUBMED_HELDOUT.is_file():
    sweep_texts = iter_jsonl_texts(PUBMED_HELDOUT, max_docs=2000)
    print(f"held-out docs: {len(sweep_texts)}")
else:
    print("pubmed_heldout.jsonl not found — using worked-example numbers")

sweep_rows, used_example = load_available_sweep_rows(ROOT, sweep_texts, sizes=SWEEP_SIZES)
caption = (
    "Worked example from a large pretraining corpus — your PubMed numbers will differ."
    if used_example else
    "Live sweep on held-out PubMed abstracts. Look at the last jump: is 100k worth it?"
)

table_rows = []
for row in sweep_rows:
    vs_prev = "—" if row.vs_previous_pct is None else f"{row.vs_previous_pct:+.1f}%"
    vs_16k = "—" if row.vs_base_pct is None else f"{row.vs_base_pct:+.1f}%"
    table_rows.append([
        row.name,
        f"{row.mean_tokens_per_doc:,.1f}",
        vs_prev,
        vs_16k,
        f"{row.actual:,}" + ("*" if row.saturated else ""),
        f"{row.fertility:.3f}",
    ])

compare_table(
    ["vocab", "avg tokens/doc", "vs previous", "vs 16k", "actual vocab", "fertility"],
    table_rows,
    title="Diminishing returns — avg tokens / document",
    caption=caption,
)

ascii_bar(
    {row.name: row.mean_tokens_per_doc for row in sweep_rows},
    title="Avg tokens/doc by vocab size",
)

if any(r.saturated for r in sweep_rows):
    print("* actual < requested — BPE ran out of min_frequency=2 merges.")
'''),
        md("""
### Step 4 — Embedding cost

`embedding params ≈ vocab_size × d_model`. Extra vocab is extra rows in **both**
the input embedding and the output softmax. `d_model = 1024` is a Qwen-scale stand-in.
"""),
        code(r'''
emb_rows = []
base = embedding_params(SWEEP_SIZES[0], D_MODEL)
for size in SWEEP_SIZES:
    n = embedding_params(size, D_MODEL)
    emb_rows.append([size_label(size), f"{n:,}", f"{n / base:.2f}×"])

compare_table(
    ["vocab", "embedding params", "vs 16k"],
    emb_rows,
    title=f"Embedding cost  (vocab × d_model={D_MODEL})",
    caption="100k is 6.25× the 16k table. After the knee you buy those rows for ~1% fewer tokens.",
)
'''),
        md("""
### Step 5 — Apply the corpus-size rule

Whitespace-word count on the train file is a proxy (not model tokens). Small corpus
(<1B) → **16k–32k**. 50k as a "sweet spot" is for medium/large pretraining.
"""),
        code(r'''
if PUBMED_TRAIN.is_file():
    n_tok = estimate_corpus_tokens(PUBMED_TRAIN)
    source = str(PUBMED_TRAIN)
else:
    n_tok = 10_000_000  # order-of-magnitude stand-in when the jsonl is absent
    source = "proxy (~10M) — pubmed_train.jsonl not in tree"

band = corpus_band(n_tok)
print(f"train tokens (whitespace proxy): {n_tok:,}   from {source}")
print(f"corpus band: {band}")
print("small <1B → 16k–32k | medium 1–50B → 32k–64k | very large 100B+ → 50k–100k")
'''),
        md("""
### Step 6 — Pick a size

> Choose the **smallest** vocabulary that achieves **near-minimum** tokenization length.
> Once token count stops improving much, stop. Extra vocab is extra parameters.

This pick is for training a tokenizer / a from-scratch model. It does **not** mean
you can load that tokenizer onto Qwen. Write one sentence: size, why (table + band + cost).
"""),
        code(r'''
pairs = [(row.requested, row.mean_tokens_per_doc) for row in sweep_rows]
pick = recommend_vocab(pairs, threshold=0.02)
ok(
    f"Knee helper: smallest vocab within 2% of best avg tokens/doc → {size_label(pick)}\n"
    f"Corpus band: {band}\n"
    "If they disagree, trust the small-corpus rule plus embedding cost — not the last 1%."
)
'''),

        # ── 13. The trap ─────────────────────────────────────────────────────
        md("""
## The trap — do not skip this

**Board sentence (write it down):**

> *Tokenizer IDs must match embedding rows.  Custom BPE changes the alphabet.
> A pretrained model's embeddings still speak the old alphabet.*

The lab proves that custom-med segments medical text better.  It does **not** mean
you can swap it onto Qwen or Llama and expect the model to understand medical text.
Read [06 — the pretrained-model trap](../docs/theory/06-pretrained-model-trap.md) for
why that swap silently scrambles the model.
"""),
        code(r'''
display(Markdown("""
### What NOT to do

```python
# ✗ WRONG — do not do this
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tok   = AutoTokenizer.from_pretrained("artifacts/medical-bpe-pubmed")  # new vocab
# LoRA on attention layers — IDs no longer map to the right embeddings
```

### What IS correct

```python
# ✓ RIGHT option A: keep Qwen's tokenizer, add specific medical tokens
tokenizer.add_tokens(["empagliflozin", "acetylcholinesterase", ...])
model.resize_token_embeddings(len(tokenizer))
# then LoRA — you extended the alphabet, not replaced it

# ✓ RIGHT option B: train a new LM from scratch using custom-med
# Embeddings and vocab are born together
```

See [06](../docs/theory/06-pretrained-model-trap.md) · Optional GPU homework: [07](../docs/theory/07-optional-lora-sft.md)
"""))
'''),

        # ── 14. Exercises ─────────────────────────────────────────────────────
        md("""
## Exercises

**1.** Find one probe in the 20-probe table where `custom-med` and `cl100k` tie (same piece count).
   Using [05](../docs/theory/05-why-custom-wins-in-healthcare.md), explain why domain training
   didn't help on that term.

**2.** The single-token-rate table shows `cl100k = 0%` and `custom-med = 10%` on 20 medical terms.
   In your own words: what does 0% mean for a language model trying to learn what `troponin` is?

**3.** Compare the fertility numbers on the 20 curated probes vs the 2000 held-out abstracts.
   Why is the held-out eval a fairer proof of "custom wins"?

**4.** `general-bpe` has the **same** 16k vocab size and the **same** algorithm as `custom-med`.
   It scores worse on medical held-out fertility.  What single variable explains that gap?

**5.** Write the board sentence from [06](../docs/theory/06-pretrained-model-trap.md) from memory.
   Then explain in one sentence why replacing Llama's tokenizer with `custom-med` and running
   LoRA would not give you a medical LLM.

**6.** Fill the diminishing-returns table from your run. Which jump saved the most tokens?
   Which saved the least?

**7.** Your corpus band says 16k–32k. Did the table agree? If 100k still wins on avg
   tokens/doc, why might you still refuse it?

**8.** Using `d_model=1024`, how many extra embedding params is 100k vs 32k?
   Is that worth a 1% token cut?

**9.** You picked a size for custom-med. Can you load that tokenizer onto
   `Qwen/Qwen3-0.6B`? Write the board sentence.

---
*Instructor answers: `INSTRUCTOR.md`*
"""),
    ]


def main() -> None:
    write("04_custom_vs_general.ipynb", custom_vs_general_notebook(), "custom vs general")


if __name__ == "__main__":
    main()
