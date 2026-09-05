# 🏥 Medical Tokenizer Lab — Custom vs General

> **Train a domain tokenizer from scratch. Beat GPT-4 on medical text. Prove it's domain — not vocab size — that wins.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-tokenizers-FFD21E?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-22C55E?style=flat-square)

---

## 🎯 What this lab proves

We compare **4 tokenizers** on the **same held-out medical text** — text none of them trained on:

| Tokenizer | Vocab | Trained on | Medical fertility | General fertility |
|-----------|-------|-----------|:-----------------:|:-----------------:|
| 🏥 `custom-med` | 16k | PubMed abstracts | **1.375 ✅ wins** | 1.575 ❌ |
| 🤖 `o200k_base` | ~200k | General web (GPT-4o) | 1.430 | **1.163 ✅** |
| 🤖 `cl100k_base` | ~100k | General web (GPT-4) | 1.460 | 1.173 |
| 📰 `general-bpe` | 16k | wikitext-103 | 1.747 ❌ worst | 1.208 |

> 💡 **The key insight:** `custom-med` and `general-bpe` have **identical algorithm and identical vocab size (16k)**.
> On medical held-out text, `custom-med` beats `general-bpe` by **0.372** — the only variable is training domain.
> That's the proof. Domain wins, not size.

**Single-token rate on 20 medical terms:**
- `cl100k` and `o200k` → **0 / 20** medical terms as a single token
- `custom-med` → **2 / 20** — it has built a medical vocabulary; general tokenizers have not

---

## 🔬 The fertility flip

`custom-med` **loses** on general text (fertility 1.575 vs 1.163 for o200k). This is expected — and is *proof of genuine specialisation*. A tokenizer that shows no flip didn't actually specialise.

```
                 Medical text     General text
  custom-med       1.375 ✅          1.575 ❌
  general-bpe      1.747 ❌          1.208 ✅
                      ↑                 ↑
               custom wins         general wins
               → domain is the variable, not vocab size
```

---

## ⚡ Quick start (offline, no network needed)

```bash
git clone https://github.com/sourangshupal/tokenization-explainer
cd tokenization-explainer
uv sync
uv run jupyter lab notebooks/04_custom_vs_general.ipynb
```

> The notebook auto-loads the checked-in fallback tokenizer (`models/pretrained/medical_bpe_tiny/`) if trained artifacts are absent. Everything works offline.

---

## 🚀 Full pipeline (needs internet)

### Step 1 — Download corpora

```bash
# 50k PubMed abstracts from HuggingFace Hub (streams Parquet, ~180 MB)
uv run python scripts/download_pubmed_sample.py --max-docs 50000

# Split into disjoint train / held-out (deterministic, seed 42)
uv run python scripts/split_corpus.py
# → data/pubmed_train.jsonl    (45k abstracts)
# → data/pubmed_heldout.jsonl  ( 5k abstracts — never used for training)

# wikitext-103 for the fairness control (downloads + splits in one step)
uv run python scripts/download_general_sample.py
# → data/general_train.jsonl    (45k paragraphs)
# → data/general_heldout.jsonl  ( 5k paragraphs)
```

### Step 2 — Train both tokenizers

```bash
# Medical tokenizer — 16k vocab on PubMed
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/pubmed_train.jsonl \
    --out artifacts/medical-bpe-pubmed/tokenizer.json

# General-BPE fairness control — 16k vocab on wikitext
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/general_train.jsonl \
    --out artifacts/general-bpe/tokenizer.json
```

### Step 3 — Compare all 4 tokenizers

```bash
uv run python scripts/compare_tokenizers.py --no-qwen \
    --tokenizer-json  artifacts/medical-bpe-pubmed/tokenizer.json \
    --general-bpe     artifacts/general-bpe/tokenizer.json \
    --heldout-medical data/pubmed_heldout.jsonl \
    --heldout-general data/general_heldout.jsonl
```

### Step 4 — Vocab-size sweep (16k → 100k)

```bash
uv run python scripts/sweep_vocab_size.py \
    --corpus  data/pubmed_train.jsonl \
    --heldout data/pubmed_heldout.jsonl \
    --no-qwen
```

Finds the **knee** of the fertility curve — the smallest vocab size near the minimum. For this PubMed corpus (< 1B tokens) the recommended band is **16k–32k**, not the 50k "modern LLM default."

### Step 5 — Run the notebook and tests

```bash
uv run jupyter lab notebooks/04_custom_vs_general.ipynb
uv run pytest tests/test_medical_compare.py tests/test_vocab_sweep.py -v
```

---

## 📁 Repository layout

```
📦 tokenization-explainer
│
├── 📓 notebooks/
│   ├── 04_custom_vs_general.ipynb   # The lab — 4-way compare + vocab-size experiment
│   └── lab_display.py               # Rich display helpers (chips, tables, charts)
│
├── 📚 docs/
│   ├── theory/                      # 9 slide-ready markdown docs (00–08)
│   │   ├── 00-overview.md           # Lab design, 4-tokenizer table, run order
│   │   ├── 01-why-tokenization-matters.md
│   │   ├── 02-general-purpose-tokenizers.md
│   │   ├── 03-custom-domain-tokenizers.md
│   │   ├── 04-metrics.md            # Fertility, single-token rate, held-out eval
│   │   ├── 05-why-custom-wins-in-healthcare.md
│   │   ├── 06-pretrained-model-trap.md  ⚠️ Read this before any LLM fine-tuning
│   │   ├── 07-optional-lora-sft.md
│   │   └── 08-vocab-size-tradeoff.md
│   └── diagrams/                    # 6 Mermaid diagrams (render on GitHub / JupyterLab)
│       ├── 01-bpe-algorithm.md      # BPE training loop
│       ├── 02-fair-comparison.md    # 2×2 fairness design
│       ├── 03-lab-pipeline.md       # Full pipeline flowchart
│       ├── 04-pretrained-trap.md    # Wrong vs right paths
│       ├── 05-fertility-concept.md  # What fertility means with real numbers
│       └── 06-vocab-size-knee.md    # Diminishing returns + embedding cost
│
├── 🐍 scripts/
│   ├── download_pubmed_sample.py    # Stream PubMed → JSONL (HuggingFace Hub)
│   ├── split_corpus.py              # Deterministic 45k / 5k disjoint split
│   ├── download_general_sample.py   # Stream wikitext-103 → JSONL
│   ├── train_medical_tokenizer.py   # Byte-level BPE trainer
│   ├── wrap_medical_tokenizer.py    # tokenizer.json → PreTrainedTokenizerFast
│   ├── compare_tokenizers.py        # 4-tokenizer CLI with held-out fertility
│   └── sweep_vocab_size.py          # 16k–100k avg-tokens/doc experiment
│
├── 🧪 tests/
│   ├── test_medical_compare.py      # Held-out fairness assertions
│   └── test_vocab_sweep.py          # Knee / band / monotonicity checks
│
├── 🗄️ data/
│   ├── medical_corpus.txt           # Small authored set (tracked)
│   ├── medical_probes.txt           # 20 curated illustration probes (tracked)
│   ├── medical_control.txt          # General English spot-check (tracked)
│   └── *.jsonl                      # Downloaded corpora (gitignored — large)
│
├── 🤗 models/pretrained/
│   └── medical_bpe_tiny/            # Fallback tokenizer if training data absent (tracked)
│
└── 🏺 artifacts/                    # Trained outputs (gitignored)
    ├── medical-bpe-pubmed/          # custom-med 16k
    ├── medical-bpe-pubmed-{16k..100k}/  # vocab-size sweep
    └── general-bpe/                 # fairness control 16k
```

---

## 📖 Theory docs — reading order

| # | Document | When to read |
|---|----------|-------------|
| 01 | [Why tokenization matters](docs/theory/01-why-tokenization-matters.md) | Start here |
| 02 | [General-purpose tokenizers](docs/theory/02-general-purpose-tokenizers.md) | Before the notebook |
| 03 | [Custom domain tokenizers](docs/theory/03-custom-domain-tokenizers.md) | Before training |
| 04 | [Metrics](docs/theory/04-metrics.md) | Before measuring |
| 05 | [Why custom wins in healthcare](docs/theory/05-why-custom-wins-in-healthcare.md) | After seeing results |
| 06 | ⚠️ [The pretrained-model trap](docs/theory/06-pretrained-model-trap.md) | **Required** |
| 07 | [Optional: LoRA + SFT](docs/theory/07-optional-lora-sft.md) | Optional homework |
| 08 | [Vocab-size tradeoff](docs/theory/08-vocab-size-tradeoff.md) | Sweep experiment |

---

## 🗺️ Visual diagrams

All diagrams are Mermaid markdown — render natively on **GitHub** and **JupyterLab**:

| Diagram | What it shows |
|---------|--------------|
| [🔬 BPE algorithm](docs/diagrams/01-bpe-algorithm.md) | How merge rules are learned from frequency |
| [⚖️ Fair comparison](docs/diagrams/02-fair-comparison.md) | Why 4 tokenizers? The isolation argument |
| [🗺️ Lab pipeline](docs/diagrams/03-lab-pipeline.md) | Every script and what it produces |
| [⚠️ Pretrained trap](docs/diagrams/04-pretrained-trap.md) | Wrong path vs right path |
| [🌱 Fertility concept](docs/diagrams/05-fertility-concept.md) | What fertility means with real numbers |
| [📉 Vocab-size knee](docs/diagrams/06-vocab-size-knee.md) | Diminishing returns + embedding cost |

---

## 🛠️ Tech stack

| Layer | Library |
|-------|---------|
| Tokenizer training | `huggingface/tokenizers` (Rust, Python bindings) |
| General baselines | `tiktoken` (`cl100k_base`, `o200k_base`) |
| HuggingFace wrapping | `transformers.PreTrainedTokenizerFast` |
| Corpus download | `datasets` (streaming Parquet) |
| Display | `rich` (tables, charts, progress bars) |
| Package manager | `uv` — **never use pip directly** |
| Python | 3.12 |

---

## ⚠️ What this lab does NOT do

- Replace Qwen's `tokenizer.json` with `custom-med` and call it fine-tuning — **this breaks the model** (see [doc 06](docs/theory/06-pretrained-model-trap.md))
- Train on MIMIC or any credentialed EHR data
- Publish the tokenizer to HuggingFace Hub (it's class-scale, 16k)

GPU LoRA homework is separate — see [`docs/theory/07-optional-lora-sft.md`](docs/theory/07-optional-lora-sft.md) and `uv sync --extra sft`. Use the **stock** Qwen tokenizer, not custom-med.

---

## 📄 License

[MIT](LICENSE) — free to use, adapt, and teach with.
