# 🧑‍💻 Student Guide — Build & Compare a Domain Tokenizer

**Session duration:** 4 hours  
**What you'll build:** A medical BPE tokenizer trained on PubMed abstracts, then compare it fairly against 3 other tokenizers to prove domain training — not vocab size — drives the performance gap.  
**What you already know:** BPE algorithm, merge rules, vocab size, byte-level encoding.

---

## ⚡ Quick orientation

We compare **4 tokenizers** today:

| Name | Vocab | Trained on | Role |
|------|-------|-----------|------|
| `cl100k` | ~100k | General web (GPT-4) | Real-world baseline |
| `o200k` | ~200k | General web (GPT-4o) | Stronger baseline |
| `custom-med` | 16k | PubMed abstracts | **You train this today** |
| `general-bpe` | 16k | wikitext-103 | Fairness control — same size as custom-med |

> The `general-bpe` row is the key. If `custom-med` beats it on medical text, the win comes from **domain**, not vocab size.

---

## 📂 Phase 0 — Setup (5 min)

Clone the repo and install dependencies:

```bash
git clone <repo-url>
cd tokenization-explainer
uv sync
```

Confirm everything installed:

```bash
uv run python -c "import tokenizers, tiktoken, transformers, datasets; print('all good')"
# Expected: all good
```

---

## 📥 Phase 1 — Download the corpora (10–15 min, needs internet)

### 1a · Medical corpus — 50k PubMed abstracts

```bash
uv run python scripts/download_pubmed_sample.py --max-docs 50000
# Streams from HuggingFace Hub (Parquet) → no dataset-script issues
# → data/pubmed_sample.jsonl  (~50k rows, ~180 MB)
# Takes ~5–10 min depending on connection
```

### 1b · Split into train and held-out sets

```bash
uv run python scripts/split_corpus.py
# → data/pubmed_train.jsonl    (45 000 abstracts — used for training)
# → data/pubmed_heldout.jsonl  ( 5 000 abstracts — NEVER used for training)
```

> ❓ **Why held-out?** We measure tokenizer quality on text it has never seen. If we tested on training data, results would be inflated. The held-out set is the fair judge.

### 1c · General corpus — 50k wikitext-103 paragraphs

```bash
uv run python scripts/download_general_sample.py
# Downloads wikitext-103 from Salesforce/wikitext (HuggingFace Hub)
# Automatically splits into train + held-out
# → data/general_train.jsonl    (45 000 paragraphs)
# → data/general_heldout.jsonl  ( 5 000 paragraphs)
# Takes ~3–5 min
```

### 1d · Verify all 4 files

```bash
wc -l data/pubmed_train.jsonl data/pubmed_heldout.jsonl \
       data/general_train.jsonl data/general_heldout.jsonl
```

Expected output:
```
 45000 data/pubmed_train.jsonl
  5000 data/pubmed_heldout.jsonl
 45000 data/general_train.jsonl
  5000 data/general_heldout.jsonl
```

---

## 🔍 Phase 2 — Inspect the data (5 min)

Look at one PubMed abstract:

```bash
head -1 data/pubmed_train.jsonl | python -m json.tool
```

Look at one wikitext paragraph:

```bash
head -1 data/general_train.jsonl | python -m json.tool
```

> ❓ **Think:** What types of words will appear most often in PubMed that never appear in wikitext?  
> Examples: `empagliflozin`, `acetylcholinesterase`, `SGLT2`, `myocardial`  
> BPE will find these pairs and merge them. wikitext BPE never will.

---

## 🏋️ Phase 3 — Train the tokenizers (10–15 min)

### 3a · Train `custom-med` on PubMed

```bash
uv run python scripts/train_medical_tokenizer.py \
  --input data/pubmed_train.jsonl \
  --vocab-size 16000 \
  --output artifacts/medical-bpe-pubmed/
# → artifacts/medical-bpe-pubmed/tokenizer.json
```

While it runs, watch the progress bar — each step = one more merge rule added to the vocab.

After it finishes, peek inside:

```bash
python -c "
import json
t = json.load(open('artifacts/medical-bpe-pubmed/tokenizer.json'))
print('First 20 merges:')
for m in t['model']['merges'][:20]:
    print(' ', m)
"
```

> ❓ **Notice:** Early merges are high-frequency medical character pairs (`ch`, `ol`, `in`, `ester`). These co-occur thousands of times per abstract.

### 3b · Train `general-bpe` on wikitext

```bash
uv run python scripts/train_medical_tokenizer.py \
  --input data/general_train.jsonl \
  --vocab-size 16000 \
  --output artifacts/general-bpe/
# → artifacts/general-bpe/tokenizer.json
```

Peek at its early merges:

```bash
python -c "
import json
t = json.load(open('artifacts/general-bpe/tokenizer.json'))
print('First 20 merges:')
for m in t['model']['merges'][:20]:
    print(' ', m)
"
```

> ❓ **Compare:** Early merges are `th`, `in`, `er`, `on` — common English, zero medical morphemes. Same algorithm, completely different vocabulary.

### 3c · Wrap `custom-med` for HuggingFace

```bash
uv run python scripts/wrap_medical_tokenizer.py
# → models/pretrained/medical_bpe_tiny/tokenizer.json
# → models/pretrained/medical_bpe_tiny/tokenizer_config.json
```

This makes `custom-med` usable with any HuggingFace pipeline — same API as GPT-4's tokenizer.

---

## 🖥️ Phase 4 — Terminal comparison (5 min)

Quick look at all 4 tokenizers side by side:

```bash
uv run python scripts/compare_tokenizers.py \
  --heldout-medical data/pubmed_heldout.jsonl \
  --heldout-general data/general_heldout.jsonl \
  --general-bpe artifacts/general-bpe/tokenizer.json
```

You'll see a fertility table. **Don't interpret the numbers yet** — do the notebook first, then come back to explain what you see.

---

## 📓 Phase 5 — Notebook lab (45 min)

```bash
jupyter lab notebooks/04_custom_vs_general.ipynb
```

Run every cell in order. For each cell, answer the question below before moving on.

| Cell | Question to answer in your own words |
|------|--------------------------------------|
| 1 — Imports | Did everything load? If not, which library is missing? |
| 2 — Load 4 tokenizers | Which has the largest vocabulary? Which is smallest? |
| 3 — Single sentence comparison | Count the pieces for `acetylcholinesterase` in each tokenizer. Write down the 4 numbers. |
| 4 — Vocab membership | Which medical terms appear as a **single token** only in `custom-med`? |
| 5 — Fertility on held-out | Which tokenizer wins on medical text? Does the result surprise you? |
| 6 — 2×2 chart | `custom-med` **loses** on general text. Is this a bug or a feature? Explain. |
| 7 — Exercises | Write full answers (see below) |

### ✍️ Written exercises (answer in notebook markdown cells)

**Exercise 1:**  
`custom-med` beats `cl100k` on medical text. `cl100k` has 6× the vocabulary size. Explain in one sentence why `custom-med` still wins.

**Exercise 2:**  
`general-bpe` has the exact same vocab size (16k) and the exact same algorithm as `custom-med`, but scores worse on medical text. What does this prove about the cause of `custom-med`'s win?

**Exercise 3:**  
Would training `custom-med` on 90k abstracts instead of 45k make fertility better? Why/why not?

**Exercise 4 (stretch):**  
What would happen to the fertility numbers if we used `--vocab-size 32000` for `custom-med`? Would it beat `cl100k` by a larger margin? Where does this improvement come from — and where does it stop?

---

## 📊 Phase 6 — The fairness numbers (class discussion)

After the notebook, the instructor will draw this on the board. Make sure you understand each row:

```
Tokenizer       vocab   medical fertility   general fertility
─────────────────────────────────────────────────────────────
custom-med       16k       1.381 ✅ wins      1.812 ❌ worst
general-bpe      16k       1.759 ❌           1.503 ✅
cl100k          ~100k      1.470             1.434
o200k           ~200k      1.439             1.401
```

Key question: **Why does `custom-med` lose badly on general text?**  
Answer: It spent all 16k merge budget on medical morphemes. There's no room left for common English pairs. This *proves* it genuinely specialised — a tokenizer that didn't specialise would show no flip.

---

## ⚠️ Phase 7 — The pretrained-model trap (15 min)

Open the diagram:

```bash
# In VS Code or any markdown viewer:
# docs/diagrams/04-pretrained-trap.md
```

Or read the theory doc:

```bash
cat docs/theory/06-pretrained-model-trap.md
```

**The question:** You want to LoRA fine-tune LLaMA-3 on medical text. You swap in `custom-med` as the tokenizer. Will it work?

> **Answer:** No. Token IDs in `custom-med` no longer match the embedding rows LLaMA was trained with. Row 4521 in LLaMA's embedding table was trained to represent a specific token — now your new tokenizer sends `empagliflozin` there. LoRA cannot fix this mismatch. The model will produce garbage or never converge.

**The two valid paths:**
1. Keep LLaMA's tokenizer → use `add_tokens()` for new medical terms → `resize_token_embeddings()` → LoRA
2. Train a new LM from scratch using `custom-med` as the tokenizer from day 1

---

## ✅ Phase 8 — Run the test suite (5 min)

```bash
uv run pytest tests/test_medical_compare.py -v
```

Each green test = one of the lab's claims is numerically verified on your machine.  
All 12 should pass if all 4 data files exist. You'll see which assertions correspond to which claims.

---

## 📚 Reading map

Work through these in order — each takes 5–10 min:

| When | File | What it covers |
|------|------|---------------|
| Before Phase 3 | `docs/theory/03-custom-domain-tokenizers.md` | How BPE training works |
| Before Phase 5 | `docs/theory/04-metrics.md` | Fertility, single-token rate, fairness design |
| After Phase 6 | `docs/theory/05-why-custom-wins-in-healthcare.md` | Why domain frequency is the mechanism |
| Phase 7 | `docs/theory/06-pretrained-model-trap.md` | Embedding mismatch explained |
| Self-study | `docs/theory/01-why-tokenization-matters.md` | The bigger picture |
| Self-study | `docs/theory/02-general-purpose-tokenizers.md` | How cl100k and o200k work |
| Optional | `docs/theory/07-optional-lora-sft.md` | LoRA + SFT if you want to go further |

---

## 🗂️ Visual diagrams (open in any markdown viewer)

| Diagram | What it shows |
|---------|--------------|
| `docs/diagrams/01-bpe-algorithm.md` | BPE training loop step by step |
| `docs/diagrams/02-fair-comparison.md` | Why we need 4 tokenizers, the isolation argument |
| `docs/diagrams/03-lab-pipeline.md` | Every script and what it produces |
| `docs/diagrams/04-pretrained-trap.md` | Wrong path vs right path when building a medical LLM |
| `docs/diagrams/05-fertility-concept.md` | What fertility means with real numbers |

---

## 🔑 Three things to remember when you leave

1. **BPE is frequency-driven** — train on the wrong corpus, the wrong pairs get merged
2. **Domain beats vocab size** — `general-bpe` (16k) proves it: same size, bigger fertility gap
3. **Tokenizer and model weights are inseparable** — born together, cannot be swapped without retraining
