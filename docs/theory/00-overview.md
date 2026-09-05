# Lab overview — custom vs general tokenizers

> 🗺️ **Visual pipeline diagram:** [Full lab pipeline → docs/diagrams/03-lab-pipeline.md](../diagrams/03-lab-pipeline.md)

**Goal.** Same medical strings. Four tokenizers. Count pieces, measure fertility on held-out text,
show the mechanism. Prove that domain training — not vocab size — drives the difference.

**Custom is better at segmentation on in-domain text.** It is not a plug-in for Llama or Qwen.

## The four-way comparison

| name | vocab | training domain | role |
|------|-------|----------------|------|
| `cl100k_base` | ~100k | general (GPT-4 web) | production general baseline |
| `o200k_base` | ~200k | general (GPT-4o) | stronger general baseline |
| `custom-med` | 16k | **medical (PubMed abstracts)** | the domain tokenizer |
| `general-bpe` | 16k | general (wikitext-103) | **fairness control** |

`general-bpe` is the critical column: **same algorithm, same 16k vocab, different corpus.**
If `custom-med` beats `general-bpe` on medical held-out text, domain is the cause — not vocab size.

## What you will do

1. Read docs `01`–`05` (short).
2. Load or train tokenizers. Corpus for `custom-med`: `data/pubmed_train.jsonl` (45k PubMed
   abstracts, disjoint from the 5k held-out eval set). Fallback: `models/pretrained/medical_bpe_tiny/`.
3. Encode the same probes side-by-side with all four tokenizers. Guess first.
4. Evaluate fertility on **held-out** sets:
   - `data/pubmed_heldout.jsonl` — 5k medical abstracts (never seen during training)
   - `data/general_heldout.jsonl` — 5k wikitext paragraphs (cross-domain check)
5. Read the 2×2 result. `custom-med` should win medical, lose general. Both are the expected result.
6. Run the vocab-size experiment ([08](08-vocab-size-tradeoff.md)): same corpus, five
   budgets, avg tokens/doc, pick the smallest size near the minimum.
7. Read `06` so you do not swap this tokenizer onto a pretrained LLM and call it fine-tuning.

## What you will not do in this lab

- Replace Qwen's `tokenizer.json` and LoRA the attention layers.
- Train on MIMIC or any credentialed EHR dump.
- Publish this tokenizer to the Hub. It is class-scale (16k).

## Session map (~90 min)

| Block | Minutes | Material |
|---|---|---|
| Theory | 25 | `01`–`05` + board examples |
| Notebook 04 | 50 | chips, single-token rate, held-out fertility, vocab-size experiment, 2×2 chart |
| Trap | 15 | `06` predict-then-reveal |
| Optional homework | — | `07` (stock tokenizer LoRA, needs GPU) |

## Board sentence

> Tokenizer IDs must match embedding rows. Custom BPE changes the alphabet. A pretrained model's
> embeddings still speak the old alphabet.

## Files

| Path | Role |
|---|---|
| `data/pubmed_train.jsonl` | Train `custom-med` (45k PubMed abstracts, gitignored) |
| `data/pubmed_heldout.jsonl` | Fair eval for medical fertility (5k, gitignored) |
| `data/general_train.jsonl` | Train `general-bpe` fairness control (45k wikitext, gitignored) |
| `data/general_heldout.jsonl` | Fair eval for general English fertility (5k, gitignored) |
| `data/medical_corpus.txt` | Small authored set for the tiny fallback tokenizer (tracked) |
| `data/medical_probes.txt` | 20 curated domain strings — illustration examples |
| `data/medical_control.txt` | General English spot-check |
| `artifacts/medical-bpe-pubmed/` | Trained `custom-med` 16k tokenizer.json (gitignored) |
| `artifacts/medical-bpe-pubmed-16k/` … `-100k/` | Vocab-size sweep artifacts (gitignored) |
| `artifacts/general-bpe/` | Trained `general-bpe` 16k tokenizer.json (gitignored) |
| `models/pretrained/medical_bpe_tiny/` | Fallback if training data absent (tracked) |
| `notebooks/04_custom_vs_general.ipynb` | The lab |
| `docs/theory/06-pretrained-model-trap.md` | The misconception this lab exists to kill |
| `docs/theory/08-vocab-size-tradeoff.md` | Compression vs vocab-size experiment |

## Scripts (instructor run order)

```bash
# 1. Download and split PubMed data (network required; gitignored)
uv run python scripts/download_pubmed_sample.py --max-docs 50000
uv run python scripts/split_corpus.py

# 2. Download wikitext for the fairness control
uv run python scripts/download_general_sample.py

# 3. Train both tokenizers
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/pubmed_train.jsonl --out artifacts/medical-bpe-pubmed/tokenizer.json
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/general_train.jsonl --out artifacts/general-bpe/tokenizer.json

# 4. Run comparison (CLI)
uv run python scripts/compare_tokenizers.py --no-qwen \
    --tokenizer-json artifacts/medical-bpe-pubmed/tokenizer.json \
    --general-bpe    artifacts/general-bpe/tokenizer.json \
    --heldout-medical data/pubmed_heldout.jsonl \
    --heldout-general data/general_heldout.jsonl

# 4b. Vocab-size sweep (same PubMed corpus, five budgets)
uv run python scripts/sweep_vocab_size.py \
    --corpus data/pubmed_train.jsonl \
    --heldout data/pubmed_heldout.jsonl \
    --no-qwen

# 5. Run lab notebook
uv run jupyter lab notebooks/04_custom_vs_general.ipynb

# 6. Run tests
uv run pytest tests/test_medical_compare.py tests/test_vocab_sweep.py -v
```
