# Instructor notes — medical custom vs general tokenizer

Keep this file closed for students who want an honest attempt.

## 90-minute block

Do **not** live-demo LoRA.

| Min | Block | Material |
|---:|---|---|
| 0–10 | Why tokens matter | [01](docs/theory/01-why-tokenization-matters.md) + [02](docs/theory/02-general-purpose-tokenizers.md) |
| 10–20 | Custom BPE + PubMedBERT | [03](docs/theory/03-custom-domain-tokenizers.md) |
| 20–30 | Metrics + fairness design | [04](docs/theory/04-metrics.md) + [05](docs/theory/05-why-custom-wins-in-healthcare.md). Ask students to predict `empagliflozin` piece counts for all 4 tokenizers. |
| 30–70 | Notebook | [`notebooks/04_custom_vs_general.ipynb`](notebooks/04_custom_vs_general.ipynb) — 4-way compare, then vocab-size experiment |
| 70–85 | Trap | [06](docs/theory/06-pretrained-model-trap.md) — predict-then-reveal |
| 85–90 | Homework pointer | [07](docs/theory/07-optional-lora-sft.md) — stock Qwen tokenizer only |

If short on time in the notebook block, run vocab-size **Steps 0, 3, 6 only** (predict, diminishing-returns table, pick). Skip the 512-vs-1024 demo. Lecture: [08](docs/theory/08-vocab-size-tradeoff.md).

**Board sentence:** Tokenizer IDs must match embedding rows. Custom BPE changes the alphabet.
A pretrained model's embeddings still speak the old alphabet.

## The golden result (what the lab proves)

**Fair eval on held-out PubMed abstracts (5000 docs, never seen during training):**

| tokenizer | fertility | vs cl100k |
|-----------|----------|-----------|
| `custom-med` 16k | **1.375** | −5.8% |
| `o200k_base` ~200k | 1.430 | −2.1% |
| `cl100k_base` ~100k | 1.460 | baseline |
| `general-bpe` 16k | 1.747 | +20% |

`custom-med` and `general-bpe` have identical algorithm and vocab size. The only difference
is training domain. That isolates domain as the cause — not vocab size.

**Do not** use `empagliflozin 10 mg daily` as the golden headline — `custom-med` ties cl100k
on that specific probe (10 vs 10) at 16k vocab. Good teaching examples are `acetylcholinesterase
inhibitor` (3 vs 7, −57%), `ST-elevation myocardial infarction` (6 vs 8), `hemoglobin A1c 8.2%`
(8 vs 10). The honest golden assert is **held-out fertility**, not a single probe.

**Single-token rate on 20 medical terms:** cl100k=0/20, o200k=0/20, custom-med=2/20
(`hemoglobin`, `lymphocyte`). Teaches: general tokenizers have no medical vocabulary.

**Cross-domain (expected, honest):** custom-med fertility on general English held-out = 1.575,
vs cl100k 1.173. Specialization costs out-of-domain performance. That is a pass, not a fail.

## Probes where custom-med wins clearly (good board examples)

| probe | cl100k | custom-med | saving | why |
|-------|--------|------------|--------|-----|
| `acetylcholinesterase inhibitor` | 7 | 3 | −57% | compound clinical noun, very frequent in PubMed |
| `hemoglobin A1c 8.2%` | 10 | 8 | −20% | HbA1c in every diabetes abstract |
| `ST-elevation myocardial infarction` | 8 | 6 | −25% | STEMI, extremely common in cardiology |
| `serum creatinine 1.4 mg/dL` | 11 | 9 | −18% | lab name + unit frequent in metabolic papers |
| `metformin 1000 mg twice daily` | 9 | 7 | −22% | drug + dose pattern common in abstracts |

## Probes where custom-med loses (honest explanation for students)

| probe | cl100k | custom-med | why |
|-------|--------|------------|-----|
| `vancomycin trough 18 μg/mL` | 9 | 13 | `μ` is 2 UTF-8 bytes; PubMed writes `mcg/mL` — byte pair never merged |
| `community-acquired pneumonia` | 4 | 8 | hyphenated phrase; `pneumonia` alone is common, compound is not |
| `COVID-19 mRNA vaccine` | 5 | 7 | term post-dates the wikitext baseline; cl100k trained on 2021+ web |

These are teaching moments: a 16k vocab must prioritise, and training-data normalisation matters.

## Exercise keys

1. Short common words (`pneumonia`, `chronic`) are already in cl100k. Long generics and lab phrases are the demo.
2. Quote fertility from held-out sets — not the 20-probe mean. Held-out is the fair bar.
3. **No, do not replace the tokenizer.** Repeat the board sentence verbatim.
4. `general-bpe` (same 16k) loses on medical held-out → domain, not vocab size, drives the win.
5. CLI output should match notebook (use `--no-qwen`).
6. Biggest jump is usually 16k→32k (or 16k→50k). Smallest is 64k→100k (~1% on the lesson table). **Do not** treat the 64k textbook example as this lab's answer — PubMed is a small corpus.
7. Band says 16k–32k. Table may still show 100k with the lowest avg tokens/doc. Refuse 100k: embedding table is 6.25× 16k, corpus cannot fill the budget, compression already flattened. Knee helper (2%) should land on 16k or 32k.
8. Extra params = `(100_000 − 32_000) × 1024 = 69,632,000`. Not worth a 1% token cut.
9. **No.** Board sentence verbatim. Vocab-size pick is for a tokenizer / from-scratch model, not a Qwen swap.

Do **not** let students leave thinking "bigger vocab always wins." That undoes the domain-vs-size lesson.

## Data files

| File | Use | Tracked? |
|---|---|---|
| `data/medical_corpus.txt` | Tiny authored set for fallback tokenizer | ✓ |
| `data/medical_probes.txt` | 20 illustration probes | ✓ |
| `data/medical_control.txt` | General English spot-check | ✓ |
| `data/pubmed_train.jsonl` | 45k PubMed train split | gitignored |
| `data/pubmed_heldout.jsonl` | 5k PubMed held-out (fair eval) | gitignored |
| `data/general_train.jsonl` | 45k wikitext train | gitignored |
| `data/general_heldout.jsonl` | 5k wikitext held-out | gitignored |
| `models/pretrained/medical_bpe_tiny/` | Fallback if training data absent | ✓ |
| `artifacts/medical-bpe-pubmed/` | Trained custom-med 16k | gitignored |
| `artifacts/medical-bpe-pubmed-{16,32,50,64,100}k/` | Vocab-size sweep | gitignored |
| `artifacts/general-bpe/` | Trained general-bpe 16k | gitignored |

## Before class (full setup, needs network, ~5 min)

```bash
uv sync

# Download and split PubMed
uv run python scripts/download_pubmed_sample.py --max-docs 50000
uv run python scripts/split_corpus.py

# Download wikitext (fairness control)
uv run python scripts/download_general_sample.py

# Train both tokenizers
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/pubmed_train.jsonl \
    --out artifacts/medical-bpe-pubmed/tokenizer.json
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/general_train.jsonl \
    --out artifacts/general-bpe/tokenizer.json

# Vocab-size sweep (reuses 16k; trains 32k/50k/64k/100k as needed)
uv run python scripts/sweep_vocab_size.py \
    --corpus data/pubmed_train.jsonl \
    --heldout data/pubmed_heldout.jsonl \
    --no-qwen

# Verify everything passes
uv run pytest tests/test_medical_compare.py tests/test_vocab_sweep.py -v
```

## Before class (no network — fallback mode)

```bash
uv sync
uv run pytest tests/test_medical_compare.py tests/test_vocab_sweep.py -v
uv run python scripts/compare_tokenizers.py --no-qwen
```

The notebook loads `models/pretrained/medical_bpe_tiny/` automatically. Students see chips
and the probe table. Held-out fertility cells print "not found" and skip gracefully.

## tiktoken

First call to `tiktoken.get_encoding("cl100k_base")` or `"o200k_base"` downloads small encoding
files (~1 MB each) and caches them. Run it once on instructor machine before class.
