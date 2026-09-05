# Medical custom vs general tokenizer lab

Train a byte-level BPE on PubMed abstracts. Compare it to `cl100k_base`, `o200k_base`, and a
same-size general-domain BPE control. Prove the win is from **domain**, not vocab size. Explain
why you must not swap that vocab onto a pretrained LLM.

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Do not use pip.

## What this lab proves

Four tokenizers — evaluated on **held-out text they never saw during training**:

| tokenizer | vocab | domain | medical fertility | general fertility |
|-----------|-------|--------|-------------------|-------------------|
| `custom-med` | 16k | PubMed | **1.375** ← wins | 1.575 (loses) |
| `o200k_base` | ~200k | general | 1.430 | **1.163** ← wins |
| `cl100k_base` | ~100k | general | 1.460 | 1.173 |
| `general-bpe` | 16k | wikitext | 1.747 (worst) | 1.208 |

`custom-med` and `general-bpe` have **identical algorithm and vocab size**. On medical held-out
text, `custom-med` is −5.8% vs cl100k while `general-bpe` is +20% worse. The only variable is
training domain. That is the proof.

Single-token rate on 20 medical terms: `cl100k` and `o200k` encode **0/20** as a single token.
`custom-med` encodes 2/20. Custom-med has built a medical vocabulary; general tokenizers have not.

Then a second experiment: same PubMed corpus, five vocab sizes (16k–100k). Measure **avg tokens
per document**. Pick the smallest size near the minimum. This corpus is small (<1B tokens) so
the recommended band is **16k–32k**, not the 50k "modern LLM default."

## Quick start (class, no network)

```bash
uv sync
uv run jupyter lab notebooks/04_custom_vs_general.ipynb
```

The notebook auto-loads the checked-in fallback tokenizer (`models/pretrained/medical_bpe_tiny/`)
if the trained artifacts are absent.

## Full pipeline (instructor, needs network)

```bash
# 1. Download 50k PubMed abstracts and split into train/held-out (gitignored)
uv run python scripts/download_pubmed_sample.py --max-docs 50000
uv run python scripts/split_corpus.py          # 45k train, 5k held-out

# 2. Download wikitext-103 for the fairness control (gitignored)
uv run python scripts/download_general_sample.py

# 3. Train both tokenizers
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/pubmed_train.jsonl \
    --out artifacts/medical-bpe-pubmed/tokenizer.json
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/general_train.jsonl \
    --out artifacts/general-bpe/tokenizer.json

# 4. Compare (CLI — all four tokenizers + held-out fertility)
uv run python scripts/compare_tokenizers.py --no-qwen \
    --tokenizer-json  artifacts/medical-bpe-pubmed/tokenizer.json \
    --general-bpe     artifacts/general-bpe/tokenizer.json \
    --heldout-medical data/pubmed_heldout.jsonl \
    --heldout-general data/general_heldout.jsonl

# 4b. Vocab-size experiment (same PubMed corpus: 16k / 32k / 50k / 64k / 100k)
uv run python scripts/sweep_vocab_size.py \
    --corpus data/pubmed_train.jsonl \
    --heldout data/pubmed_heldout.jsonl \
    --no-qwen

# 5. Run the notebook
uv run jupyter lab notebooks/04_custom_vs_general.ipynb

# 6. Run tests
uv run pytest tests/test_medical_compare.py tests/test_vocab_sweep.py -v
```

GPU LoRA homework is **not** this lab. See [`docs/theory/07-optional-lora-sft.md`](docs/theory/07-optional-lora-sft.md)
and `uv sync --extra sft`. Use the **stock** Qwen tokenizer.

## Layout

```
docs/theory/                         # 00–08 slide-ready markdown
  00-overview.md                     # lab design + 4-tokenizer table + run order
  01-why-tokenization-matters.md
  02-general-purpose-tokenizers.md   # cl100k, o200k, general-bpe explained
  03-custom-domain-tokenizers.md     # PubMed BPE + fairness control design
  04-metrics.md                      # fertility, single-token rate, held-out eval
  05-why-custom-wins-in-healthcare.md
  06-pretrained-model-trap.md        # required — board sentence
  07-optional-lora-sft.md
  08-vocab-size-tradeoff.md          # compression vs vocab size experiment

docs/diagrams/
  06-vocab-size-knee.md              # flattening curve + embedding cost

data/
  medical_corpus.txt                 # small authored set (tracked)
  medical_probes.txt                 # 20 curated illustration probes (tracked)
  medical_control.txt                # general English spot-check (tracked)
  pubmed_sample.jsonl                # 50k downloaded abstracts (gitignored)
  pubmed_train.jsonl                 # 45k train split (gitignored)
  pubmed_heldout.jsonl               # 5k held-out eval (gitignored)
  general_train.jsonl                # 45k wikitext train (gitignored)
  general_heldout.jsonl              # 5k wikitext held-out (gitignored)

notebooks/
  04_custom_vs_general.ipynb        # the lab (38 cells)
  lab_display.py                    # Rich display helpers (chips, charts, tables)

scripts/
  download_pubmed_sample.py         # stream PubMed → JSONL
  split_corpus.py                   # deterministic 45k/5k disjoint split
  download_general_sample.py        # stream wikitext-103 → JSONL
  train_medical_tokenizer.py        # byte-level BPE trainer (generic --corpus)
  wrap_medical_tokenizer.py         # tokenizer.json → PreTrainedTokenizerFast
  compare_tokenizers.py             # 4-tokenizer CLI with held-out eval
  sweep_vocab_size.py               # 16k–100k avg-tokens/doc experiment
  build_notebooks.py                # regenerates notebook 04

models/pretrained/medical_bpe_tiny/ # fallback if live train fails (tracked)

artifacts/                           # trained outputs (gitignored)
  medical-bpe-pubmed/               # custom-med 16k
  medical-bpe-pubmed-16k/ … -100k/  # vocab-size sweep
  general-bpe/                      # general-bpe 16k fairness control

tests/
  test_medical_compare.py           # structure + held-out fairness asserts
  test_vocab_sweep.py               # knee / band / tiny-corpus monotonicity
```

## Network

- `tiktoken` downloads `cl100k_base` and `o200k_base` encoding files on first call (small, cached).
- PubMed and wikitext downloads are instructor-only; students use the fallback.
- Qwen tokenizer is optional; `--no-qwen` skips it everywhere.

## License

[MIT](LICENSE)
