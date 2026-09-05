# General-purpose tokenizers

A general tokenizer was trained on a **web-scale mix**: crawl, code, books, Wikipedia. Medical
terms appear, but not often enough to earn their own merge. Long drug names, ICD strings, and lab
units get chopped.

## The three general baselines in this lab

**tiktoken `cl100k_base`.** Byte-level BPE used by the GPT-4 family. ~100k vocab. Shipped merge
tables. First call downloads encoding files. Does **not** train on `medical_corpus.txt`.

**tiktoken `o200k_base`.** Byte-level BPE used by GPT-4o. ~200k vocab. Stronger general baseline
— more merges means better compression even on medical text, but still a general corpus.

**`general-bpe` (fairness control).** A 16k byte-level BPE trained in this lab on
`data/general_train.jsonl` (wikitext-103). Same algorithm and same vocab size as `custom-med`.
This is the honest control: if `custom-med` beats `general-bpe` on medical text, the gap is
caused by **domain**, not vocab size or algorithm choice.

```python
import tiktoken
enc_cl = tiktoken.get_encoding("cl100k_base")
enc_o  = tiktoken.get_encoding("o200k_base")

enc_cl.encode("empagliflozin 10 mg daily")
# → many ids; decode each to see the fragments

enc_o.encode("empagliflozin 10 mg daily")
# → slightly fewer ids (200k vocab > 100k), still fragments
```

If network allows, Qwen (`Qwen/Qwen3-0.6B`) is loadable via `AutoTokenizer` and behaves like
another large general tokenizer. It is **optional** — tiktoken + `custom-med` + `general-bpe`
is enough to show the full pattern.

## What general tokenizers are good at

Everyday English, code, mixed web text. `data/medical_control.txt` and `data/general_heldout.jsonl`
are that world. A medical BPE is **not** required to beat them there. Domain fit, not universal
superiority.

Expected result on `general_heldout.jsonl`:

| tokenizer | fertility |
|-----------|----------|
| `o200k` | lowest |
| `cl100k` | slightly higher |
| `general-bpe` (16k) | slightly higher — small vocab costs something on general text |
| `custom-med` (16k) | highest — paid for medical specialization |

## What they are bad at (healthcare)

- Brand and generic drug names (`empagliflozin`, `oseltamivir`, `acetylcholinesterase`)
- Gene symbols glued to clinical words (`BRCA1 pathogenic variant`)
- Billing codes (`ICD-10-CM E11.65`)
- Lab units (`mg/dL`, `μg/mL`, `pg/mL`)
- Greek / mixed Unicode (`β-lactam`, `Naïve CD4+`)

Single-token rate on 20 medical terms: **cl100k 0/20, o200k 0/20** — not a single medical term
exists as one token. `custom-med` gets 2/20. That gap is the vocabulary story.

Count pieces. That is the whole demo.
