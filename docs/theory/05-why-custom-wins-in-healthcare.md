# Why custom wins in healthcare

> ⚖️ **Fairness design diagram:** [4-tokenizer 2×2 design → docs/diagrams/02-fair-comparison.md](../diagrams/02-fair-comparison.md)

General BPE was not paid to memorize drug names. Your BPE was — and the data proves it.

## BPE is frequency-driven

Byte-pair encoding is a greedy merge algorithm. At each step it finds the most frequent pair
of adjacent tokens in the training corpus and merges them into one symbol. After enough merges:

- A word seen thousands of times (`empagliflozin` in PubMed abstracts) accumulates enough
  adjacent-pair frequency to survive as a **single merge** — one token.
- The same word is **vanishingly rare** on the general web. `cl100k` or `o200k` has never seen
  enough `empagliflozin` occurrences to merge it, so it falls back to byte-level fragments:
  `Ġemp`, `ag`, `lif`, `lo`, `zin` — five unrelated-looking pieces until the model learns to
  glue them.

## What the numbers mean

| Metric | custom-med | cl100k | o200k |
|--------|-----------|--------|-------|
| `empagliflozin 10 mg daily` | ~6–7 pieces | 10 pieces | 9 pieces |
| `acetylcholinesterase inhibitor` | 3–4 pieces | 7 pieces | 6 pieces |
| Held-out PubMed fertility | **lower** | higher | higher |
| Held-out general English | higher | **lower** | **lower** |

The last two rows are the teaching payoff. Custom-med wins in-domain and loses out-of-domain —
exactly what specialization should do.

## The fairness control: general-bpe (16k on wikitext)

The naive objection: "maybe custom wins because it has a smaller vocabulary, not because it
knows medicine." The `general-bpe` control (same 16k vocab, same byte-level BPE algorithm,
trained on wikitext-103) destroys that argument:

- `general-bpe` and `custom-med` have **identical algorithm and vocab size**.
- `general-bpe` beats `cl100k`/`o200k` on general English — it spends its 16k merges where
  general text is dense.
- `custom-med` beats `general-bpe` on medical text — it spends its 16k merges where medical
  text is dense (`empagliflozin`, `acetylcholinesterase`, `creatinine`, ...).

The only variable left is **training domain**. That is the proof.

## Worked probes

| Probe | Why the general tokenizer over-splits |
|-------|--------------------------------------|
| `empagliflozin 10 mg daily` | Long generic drug name; rare on the open web |
| `acetylcholinesterase inhibitor` | Compound chemistry + clinical noun |
| `BRCA1 pathogenic variant` | Gene symbol + standardized phrase |
| `ICD-10-CM E11.65` | Dots, hyphens, digits, billing syntax |
| `serum creatinine 1.4 mg/dL` | Lab name + unit |
| `β-lactam allergy` | Greek letter + hyphenated drug class |
| `levothyroxine 75 μg daily` | Microgram sign; uncommon outside clinical notes |
| `vancomycin trough 18 μg/mL` | Drug + PK jargon + unit |
| `ST-elevation myocardial infarction` | Hyphenated syndrome name |

Prediction exercise: before running any cell, guess the piece counts. General tokenizers
usually shatter the drug names. Custom should not.

## What "win" means in this lab

Fewer tokens for the same clinical string → shorter sequences → more of a clinical note fits
in the model's context window → one (or two) embedding rows per concept instead of a jigsaw
of fragments.

`empagliflozin` as one token ID is a drug. As five fragment IDs it is noise until the model
accumulates many training examples to learn the composition.

## Honest limit

Fertility is **not** downstream F1. PubMedBERT's gains on NER/RE came from **pretraining with**
the biomedical vocab, not just swapping the tokenizer. This lab proves the segmentation story.

See [06 — the pretrained-model trap](06-pretrained-model-trap.md) for exactly why you cannot
drop `artifacts/medical-bpe-hf/` onto Qwen and LoRA it.

## Not a win on every probe

Some probes are already common enough in general corpora that `cl100k` handles them well
(`pneumonia`, `COVID-19`, `chronic kidney disease`). The teaching examples are the ones that
**explode** in general tokenizers. Those are the ones to show first.
