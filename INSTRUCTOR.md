# Instructor notes — Tokenization Deep Dive

Keep this file closed for students who want an honest attempt. Use it for live demos,
TA office hours, and after-class review.

## Live site

- **URL:** https://sourangshupal.github.io/tokenization-explainer/
- **Deep links:** `?tab=bpe#algo-labs`, `?tab=wordpiece#algo-labs`, `?tab=sentencepiece&sp=unigram#algo-labs`
- **Theme:** Light toggle for projectors (cyan-on-zinc is harsh under some lamps).
- **Predict mode** defaults on. Reveal after students guess.
- **Schematics:** https://sourangshupal.github.io/tokenization-explainer/diagrams/ (project these; light paper + cyan)

## Suggested 90-minute script

| Min | Block | What to project |
|---:|---|---|
| 0–8 | Taxonomy + pipeline | `#taxonomy` + [01-pipeline](https://sourangshupal.github.io/tokenization-explainer/diagrams/01-pipeline.html) |
| 8–18 | Four-way lab | English → emoji → Bengali. Quiz: naive ≠ BPE. Optional [02-granularity](https://sourangshupal.github.io/tokenization-explainer/diagrams/02-granularity.html) |
| 18–28 | OOV trap | Vocab `the cat sat on the mat`, probe with `dog` / `cats` / a name. Predict first. |
| 28–50 | Algorithm labs | Sennrich toy + [03-bpe](https://sourangshupal.github.io/tokenization-explainer/diagrams/03-bpe-train.html) / [04-score](https://sourangshupal.github.io/tokenization-explainer/diagrams/04-bpe-vs-wordpiece.html). Probe `lowest`. |
| 50–55 | UTF-8 + marks | Inspector + [05-detokenize](https://sourangshupal.github.io/tokenization-explainer/diagrams/05-detokenize-marks.html) |
| 55–85 | Notebooks | Run `01_bpe` from-scratch + golden assert; skim HF; tiktoken if network ok. |
| 85–90 | Capstone | BERT / GPT-4o / T5 table from `01_bpe.ipynb`. |

Messy-paragraph demo: switch corpus preset to **Course corpus** and watch WordPiece’s
first merges get weird — teach “tiny corpus ≠ Wikipedia.”

## Two corpora (say this out loud)

| File | Use |
|---|---|
| `data/sennrich_toy.txt` | Website default + from-scratch BPE / WordPiece |
| `data/tiny_corpus.txt` | HuggingFace + SentencePiece production cells |

Same probe `lowest` → different merges across the two files. Expected.

## Golden encodings (Sennrich toy)

Website / from-scratch (after enough merges):

| Algo | Probe | Pieces |
|---|---|---|
| BPE | `lowest` | `low` + `est</w>` |
| WordPiece | `lowest` | `low` + `##est` |

Notebook asserts check these. Site compare pane should match on the Sennrich preset.

## Exercise keys (short)

### 01_bpe

1. Fewer merges → more character-ish pieces; more merges → longer pieces, shorter lists.
2. `</w>` marks word end so `st` in `star` ≠ final `st` in `widest`.
3. Without Whitespace pre-tokenizer, BPE would merge across spaces (`the▁cat`-style or glued words depending on model).
4. Non-Latin scripts usually cost more `tiktoken` pieces per grapheme.

### 02_wordpiece

1. Step-0 winners often differ: BPE max freq vs WP likelihood ratio.
2. `##` marks continuations → detokenize by stripping `##` and concatenating; `</w>` marks ends.
3. Shared stem depends on vocab; often `play` + `##ing` / `##ed`.
4. `xyzzy` → `[UNK]` when letters never entered the vocab.

### 03_sentencepiece

1. Space lives inside a piece as `▁` (U+2581), not as a separate whitespace token.
2. Unigram segmentations can jump more when vocab_size changes; BPE merges are more path-dependent.
3. Without `byte_fallback`, unknown scripts fall to `<unk>` walls.
4. Sampling = data augmentation for the *language model*; inference wants the single best path.

## Install failures

- `tiktoken`: first call downloads encodings — run once on network.
- `sentencepiece`: use `models/pretrained/sp_unigram_tiny.model` if the wheel fails.
- No Colab required; optional: upload the repo zip and `pip` is discouraged — prefer Codespaces / local `uv`.

## Regression check before class

```bash
uv sync
node tests/golden_algos.mjs
uv run python scripts/build_notebooks.py
uv run python -m http.server 8000 --directory site
```

Open `http://localhost:8000/?tab=wordpiece#algo-labs` — predict, reveal, confirm `low` + `##est`.
