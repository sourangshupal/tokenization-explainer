# Tokenization Deep Dive

[![Live demo](https://img.shields.io/badge/demo-GitHub%20Pages-22d3ee?style=flat-square&logo=github&logoColor=white)](https://sourangshupal.github.io/tokenization-explainer/)
[![Pages](https://img.shields.io/github/actions/workflow/status/sourangshupal/tokenization-explainer/pages.yml?label=pages&style=flat-square)](https://github.com/sourangshupal/tokenization-explainer/actions/workflows/pages.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/packaging-uv-DE5FE9?style=flat-square)](https://docs.astral.sh/uv/)
[![tokenizers](https://img.shields.io/badge/tokenizers-0.23.1-FFD21E?style=flat-square)](https://github.com/huggingface/tokenizers)
[![sentencepiece](https://img.shields.io/badge/sentencepiece-0.2.2-4285F4?style=flat-square)](https://github.com/google/sentencepiece)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)](LICENSE)

Interactive course for AI engineers: a static **token lab** in the browser, then three Jupyter notebooks that train BPE, WordPiece, and SentencePiece from scratch before using current PyPI libraries via **uv**.

**[Open the live site](https://sourangshupal.github.io/tokenization-explainer/)** — no install. All playgrounds run in your browser.

<p align="center">
  <img src="assets/readme/hero.png" alt="Token lab homepage: numbered syllabus and taxonomy" width="100%">
</p>

## What you get

| Surface | Runs where | What students do |
|---|---|---|
| [Live site](https://sourangshupal.github.io/tokenization-explainer/) | GitHub Pages (static HTML/JS) | Taxonomy, four-way splitter, OOV trap, UTF-8 inspector, in-browser BPE / WordPiece / SentencePiece |
| Jupyter notebooks | Local Python 3.12 + uv | Same algorithms from scratch, then HuggingFace `tokenizers` 0.23.1, `sentencepiece` 0.2.2, `tiktoken` 0.14.0 |

The website does **not** need Python. Notebooks do **not** run on GitHub Pages.

## Syllabus

1. Taxonomy of tokenization
2. Word / subword / character / byte level (live labs)
3. Byte Pair Encoding — site lab + `notebooks/01_bpe.ipynb`
4. WordPiece — site lab + `notebooks/02_wordpiece.ipynb`
5. SentencePiece — site lab + `notebooks/03_sentencepiece.ipynb`

## Live playgrounds

Four-way split of the same string (word, naive subword, character, UTF-8 bytes):

<p align="center">
  <img src="assets/readme/four-way.png" alt="Four-way live splitter playground" width="100%">
</p>

Same corpus, three trainers. Switch **BPE**, **WordPiece**, and **SentencePiece** without leaving the page:

<p align="center">
  <img src="assets/readme/algo-labs.png" alt="Algorithm labs: BPE, WordPiece, and SentencePiece tabs" width="100%">
</p>

| Lab | What it teaches |
|---|---|
| Four-way splitter | How token *count* explodes as units get smaller |
| OOV trap | Word-level `[UNK]` when a type was never in the vocab |
| UTF-8 inspector | Graphemes vs code points vs bytes (emoji, Indic, Japanese) |
| Length bars | Vocab size vs sequence length |
| BPE tab | Max pair frequency, `</w>` end-of-word mark |
| WordPiece tab | Score `freq(ab)/(freq(a)·freq(b))`, greedy `##` encode |
| SentencePiece tab | Unigram Viterbi with `▁`, or BPE with no whitespace split |

Default teaching corpus is the Sennrich toy set (`low` / `lower` / `newest` / `widest`). WordPiece on that set encodes `lowest` as `low` + `##est`.

## Quickstart (notebooks)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. Do not use pip.

```bash
git clone https://github.com/sourangshupal/tokenization-explainer.git
cd tokenization-explainer
uv sync
uv run jupyter lab notebooks
```

Work through the notebooks in order:

| Notebook | Topic |
|---|---|
| [`notebooks/01_bpe.ipynb`](notebooks/01_bpe.ipynb) | Frequency BPE from scratch, HuggingFace `BpeTrainer`, then `tiktoken` (`cl100k_base`, `o200k_base`) |
| [`notebooks/02_wordpiece.ipynb`](notebooks/02_wordpiece.ipynb) | Likelihood WordPiece from scratch, `##` continuation, `WordPieceTrainer` |
| [`notebooks/03_sentencepiece.ipynb`](notebooks/03_sentencepiece.ipynb) | Unigram Viterbi intuition, then `sentencepiece` 0.2.2 (`return_type=`, not deprecated `out_type`) |

Each notebook trains only on [`data/tiny_corpus.txt`](data/tiny_corpus.txt). Written models go to `artifacts/` (gitignored).

Local site (optional; the Pages URL is enough for class):

```bash
uv run python -m http.server 8000 --directory site
```

Then open `http://localhost:8000`. You can also open `site/index.html` directly.

## Project layout

```
tokenization-explainer/
├── site/                      # GitHub Pages root (static)
│   ├── index.html
│   ├── css/main.css
│   └── js/                    # playgrounds.js → algorithms.js → main.js
├── notebooks/                 # 01 BPE, 02 WordPiece, 03 SentencePiece
├── data/tiny_corpus.txt       # shared toy corpus
├── scripts/build_notebooks.py # regenerates the three notebooks
├── assets/readme/             # README screenshots
├── pyproject.toml             # uv dependencies
└── .github/workflows/pages.yml
```

## Libraries (pinned via uv)

Resolved from PyPI; see `uv.lock`.

| Package | Role |
|---|---|
| `tokenizers` 0.23.1 | HuggingFace BPE and WordPiece trainers |
| `sentencepiece` 0.2.2 | Google SentencePiece (pybind11 API) |
| `tiktoken` 0.14.0 | Production byte-level BPE (OpenAI models) |
| `jupyterlab` 4.6.3, `ipykernel`, `ipywidgets` | Notebooks and live widgets |

SentencePiece 0.2.2: use `encode(..., return_type=str)` and `SentencePieceProcessor.from_file(...)`. Do not use deprecated `out_type` or `EncodeAsImmutableProto`.

## GitHub Pages

The site is static. GitHub Actions (`.github/workflows/pages.yml`) uploads the `site/` folder on every push to `main`. No Jekyll (`.nojekyll`).

Notebooks, `uv`, and trained `.model` files are **not** part of Pages.

## License

[MIT](LICENSE)
