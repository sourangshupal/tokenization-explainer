#!/usr/bin/env python3
"""Generate the three course notebooks. Run: uv run python scripts/build_notebooks.py"""

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


SETUP = r'''
from __future__ import annotations

from collections import Counter
from pathlib import Path

from IPython.display import display, Markdown
import ipywidgets as widgets

def find_root() -> Path:
    here = Path.cwd()
    for candidate in [here, here.parent]:
        if (candidate / "data" / "tiny_corpus.txt").exists():
            return candidate
    raise FileNotFoundError("Run the notebook from the repo root or the notebooks/ folder.")

ROOT = find_root()
CORPUS = ROOT / "data" / "tiny_corpus.txt"          # richer file for HuggingFace / SentencePiece
SENNRICH = ROOT / "data" / "sennrich_toy.txt"      # classic four-word set for from-scratch labs
ARTIFACTS = ROOT / "artifacts"
PRETRAINED = ROOT / "models" / "pretrained"
ARTIFACTS.mkdir(exist_ok=True)
print(f"course corpus: {CORPUS}")
print(f"Sennrich toy:  {SENNRICH}")
print()
print("Mermaid diagrams: GitHub and JupyterLab often render ```mermaid fences.")
print("If you see raw fences, read the flowchart as text — the algorithms still run.")
'''.strip()

MERMAID_NOTE = """
> **Diagram tip.** If your Jupyter UI shows a raw ` ```mermaid ` fence instead of a
> picture, that is a renderer gap (common in plain Classic Notebook). Read the
> flowchart as text, or open the notebook on GitHub / VS Code.
>
> Full-page schematics (Token Lab):
> [diagrams gallery](https://sourangshupal.github.io/tokenization-explainer/diagrams/)
> · local `../site/diagrams/index.html`
"""

DIAGRAM_PIPELINE = """
> Schematic: [Tokenizer pipeline](https://sourangshupal.github.io/tokenization-explainer/diagrams/01-pipeline.html)
> · local [`../site/diagrams/01-pipeline.html`](../site/diagrams/01-pipeline.html)
"""

DIAGRAM_BPE = """
> Schematic: [BPE train loop](https://sourangshupal.github.io/tokenization-explainer/diagrams/03-bpe-train.html)
> · local [`../site/diagrams/03-bpe-train.html`](../site/diagrams/03-bpe-train.html)
"""

DIAGRAM_WP = """
> Schematic: [BPE vs WordPiece](https://sourangshupal.github.io/tokenization-explainer/diagrams/04-bpe-vs-wordpiece.html)
> · [Detokenize marks](https://sourangshupal.github.io/tokenization-explainer/diagrams/05-detokenize-marks.html)
> · local [`../site/diagrams/`](../site/diagrams/)
"""

DIAGRAM_SP = """
> Schematic: [Detokenize marks (▁)](https://sourangshupal.github.io/tokenization-explainer/diagrams/05-detokenize-marks.html)
> · local [`../site/diagrams/05-detokenize-marks.html`](../site/diagrams/05-detokenize-marks.html)
"""


FADED_BPE = r'''
# Faded example: you fill merge #1 after seeing merge #0 annotated.
from collections import Counter

def _pair_counts(splits: dict[str, list[str]], counts: Counter[str]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for word, freq in counts.items():
        symbols = splits[word]
        for left, right in zip(symbols, symbols[1:]):
            pairs[(left, right)] += freq
    return pairs

def _apply_merge(splits: dict[str, list[str]], pair: tuple[str, str]) -> dict[str, list[str]]:
    a, b = pair
    glued = a + b
    updated: dict[str, list[str]] = {}
    for word, symbols in splits.items():
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(glued)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        updated[word] = out
    return updated

toy = Counter({"low": 5, "lower": 2, "newest": 3, "widest": 2})
splits0 = {w: list(w) + ["</w>"] for w in toy}
pairs0 = _pair_counts(splits0, toy)
best0, freq0 = pairs0.most_common(1)[0]
print("Annotated merge 0 (already done for you):")
print(f"  best pair {best0!r} with freq {freq0}")
print("  → glue into", repr(best0[0] + best0[1]))
splits1 = _apply_merge(splits0, best0)

# YOUR TURN: pick the best pair after merge 0
pairs1 = _pair_counts(splits1, toy)
# student_pair = pairs1.most_common(1)[0][0]   # uncomment and run
student_pair = pairs1.most_common(1)[0][0]
print("Your merge 1 pair:", student_pair)

golden_pair = pairs1.most_common(1)[0][0]
assert student_pair == golden_pair, f"expected {golden_pair}, got {student_pair}"
print("assert ok — merge 1 matches the golden pair")
'''.strip()


CAPSTONE = """
## Capstone (after all three notebooks)

Given a model card, name the tokenizer family and one consequence for length:

| Card clue | Algorithm |
|---|---|
| BERT / `##` pieces / `[CLS]` | WordPiece |
| GPT-4o / `cl100k` / `o200k` / tiktoken | byte-level BPE |
| T5 / `spiece.model` / `▁` | SentencePiece (usually Unigram) |

Then encode the same prompt with `tiktoken` (`o200k_base`) and estimate how many
tokens a 4k-character English email costs vs the same email in Bengali.
"""



def bpe_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
# Byte Pair Encoding (BPE)

**Audience:** complete beginners. You do not need prior NLP.

BPE is the algorithm behind GPT-style tokenizers (and many Llama tokenizers). It starts from
tiny pieces (characters, or even bytes) and **glues the most common adjacent pair** over and
over until the vocabulary is big enough.

After this notebook you will be able to:

1. Explain why word-level vocabularies break on new words.
2. Train a tiny BPE model by hand and watch each merge.
3. Encode a new word with the merge list you learned.
4. Train the same idea with HuggingFace `tokenizers` (current API).
5. Inspect a production BPE vocabulary with `tiktoken`.
"""
        ),
        md(
            """
## Learning path

```mermaid
flowchart LR
  problem[OOV problem] --> scratch[From-scratch BPE]
  scratch --> encode[Encode with merges]
  encode --> hf[HuggingFace BpeTrainer]
  hf --> tik[tiktoken production BPE]
```
"""
            + MERMAID_NOTE
            + DIAGRAM_BPE
        ),
        md(
            """
## Why BPE exists

A **word-level** tokenizer stores every whole word. The moment a student types `unhappiness`
and that string never appeared in training, the model sees `[UNK]`. Spelling information is
gone.

BPE's bet: keep **frequent** words intact, and build rare words from **reusable pieces**.
`low` + `est` can form `lowest` even if `lowest` itself was rare.

The original NLP paper is Sennrich, Haddow, and Birch (2016). Neural nets had already used a
similar compression trick (Gage, 1994). GPT-2 later ran BPE on **UTF-8 bytes** so nothing is
unknown.
"""
        ),
        md(
            """
## The algorithm (picture)

```mermaid
flowchart TD
  corpus[Corpus of words plus counts] --> split[Split each word into characters]
  split --> endmark["Append an end-of-word mark </w>"]
  endmark --> count[Count every adjacent pair]
  count --> pick[Pick the pair with the highest count]
  pick --> merge[Glue that pair into one new symbol]
  merge --> vocab[Add the new symbol to the vocabulary]
  vocab --> enough{Reached N merges?}
  enough -->|no| count
  enough -->|yes| done[Save merge list and vocab]
```

`</w>` matters. Without it, BPE cannot tell the `st` inside `star` from the `st` at the end
of `widest`. The end mark is a boundary.
"""
            + MERMAID_NOTE
        ),
        md("## Setup — paths and corpora"),
        code(SETUP),
        md(
            """
## Two corpora (do not mix them up)

| File | Role |
|---|---|
| `data/sennrich_toy.txt` | Classic four-word set. **From-scratch BPE below uses this.** Same default as the website lab. |
| `data/tiny_corpus.txt` | Richer English paragraphs. **HuggingFace / SentencePiece cells use this.** |

If you train on the course corpus and compare to the website Sennrich slider, merges will
differ. That is expected — not a bug.
"""
        ),
        code(
            """
print("Sennrich toy:\\n")
print(SENNRICH.read_text(encoding="utf-8"))
print("---")
print("Course corpus first 12 lines:\\n")
print("\\n".join(CORPUS.read_text(encoding="utf-8").splitlines()[:12]))
"""
        ),
        md(
            """
## From scratch: count words, then characters

We only keep alphabetic tokens so the first merges stay readable. Real BPE also sees
punctuation and digits. Production GPT BPE sees raw bytes.
"""
        ),
        code(
            """
def load_word_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        for raw in line.lower().split():
            word = "".join(ch for ch in raw if ch.isalpha())
            if word:
                counts[word] += 1
    return counts


word_counts = load_word_counts(SENNRICH)
print(f"{len(word_counts)} word types, {sum(word_counts.values())} tokens")
print("Counts:", dict(word_counts))
print("Course-corpus types (HF later):", len(load_word_counts(CORPUS)))
"""
        ),
        code(
            """
def initial_splits(counts: Counter[str]) -> dict[str, list[str]]:
    return {word: list(word) + ["</w>"] for word in counts}


splits = initial_splits(word_counts)
for word in ["low", "lower", "newest", "widest"]:
    if word in splits:
        print(f"{word:8}  {splits[word]}   x{word_counts[word]}")
"""
        ),
        md(
            """
## Count pairs and merge the winner

A **pair** is two neighbouring symbols inside a word. If `low` appears 5 times as
`['l','o','w','</w>']`, the pair `('l','o')` gets +5, not +1. Frequency is corpus count, not
type count.
"""
        ),
        code(
            """
def pair_counts(splits: dict[str, list[str]], counts: Counter[str]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for word, freq in counts.items():
        symbols = splits[word]
        for left, right in zip(symbols, symbols[1:]):
            pairs[(left, right)] += freq
    return pairs


def apply_merge(splits: dict[str, list[str]], pair: tuple[str, str]) -> dict[str, list[str]]:
    a, b = pair
    glued = a + b
    updated: dict[str, list[str]] = {}
    for word, symbols in splits.items():
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(glued)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        updated[word] = out
    return updated


stats = pair_counts(splits, word_counts)
print("Top 10 pairs before any merge:")
for pair, freq in stats.most_common(10):
    print(f"  {pair!s:30} {freq}")
"""
        ),
        md(
            """
## Train: repeat the merge N times

Each round we record `(left, right) → left+right`. That ordered list **is** the tokenizer.
Encoding a new word later means replaying these merges in the same order.
"""
        ),
        code(
            """
def train_bpe(
    counts: Counter[str],
    num_merges: int,
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    splits = initial_splits(counts)
    merges: list[tuple[str, str]] = []
    for step in range(1, num_merges + 1):
        stats = pair_counts(splits, counts)
        if not stats:
            break
        pair, freq = stats.most_common(1)[0]
        if freq < 2:
            print(f"stop at step {step}: best pair only appears {freq} time(s)")
            break
        merges.append(pair)
        splits = apply_merge(splits, pair)
        a, b = pair
        print(f"{step:02d}. merge {a!r} + {b!r}  →  {a + b!r}   (freq {freq})")
    return merges, splits


merges, trained_splits = train_bpe(word_counts, num_merges=25)
print(f"\\nlearned {len(merges)} merges")
print("\\nHow the classic words look after training:")
for word in ["low", "lower", "newest", "widest", "lowest"]:
    if word in trained_splits:
        print(f"  {word:8} {trained_splits[word]}")
"""
        ),
        md(
            """
## Encode a new word

Training never saw every possible word. Encoding still works: split into characters, then
apply **the same merges in the same order**. If `e` + `s` was merged during training, it
will merge here too.
"""
        ),
        code(
            """
def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(word.lower()) + ["</w>"]
    for pair in merges:
        symbols = apply_merge({"w": symbols}, pair)["w"]
    return symbols


def encode_text(text: str, merges: list[tuple[str, str]]) -> list[str]:
    pieces: list[str] = []
    for raw in text.split():
        word = "".join(ch for ch in raw.lower() if ch.isalpha())
        if word:
            pieces.extend(encode_word(word, merges))
        else:
            pieces.append(raw)
    return pieces


for sample in ["lowest", "newer", "tokenization", "unhappiness"]:
    print(f"{sample:15} → {encode_word(sample, merges)}")

# Golden check — Sennrich toy (~15 merges before pairs die out)
lowest_pieces = encode_word("lowest", merges)
print("\\ngolden assert on lowest:", lowest_pieces)
assert lowest_pieces == ["low", "est</w>"], lowest_pieces
print("assert ok — lowest → low + est</w>")
"""
        ),
        md(
            """
## Faded example — fill merge #1

Below, merge 0 is annotated for you. Confirm merge 1 matches the golden pair (the cell
asserts). This is the scaffolding step before you re-run `train_bpe` with other budgets.
"""
        ),
        code(FADED_BPE),
        md(
            """
## Interactive playground

Type a phrase. The encoder uses **your** merge list from the cell above. Re-run training
with a different `num_merges` and this widget will still use whatever `merges` is in memory.
"""
        ),
        code(
            """
box = widgets.Text(
    value="lowest newest unhappiness",
    description="Text:",
    layout=widgets.Layout(width="90%"),
    style={"description_width": "50px"},
)
out = widgets.Output()


def _run(_change=None) -> None:
    with out:
        out.clear_output()
        pieces = encode_text(box.value, merges)
        print("pieces:", pieces)
        print("count: ", len(pieces))


box.observe(_run, names="value")
_run()
display(box, out)
"""
        ),
        md(
            """
## Production library: HuggingFace `tokenizers`

The Rust library `tokenizers` (we installed **0.23.x** via uv) trains BPE at production
speed. The ideas are the same: a model, a pre-tokenizer, a trainer, then `encode`.

Current API — do not use old `BertTokenizer` constructors from `transformers` here. We stay
on the `tokenizers` package.
"""
        ),
        code(
            """
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

hf_bpe = Tokenizer(BPE(unk_token="[UNK]"))
hf_bpe.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
    vocab_size=120,
    min_frequency=1,
    show_progress=False,
)
hf_bpe.train([str(CORPUS)], trainer)

encoded = hf_bpe.encode("lower newest unhappiness tokenization")
print("tokens:", encoded.tokens)
print("ids:   ", encoded.ids)
print("vocab size:", hf_bpe.get_vocab_size())
"""
        ),
        code(
            """
hf_path = ARTIFACTS / "hf_bpe.json"
hf_bpe.save(str(hf_path))
reloaded = Tokenizer.from_file(str(hf_path))
print("reloaded:", reloaded.encode("widest lower").tokens)
"""
        ),
        md(
            """
## Production BPE: OpenAI `tiktoken`

You do not train `tiktoken`. OpenAI already ran byte-level BPE on a huge corpus and shipped
the merge table. `cl100k_base` is the GPT-4 / GPT-3.5 family. `o200k_base` is the newer
GPT-4o family. Notice how one English word is often **one** token, while a rare or
non-English string becomes several.

This is still BPE. The difference is scale and the byte-level base vocabulary.

**Campus / offline note.** The first `tiktoken.get_encoding(...)` call **downloads** the
encoding files. Do that once on a network (home wifi), then re-run offline. If the download
is blocked in class, skip this cell and use the HuggingFace BPE section above.
"""
        ),
        code(
            """
import tiktoken

for name in ["cl100k_base", "o200k_base"]:
    enc = tiktoken.get_encoding(name)
    samples = [
        "tokenization",
        "unhappiness",
        "lowest newest",
        "বাংলা",
        "👋",
    ]
    print(f"\\n=== {name}  (vocab ~ {enc.n_vocab}) ===")
    for s in samples:
        ids = enc.encode(s)
        pieces = [enc.decode([i]) for i in ids]
        print(f"  {s!r:20} ids={ids}  pieces={pieces!r}")
"""
        ),
        md(
            """
## Where you will see BPE

| System | Flavour |
|---|---|
| GPT-2 / GPT-3 / GPT-4 / GPT-4o | byte-level BPE (`tiktoken`) |
| Many Llama / Mistral tokenizers | BPE (SentencePiece or HuggingFace) |
| The mini lab on the website | character BPE with `</w>` |

BPE never asks “is this a linguistically nice morpheme?” It only asks “did this pair occur a
lot?” WordPiece (next notebook) changes that scoring rule.
"""
        ),
        md(
            """
## Exercises

1. Re-run `train_bpe` with `num_merges=5` and `num_merges=40`. Encode `lowest`. What changed?
2. Why does `</w>` exist? Try a thought experiment: merge `st` in both `star` and `widest`.
3. HuggingFace BPE used a `Whitespace` pre-tokenizer. What would break if you skipped it?
4. Using `tiktoken`, encode your name in English and in another script. Compare token counts.

Write answers in a new cell below. Instructor golden notes live in `INSTRUCTOR.md` at the
repo root — keep that file closed during the lab if you want an honest attempt.
"""
        ),
        md(CAPSTONE),
    ]


def wordpiece_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
# WordPiece

**Audience:** complete beginners who finished the BPE notebook.

WordPiece looks like BPE on the surface: start from characters, glue pairs, grow a
vocabulary. The **score** is different. BPE picks the pair that occurs most often.
WordPiece picks the pair that is most “surprising” given the parts — a likelihood ratio.

Google built WordPiece for speech, then BERT made it famous. Continuation pieces inside a
word are marked with `##` so the decoder knows they do not start a new word.
"""
        ),
        md(
            """
## Learning path

```mermaid
flowchart LR
  bpe[BPE: max frequency] --> wp[WordPiece: max score]
  wp --> marks["## continuation marks"]
  marks --> greedy[Greedy longest-match encode]
  greedy --> hf[HuggingFace WordPieceTrainer]
```
"""
            + MERMAID_NOTE
            + DIAGRAM_WP
        ),
        md(
            """
## BPE vs WordPiece in one picture

```mermaid
flowchart TD
  subgraph bpe [BPE]
    b1[Count pair AB] --> b2["Pick max count(AB)"]
  end
  subgraph wp [WordPiece]
    w1["count(AB) / count(A) / count(B)"] --> w2[Pick max score]
  end
  bpe --> same[Both glue AB into a new vocab piece]
  wp --> same
```

If `t` and `h` are already extremely common, BPE still loves merging `th` because it appears
a lot. WordPiece asks: *does `th` occur more than you'd expect from `t` and `h` alone?*
That is why BERT pieces often look a bit more “word-like”.
"""
            + MERMAID_NOTE
        ),
        md("## Setup"),
        code(SETUP),
        md(
            """
## Toy WordPiece trainer

We reuse the BPE bookkeeping (splits + pair counts) and only change the **ranking**.

Score of pair `(a, b)`:

\\[
\\mathrm{score}(a,b) = \\frac{\\mathrm{freq}(a,b)}{\\mathrm{freq}(a)\\,\\mathrm{freq}(b)}
\\]

High score: `a` and `b` stick together more than chance. Low score: they just happen to
sit next to each other because both are common.

**Corpus alignment.** From-scratch WordPiece uses the same classic counts as the website
Sennrich lab (`low×5`, `lower×2`, `newest×3`, `widest×2`). HuggingFace below trains on
`data/tiny_corpus.txt` — piece lists will differ. That is intentional.
"""
        ),
        code(
            """
def load_word_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        for raw in line.lower().split():
            word = "".join(ch for ch in raw if ch.isalpha())
            if word:
                counts[word] += 1
    return counts


def initial_splits(counts: Counter[str]) -> dict[str, list[str]]:
    return {word: list(word) for word in counts}


def symbol_freqs(splits: dict[str, list[str]], counts: Counter[str]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for word, n in counts.items():
        for sym in splits[word]:
            freq[sym] += n
    return freq


def pair_counts(splits: dict[str, list[str]], counts: Counter[str]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for word, n in counts.items():
        symbols = splits[word]
        for left, right in zip(symbols, symbols[1:]):
            pairs[(left, right)] += n
    return pairs


def apply_merge(splits: dict[str, list[str]], pair: tuple[str, str]) -> dict[str, list[str]]:
    a, b = pair
    glued = a + b
    updated: dict[str, list[str]] = {}
    for word, symbols in splits.items():
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(glued)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        updated[word] = out
    return updated


word_counts = Counter({"low": 5, "lower": 2, "newest": 3, "widest": 2})
print("classic toy counts:", dict(word_counts))
print("full-file types (used later by HuggingFace):", len(load_word_counts(CORPUS)))
"""
        ),
        code(
            """
def best_wordpiece_pair(
    splits: dict[str, list[str]],
    counts: Counter[str],
    min_freq: int = 2,
) -> tuple[tuple[str, str], float, int] | None:
    # Rank pairs by likelihood score, ignoring hapax pairs.
    # On a tiny corpus, raw argmax(score) prefers rare letters like q+u
    # because freq(q)*freq(u) is tiny. Real WordPiece uses a count floor
    # (and a huge corpus). We keep the floor so later merges still happen.
    pairs = pair_counts(splits, counts)
    singles = symbol_freqs(splits, counts)
    ranked: list[tuple[tuple[str, str], float, int]] = []
    for pair, freq in pairs.items():
        if freq < min_freq:
            continue
        a, b = pair
        score = freq / (singles[a] * singles[b])
        ranked.append((pair, score, freq))
    if not ranked:
        return None
    ranked.sort(key=lambda row: row[1], reverse=True)
    return ranked[0]


def train_wordpiece(counts: Counter[str], num_merges: int) -> list[str]:
    splits = initial_splits(counts)
    vocab: set[str] = set()
    for symbols in splits.values():
        vocab.update(symbols)

    for step in range(1, num_merges + 1):
        ranked = best_wordpiece_pair(splits, counts)
        if ranked is None:
            print(f"stop at step {step}: no pair left with frequency >= 2")
            break
        pair, score, freq = ranked
        a, b = pair
        glued = a + b
        vocab.add(glued)
        splits = apply_merge(splits, pair)
        print(f"{step:02d}. {a!r} + {b!r} → {glued!r}   score={score:.4f}  pair_freq={freq}")
    return sorted(vocab, key=lambda s: (len(s), s))


wp_vocab = train_wordpiece(word_counts, num_merges=25)
print(f"\\nvocab size {len(wp_vocab)}")
print("longest pieces:", sorted(wp_vocab, key=len, reverse=True)[:12])
"""
        ),
        md(
            """
## Encoding with `##`

BERT-style WordPiece does **not** replay merges. It uses **greedy longest-match** on each
word:

1. Look at the whole word. If it is in the vocab, emit it.
2. If not, find the longest prefix that is in the vocab.
3. The remainder must be found with a `##` prefix (`est` → `##est`).
4. If a leftover character is missing, emit `[UNK]` for the whole word (BERT's original
   behaviour) or skip — we mark `[UNK]` so you can see the failure.

```mermaid
flowchart TD
  word[Input word] --> whole{Whole word in vocab?}
  whole -->|yes| emit[Emit word]
  whole -->|no| prefix[Longest prefix in vocab]
  prefix --> rest[Remainder]
  rest --> hash["Look up ##remainder"]
  hash --> more{More leftover?}
  more -->|yes| prefix
  more -->|no| done[Done]
```
"""
        ),
        code(
            """
def wordpiece_tokenize_word(word: str, vocab: set[str]) -> list[str]:
    # Greedy longest-match. Non-initial pieces are printed with ##.
    word = word.lower()
    pieces: list[str] = []
    start = 0
    while start < len(word):
        end = len(word)
        found: str | None = None
        while end > start:
            piece = word[start:end]
            if piece in vocab:
                found = piece
                break
            end -= 1
        if found is None:
            return ["[UNK]"]
        pieces.append(found if start == 0 else f"##{found}")
        start += len(found)
    return pieces


LOOKUP = set(wp_vocab)

for sample in ["low", "lower", "newest", "tokenization", "unhappiness", "xyzzy"]:
    print(f"{sample:15} → {wordpiece_tokenize_word(sample, LOOKUP)}")

lowest = wordpiece_tokenize_word("lowest", LOOKUP)
print("\\ngolden assert on lowest:", lowest)
assert lowest == ["low", "##est"], lowest
print("assert ok — lowest → low + ##est (matches the website lab)")
"""
        ),
        md("## Interactive playground"),
        code(
            """
box = widgets.Text(
    value="lower newest unhappiness",
    description="Text:",
    layout=widgets.Layout(width="90%"),
    style={"description_width": "50px"},
)
out = widgets.Output()


def encode_sentence(text: str) -> list[str]:
    pieces: list[str] = []
    for raw in text.split():
        word = "".join(ch for ch in raw.lower() if ch.isalpha())
        if word:
            pieces.extend(wordpiece_tokenize_word(word, LOOKUP))
    return pieces


def _run(_change=None) -> None:
    with out:
        out.clear_output()
        pieces = encode_sentence(box.value)
        print("pieces:", pieces)
        print("count: ", len(pieces))


box.observe(_run, names="value")
_run()
display(box, out)
"""
        ),
        md(
            """
## Side-by-side with BPE on one word

Same toy corpus, two scoring rules. They often agree on this tiny file. Differences show up
on real Wikipedia-scale data. The `##` mark is the part you will always notice in BERT.
"""
        ),
        code(
            """
def bpe_style_splits(counts: Counter[str], num_merges: int) -> dict[str, list[str]]:
    splits = {w: list(w) + ["</w>"] for w in counts}

    def pairs(sp):
        c: Counter[tuple[str, str]] = Counter()
        for word, n in counts.items():
            s = sp[word]
            for a, b in zip(s, s[1:]):
                c[(a, b)] += n
        return c

    def merge(sp, pair):
        a, b = pair
        glued = a + b
        out = {}
        for word, symbols in sp.items():
            row, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    row.append(glued)
                    i += 2
                else:
                    row.append(symbols[i])
                    i += 1
            out[word] = row
        return out

    for _ in range(num_merges):
        stats = pairs(splits)
        if not stats:
            break
        splits = merge(splits, stats.most_common(1)[0][0])
    return splits


bpe_splits = bpe_style_splits(word_counts, 20)
word = "lower"
print("WordPiece greedy:", wordpiece_tokenize_word(word, LOOKUP))
print("BPE after 20 merges (with </w>):", bpe_splits.get(word, "not in corpus as a type"))
print("WordPiece on 'lowest':", wordpiece_tokenize_word("lowest", LOOKUP))
print("BPE on corpus type 'lowest':", bpe_splits.get("lowest", "lowest was not a training type"))
"""
        ),
        md(
            """
## Production library: HuggingFace `WordPieceTrainer`

`tokenizers` 0.23 trains WordPiece and adds `##` for you. Pre-tokenization is still
whitespace — BERT also uses a punctuation splitter in the full pipeline. We keep whitespace
so the lesson stays aligned with BPE.
"""
        ),
        code(
            """
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

hf_wp = Tokenizer(WordPiece(unk_token="[UNK]"))
hf_wp.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    vocab_size=120,
    min_frequency=1,
    show_progress=False,
)
hf_wp.train([str(CORPUS)], trainer)

for sample in ["lower newest", "unhappiness", "tokenization", "xyzzy"]:
    enc = hf_wp.encode(sample)
    print(f"{sample:20} {enc.tokens}")

hf_wp.save(str(ARTIFACTS / "hf_wordpiece.json"))
print("vocab size", hf_wp.get_vocab_size())
"""
        ),
        md(
            """
## Where you will see WordPiece

- **BERT**, DistilBERT, Electra — `##` pieces, `[CLS]` / `[SEP]` / `[MASK]`
- **Original WordPiece** at Google — speech + multilingual search

If you see `##ing` or `##ization` in a model card, you are looking at WordPiece (or a
BERT-compatible clone), not GPT BPE.
"""
        ),
        md(
            """
## Exercises

1. In the from-scratch trainer, print the **top 5 BPE pairs** and the **top 5 WordPiece
   scores** at step 1 (before any merge). Do they pick the same winner?
2. Why does BERT use `##` instead of `</w>`? Which one makes detokenization easier?
3. Encode `playing` and `played` with the HuggingFace WordPiece model. Did they share a stem?
4. What should happen if you encode a character the vocab never saw? Check with `xyzzy`.

Golden discussion notes: `INSTRUCTOR.md`. Capstone table is at the end of `01_bpe.ipynb`.
"""
        ),
    ]


def sentencepiece_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
# SentencePiece

**Audience:** beginners who know BPE exists.

SentencePiece (Kudo & Richardson) is a tokenizer that treats the **raw Unicode string** as
the training unit. It does **not** have to split on spaces first. Spaces become a normal
character, drawn as `▁` (U+2581). That is why T5 and many multilingual models can handle
Chinese, Japanese, Thai, and English with one recipe.

Two inner models ship in the same library:

- **Unigram LM** (default) — start with a huge candidate vocab, then drop pieces that hurt
  likelihood the least.
- **BPE mode** — the same merge algorithm, but on the raw string (with `▁`), not on
  whitespace words.

We will build Unigram **intuition** (Viterbi segmentation) by hand, then train both modes
with `sentencepiece` **0.2.2** (pybind11 API).
"""
        ),
        md(
            """
## Learning path

```mermaid
flowchart LR
  raw[Raw text with spaces] --> mark["Spaces become ▁"]
  mark --> uni[Unigram: best segmentation]
  mark --> bpe[Optional BPE mode]
  uni --> lib[sentencepiece 0.2.2]
  bpe --> lib
```
"""
            + MERMAID_NOTE
            + DIAGRAM_SP
        ),
        md(
            """
## Why skip whitespace pre-tokenization?

English looks space-separated. Japanese and Chinese do not. If your first step is
`text.split()`, you have already baked in an English assumption.

SentencePiece's answer: the space is just another character. `"the cat"` is learned as
`▁the ▁cat` or `▁the ▁c at`, depending on the model. Decoding: every `▁` becomes a space.

```mermaid
flowchart TD
  s["the cat sat"] --> n["▁the ▁cat ▁sat"]
  n --> pieces["▁the / ▁cat / ▁sat"]
  pieces --> decode["the cat sat"]
```
"""
            + MERMAID_NOTE
        ),
        md("## Setup"),
        code(SETUP),
        md(
            """
## Unigram intuition (no EM, just Viterbi)

A full Unigram trainer is an EM loop: guess piece probabilities, segment the corpus,
update probabilities, drop useless pieces. That is too much machinery for a first lesson.

The part you must feel: **given a piece vocabulary with scores, pick the segmentation
with the highest total score.** That search is Viterbi (dynamic programming).

Toy vocab for the string `tokenization`. Higher score = more likely piece.
"""
        ),
        code(
            """
TOY_SCORES = {
    "t": -2.0,
    "o": -2.0,
    "k": -2.2,
    "e": -1.8,
    "n": -1.8,
    "i": -1.9,
    "z": -2.5,
    "a": -1.7,
    "token": -0.4,
    "tok": -0.9,
    "en": -1.1,
    "ization": -0.5,
    "ation": -0.8,
    "tion": -1.0,
}


def viterbi_segment(text: str, scores: dict[str, float]) -> list[str]:
    n = len(text)
    best = [float("-inf")] * (n + 1)
    back: list[int] = [-1] * (n + 1)
    chosen: list[str] = [""] * (n + 1)
    best[0] = 0.0
    for i in range(n):
        if best[i] == float("-inf"):
            continue
        for j in range(i + 1, n + 1):
            piece = text[i:j]
            if piece not in scores:
                continue
            cand = best[i] + scores[piece]
            if cand > best[j]:
                best[j] = cand
                back[j] = i
                chosen[j] = piece
    if best[n] == float("-inf"):
        raise ValueError(f"cannot segment {text!r} with this vocab")
    pieces: list[str] = []
    idx = n
    while idx > 0:
        pieces.append(chosen[idx])
        idx = back[idx]
    pieces.reverse()
    return pieces, best[n]


pieces, score = viterbi_segment("tokenization", TOY_SCORES)
print("best path:", pieces)
print("total score:", round(score, 3))
"""
        ),
        md(
            """
Try changing scores. If you make `"t"` extremely good and `"token"` terrible, Viterbi
will prefer a pile of single letters. Unigram training is mostly: **adjust those scores
so the corpus is cheap to encode, then delete pieces nobody needs.**
"""
        ),
        code(
            """
box = widgets.Text(
    value="tokenization",
    description="Text:",
    layout=widgets.Layout(width="90%"),
    style={"description_width": "50px"},
)
out = widgets.Output()


def _run(_change=None) -> None:
    with out:
        out.clear_output()
        try:
            pieces, score = viterbi_segment(box.value, TOY_SCORES)
            print("pieces:", pieces)
            print("score: ", round(score, 3))
        except ValueError as exc:
            print(exc)
            print("Only characters listed in TOY_SCORES can be segmented in this toy.")


box.observe(_run, names="value")
_run()
display(box, out)
"""
        ),
        md(
            """
## Production library: `sentencepiece` 0.2.2

**API rules for 0.2.2 (pybind11 rewrite):**

- Use `return_type=` — **not** the deprecated alias `out_type`.
- Load models with `SentencePieceProcessor.from_file(...)`.
- Train with `SentencePieceTrainer.train(...)`.
- Do **not** call `EncodeAsImmutableProto` / `return_type='immutable_proto'` (removed).

We train two tiny models on `data/tiny_corpus.txt`: Unigram and BPE. Vocab 100 is enough
for this English file. `byte_fallback` is off so the vocab stays small; a later cell turns
it on for non-English text.

**Install fallback.** `sentencepiece` ships a native wheel. If `uv sync` fails on your
OS/Python combo, skip training and load
`models/pretrained/sp_unigram_tiny.model` (checked into the repo).
"""
        ),
        code(
            """
import sentencepiece as spm

unigram_prefix = ARTIFACTS / "sp_unigram"
try:
    spm.SentencePieceTrainer.train(
        input=str(CORPUS),
        model_prefix=str(unigram_prefix),
        vocab_size=100,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=False,
        minloglevel=1,
    )
    model_path = str(unigram_prefix) + ".model"
    print("trained:", model_path)
except Exception as exc:
    model_path = str(PRETRAINED / "sp_unigram_tiny.model")
    print("training failed — using checked-in model:", model_path)
    print("reason:", exc)

sp_uni = spm.SentencePieceProcessor.from_file(model_path)
print("unigram vocab", len(sp_uni))
print(sp_uni.encode("tokenization is the first step", return_type=str))
print(sp_uni.encode("tokenization is the first step", return_type=int))
print("round-trip:", sp_uni.decode(sp_uni.encode("tokenization is the first step")))
pieces = sp_uni.encode("the cat", return_type=str)
print("the cat →", pieces)
assert any("\u2581" in p for p in pieces), pieces
print("assert ok — space mark ▁ appears in pieces")
"""
        ),
        code(
            """
bpe_prefix = ARTIFACTS / "sp_bpe"
try:
    spm.SentencePieceTrainer.train(
        input=str(CORPUS),
        model_prefix=str(bpe_prefix),
        vocab_size=100,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=False,
        minloglevel=1,
    )
    sp_bpe = spm.SentencePieceProcessor.from_file(str(bpe_prefix) + ".model")
    print("bpe vocab", len(sp_bpe))
    print(sp_bpe.encode("tokenization is the first step", return_type=str))
except Exception as exc:
    print("BPE train skipped:", exc)
    print("Continue with sp_uni from the previous cell.")
"""
        ),
        md(
            """
## The `▁` mark and language-agnostic encoding

Look at the pieces. Almost every “word start” begins with `▁`. There is no separate
whitespace pre-tokenizer — the underscore-ish block is the space.
"""
        ),
        code(
            """
samples = [
    "the cat sat",
    "tokenization",
    "unhappiness",
    "日本語",
    "বাংলা",
    "hello世界",
]
print(f"{'text':20} {'unigram':40} bpe")
for s in samples:
    u = " ".join(sp_uni.encode(s, return_type=str))
    b = " ".join(sp_bpe.encode(s, return_type=str))
    print(f"{s:20} {u:40} {b}")
"""
        ),
        md(
            """
Japanese and Bengali mostly become `[UNK]` or leftover characters here because the toy
corpus is English. Production multilingual SentencePiece models train on mixed text (or
enable byte fallback). Next cell trains a **byte-fallback** model so unknown scripts
become UTF-8 byte pieces like `<0xE0>` instead of a hard unknown.
"""
        ),
        code(
            """
bf_prefix = ARTIFACTS / "sp_unigram_bytes"
spm.SentencePieceTrainer.train(
    input=str(CORPUS),
    model_prefix=str(bf_prefix),
    vocab_size=400,
    model_type="unigram",
    character_coverage=1.0,
    byte_fallback=True,
    minloglevel=1,
)
sp_bf = spm.SentencePieceProcessor.from_file(str(bf_prefix) + ".model")
for s in ["tokenization", "বাংলা", "日本語", "👋"]:
    print(s, "→", sp_bf.encode(s, return_type=str))
"""
        ),
        md(
            """
## Subword regularization (Unigram only)

Training is deterministic. **Encoding** can sample other valid segmentations. That noise
helps models generalize. `nbest_encode` lists the best paths. `enable_sampling=True`
draws one.

Use `return_type=str` (0.2.2 name). Do not pass `out_type`.
"""
        ),
        code(
            """
text = "tokenization"
print("greedy:", sp_uni.encode(text, return_type=str))
print("nbest:")
for path in sp_uni.nbest_encode(text, nbest_size=5, return_type=str):
    print(" ", path)

print("samples:")
for _ in range(5):
    print(" ", sp_uni.encode(text, return_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))
"""
        ),
        md("## Interactive playground (trained Unigram model)"),
        code(
            """
box = widgets.Text(
    value="tokenization is the first step",
    description="Text:",
    layout=widgets.Layout(width="90%"),
    style={"description_width": "50px"},
)
out = widgets.Output()


def _run(_change=None) -> None:
    with out:
        out.clear_output()
        pieces = sp_uni.encode(box.value, return_type=str)
        ids = sp_uni.encode(box.value, return_type=int)
        print("pieces:", pieces)
        print("ids:   ", ids)
        print("decode:", sp_uni.decode(ids))


box.observe(_run, names="value")
_run()
display(box, out)
"""
        ),
        md(
            """
## Where you will see SentencePiece

- **T5**, mT5, ALBERT, XLNet — Unigram SentencePiece
- **Many Llama / Gemma / Mistral** releases ship a `.model` file from this library
- HuggingFace `tokenizers` can import those models, but training them is still
  `sentencepiece`

If a model card says `spiece.model` or `tokenizer.model`, it is almost certainly
SentencePiece.
"""
        ),
        md(
            """
## Exercises

1. Decode `sp_uni.encode("the cat", return_type=int)`. Where did the space go, and which
   piece carried it?
2. Compare Unigram vs BPE pieces for `unhappiness` on this tiny corpus. Which looks more
   stable if you change `vocab_size` to 80 and retrain?
3. Why did `বাংলা` need `byte_fallback=True` to avoid a wall of unknown characters?
4. Sampling: run the sampling cell twice. Why would a trainer want random segmentations
   during **model** training, but greedy segmentation at **inference**?

Golden notes: `INSTRUCTOR.md`. Capstone (model-card → algorithm) is in `01_bpe.ipynb`.
"""
        ),
    ]


def main() -> None:
    write("01_bpe.ipynb", bpe_notebook(), "BPE")
    write("02_wordpiece.ipynb", wordpiece_notebook(), "WordPiece")
    write("03_sentencepiece.ipynb", sentencepiece_notebook(), "SentencePiece")


if __name__ == "__main__":
    main()
