# 🔬 How Byte-Level BPE Training Works

> **The core idea:** BPE is a greedy frequency algorithm. It finds the most common pair of adjacent
> symbols and merges them. Run it on medical text and medical morphemes win. Run it on the web
> and medical terms stay fragmented.

```mermaid
flowchart TD
    A["📄 Raw Corpus\n(e.g. 45k PubMed abstracts\nor 45k wikitext paragraphs)"]
    B["🔡 Byte Initialisation\nEvery UTF-8 byte becomes\na single token  →  256 base symbols\nNo unknown tokens possible"]
    C["📊 Count All Adjacent Pairs\nScan the entire corpus\ne.g.  'e','m' appears 180k times\n      'a','c' appears 92k times"]
    D{"🏆 Most Frequent Pair?"}
    E["⚡ Merge → New Token\nReplace every occurrence of the pair\nwith a single new symbol\ne.g.  'em'  is now one token\nVocab size grows by 1"]
    F{"🎯 Reached vocab_size?\n(e.g. 16 000)"}
    G["💾 Save tokenizer.json\nMerge rules + vocab written to disk\nRe-usable for encoding any new text"]
    H["🏥 Medical corpus result\nHigh-freq medical pairs merged first\n'cholinesterase' survives as 1–2 tokens\n'empagliflozin' → partially merged"]
    I["📰 General corpus result\nHigh-freq web pairs merged first\n'cholinesterase' never seen enough\n→ stays fragmented  (6–8 pieces)"]

    A --> B
    B --> C
    C --> D
    D -->|"yes — merge it"| E
    E --> C
    D -->|"no more pairs\nor budget met"| F
    F -->|"no — keep going"| C
    F -->|"yes — done"| G
    G --> H
    G --> I

    style E fill:#fff3e0,stroke:#e65100
    style D fill:#e8f5e9,stroke:#2e7d32
    style F fill:#e8f5e9,stroke:#2e7d32
```

## 🔑 Key insight

The algorithm is identical for medical and general corpora.
The **only** difference is which pairs appear most often in the training text.

| Step | Medical corpus (PubMed) | General corpus (wikitext) |
|------|------------------------|--------------------------|
| Most frequent pairs | `ch`→`o`→`li`→`ne`→`ster`→`ase` (clinical chemistry) | `th`→`e`, `in`→`g`, common web morphemes |
| After 16k merges | `cholinesterase` = 1 token | `cholinesterase` = 6–8 fragments |
| Fertility on PubMed held-out | **1.381** (winner) | 1.747 (worst) |

> This is why `general-bpe` (same 16k vocab, same algorithm, general corpus) scores **worse** than
> `custom-med` on medical text — it spent its merge budget on the wrong frequency distribution.

---
*Back to theory: [03 custom domain tokenizers](../theory/03-custom-domain-tokenizers.md)*
