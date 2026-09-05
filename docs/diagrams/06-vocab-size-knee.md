# Vocab-size knee: compression vs embedding cost

> Companion to [08 — vocab size trade-off](../theory/08-vocab-size-tradeoff.md).
> Avg tokens/doc falls fast, then flattens. Embedding params keep climbing.

## Diminishing returns (lesson numbers)

Large-corpus **worked example**. PubMed numbers in the notebook will sit left of this knee.

```mermaid
flowchart LR
  subgraph compression [Avg tokens per document]
    v16["16k\n1250"]
    v32["32k\n1050"]
    v50["50k\n980"]
    v64["64k\n960"]
    v100["100k\n950"]
    v16 --> v32 --> v50 --> v64 --> v100
  end
```

```text
16k   ████████████████████████████████████  1250
32k   ██████████████████████████████        1050   −16%
50k   ████████████████████████████          980    −7%
64k   ███████████████████████████           960    −2%
100k  ███████████████████████████           950    −1%   ← stop
```

64k → 100k saves ~1% of tokens. Rule of thumb: pick **64k** on that corpus.

## Embedding table still grows

```text
params ≈ vocab × d_model   (d_model = 1024)

16k    ████                 16.4M   1.00×
32k    ████████             32.8M   2.00×
50k    ████████████         51.2M   3.12×
64k    ████████████████     65.5M   4.00×
100k   ████████████████████ 102.4M  6.25×
```

After the knee you pay 6× the embedding rows for almost no shorter sequences.

## This lab's corpus band

```mermaid
flowchart TD
  corpus["PubMed train\n45k abstracts\ntens of millions of tokens"]
  band["Small corpus less than 1B\nrecommended 16k to 32k"]
  knee["Notebook knee helper\nsmallest size within 2% of best\navg tokens/doc"]
  trap["Do not load that tokenizer onto Qwen\nsee theory 06"]
  corpus --> band
  band --> knee
  knee --> trap
```

50k as a "modern LLM sweet spot" assumes 1B–100B+ pretraining tokens. Not 45k abstracts.
