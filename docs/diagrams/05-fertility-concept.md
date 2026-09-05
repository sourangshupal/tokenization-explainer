# 🌱 Fertility: Why Fewer Tokens Per Word Wins

> **Fertility** = average tokens per whitespace word.  
> Lower is better. A tokenizer that splits `acetylcholinesterase` into 2 pieces knows medicine.
> One that splits it into 8 is wasting context window and gradient signal.

## What happens to one clinical term

```mermaid
flowchart LR
    WORD["🔬 Input word\n'acetylcholinesterase'\n(Alzheimer's enzyme)"]

    subgraph cl100k_path ["🤖 cl100k  (GPT-4, 100k vocab)"]
        C1["ac"] --> C2["et"] --> C3["yl"] --> C4["cho"] --> C5["lin"] --> C6["ester"] --> C7["ase"]
        CSCORE["📊 7 tokens\nFertility = 7.0 for this word"]
    end

    subgraph custom_path ["🏥 custom-med  (16k, PubMed-trained)"]
        M1["acetylcholin"] --> M2["ester"] --> M3["ase"]
        MSCORE["📊 3 tokens\nFertility = 3.0 for this word"]
    end

    WORD --> cl100k_path
    WORD --> custom_path

    style MSCORE fill:#e8f5e9,stroke:#2e7d32
    style CSCORE fill:#ffebee,stroke:#c62828
```

## Why low fertility matters

```mermaid
flowchart TD
    LF["🎯 Low Fertility\n(fewer tokens per word)"]

    B1["📐 Context window efficiency\nGPT-4 context = 128k tokens\nA 200-word abstract:\n  cl100k → ~294 tokens\n  custom-med → ~276 tokens\n  = 18 extra abstracts fit per batch"]

    B2["🎓 Better gradient signal\nEach token gets its own embedding vector\nFewer, longer tokens → richer representation\nof domain concepts per embedding slot"]

    B3["💰 Lower compute cost\nFewer tokens = fewer attention ops (O(n²))\n5 % fewer tokens → ~10 % cheaper attention"]

    B4["🧩 Semantic wholeness\n'acetylcholinesterase' in 3 pieces\n→ model can learn the whole concept\nin 3 vs 7 embedding updates"]

    LF --> B1
    LF --> B2
    LF --> B3
    LF --> B4

    style LF fill:#fff3e0,stroke:#e65100
```

## The actual numbers from this lab

**Medical held-out set** (`pubmed_heldout.jsonl` — 5 000 abstracts, lower = better)

| Rank | Tokenizer | Fertility | Notes |
|------|-----------|-----------|-------|
| 🥇 1 | 🏥 **custom-med** (16k) | **1.381** | Domain training wins |
| 🥈 2 | 🤖 o200k (200k) | 1.439 | 12× bigger vocab, still loses |
| 🥉 3 | 🤖 cl100k (100k) | 1.470 | 6× bigger vocab, still loses |
| 4 | 📰 general-bpe (16k) | 1.759 | Same size as custom-med — domain is the secret |

> 🏆 `custom-med` wins with fertility **1.381**  
> Even `cl100k` (6× bigger vocab) loses — at **1.470**  
> `general-bpe` (same 16k size) is worst at **1.759** — vocab size isn't the secret, **domain is**

## The flip side — general text

**General held-out set** (`general_heldout.jsonl` — 5 000 wikitext paragraphs, lower = better)

| Rank | Tokenizer | Fertility | Notes |
|------|-----------|-----------|-------|
| 🥇 1 | 🤖 o200k (200k) | **1.401** | Web-scale vocab wins on web text |
| 🥈 2 | 🤖 cl100k (100k) | 1.434 | Strong general baseline |
| 🥉 3 | 📰 general-bpe (16k) | 1.503 | Small but general-trained |
| 4 | 🏥 **custom-med** (16k) | 1.812 | Specialised vocab — expected to lose here |

> 🔄 `custom-med` **loses** on general text (1.812 — worst!)  
> This is expected and healthy — it specialised its 16k budget on medical morphemes  
> A student who notices this flip has understood domain specialisation

## Fertility over a full abstract

```mermaid
flowchart LR
    ABS["📄 One PubMed abstract\n~200 whitespace words"]

    subgraph tokens_cl ["🤖 cl100k encoding"]
        TC["~294 tokens\n(fertility 1.470 × 200)"]
    end

    subgraph tokens_cm ["🏥 custom-med encoding"]
        TM["~276 tokens\n(fertility 1.381 × 200)"]
    end

    ABS --> tokens_cl
    ABS --> tokens_cm

    DIFF["💡 18 fewer tokens per abstract\n× 1 000 abstracts in a batch\n= 18 000 fewer tokens\n≈ 141 extra abstracts fit in same\n128k context window"]

    tokens_cl --> DIFF
    tokens_cm --> DIFF

    style TM fill:#e8f5e9,stroke:#2e7d32
    style DIFF fill:#fff3e0,stroke:#e65100
```

---

## Glossary at a glance

| Term | Formula | What it tells you |
|------|---------|------------------|
| **Fertility** | tokens ÷ whitespace words | Avg pieces per word (lower = more efficient) |
| **Chars/token** | characters ÷ tokens | Avg characters per token (higher = denser) |
| **Single-token rate** | fraction of terms = 1 token | Vocab coverage of domain jargon |

---
*Back to theory: [01 why tokenization matters](../theory/01-why-tokenization-matters.md) · [04 metrics](../theory/04-metrics.md)*
