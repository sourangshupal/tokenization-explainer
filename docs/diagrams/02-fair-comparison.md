# ⚖️ The 4-Tokenizer Fairness Design

> **The problem with a naive comparison:** Comparing a 16k domain BPE against cl100k (100k vocab)
> mixes two variables — **domain** AND **vocab size**. To isolate domain as the cause, we need
> a same-size general baseline.

## The 2×2 design

```mermaid
flowchart TB
    subgraph large ["📚 Large vocab (100k+)"]
        CL["🤖 cl100k  ~100k\nGPT-4 training data\n🌐 General domain"]
        O2["🤖 o200k  ~200k\nGPT-4o training data\n🌐 General domain"]
        MISSING["🚫 No large-vocab\nmedical BPE\nin this lab"]
    end

    subgraph small ["📦 Small vocab (16k)"]
        GB["📰 general-bpe  16k\nwikitext-103\n🌐 General domain"]
        CM["🏥 custom-med  16k\nPubMed abstracts\n🏥 Medical domain"]
    end

    ISO["🔬 Isolation: general-bpe vs custom-med\nSame algorithm · Same 16k vocab\nOnly training domain differs\n→ domain is the only variable"]

    GB <-->|"same size\ndifferent domain"| CM
    CM --> ISO
    GB --> ISO

    style CM fill:#fff3e0,stroke:#e65100
    style GB fill:#e3f2fd,stroke:#1565c0
    style ISO fill:#e8f5e9,stroke:#2e7d32
    style MISSING fill:#f5f5f5,stroke:#9e9e9e,color:#9e9e9e
```

## What each tokenizer proves

```mermaid
flowchart LR
    subgraph general_large ["🌐 Large General Vocab"]
        CL["🤖 cl100k\n~100k vocab\nGPT-4 training data\nBaseline to beat"]
        O2["🤖 o200k\n~200k vocab\nGPT-4o training data\nStronger baseline"]
    end

    subgraph control_zone ["🔬 Fairness Control Zone\n(same 16k vocab — different domain)"]
        GB["📰 general-bpe\n16k vocab\nwikitext-103\nSame algo as custom-med"]
        CM["🏥 custom-med\n16k vocab\nPubMed abstracts\nSame algo as general-bpe"]
    end

    subgraph eval_sets ["📋 Held-Out Evaluation Sets\n(never seen during training)"]
        MH["🔬 pubmed_heldout.jsonl\n5 000 medical abstracts"]
        GH["📰 general_heldout.jsonl\n5 000 wikitext paragraphs"]
    end

    CL --> MH
    CL --> GH
    O2 --> MH
    O2 --> GH
    GB --> MH
    GB --> GH
    CM --> MH
    CM --> GH

    style CM fill:#fff3e0,stroke:#e65100
    style GB fill:#e3f2fd,stroke:#1565c0
    style MH fill:#e8f5e9,stroke:#2e7d32
```

## The proof table — held-out fertility results

| Tokenizer | Vocab | Medical held-out fertility | General held-out fertility |
|-----------|-------|---------------------------|---------------------------|
| 🏥 **custom-med** | 16k | **1.381 ✅ wins** | 1.812 ❌ loses |
| 📰 general-bpe | 16k | 1.759 ❌ | **1.503 ✅ wins** |
| 🤖 cl100k | ~100k | 1.470 | 1.434 |
| 🤖 o200k | ~200k | 1.439 | 1.401 |

> **Reading the table:** `custom-med` (1.381) beats `general-bpe` (1.759) by **0.378**
> on the same medical held-out set.  
> They have **identical algorithm** and **identical vocab size (16k)**.  
> The only variable is training domain. **Domain is the cause.**

## The isolation argument

```mermaid
flowchart TD
    Q["❓ Why does custom-med win\non medical text?"]

    H1["Hypothesis 1:\nVocab size\n(16k vs 100k)"]
    H2["Hypothesis 2:\nAlgorithm\n(byte BPE vs tiktoken)"]
    H3["Hypothesis 3:\nTraining domain\n(PubMed vs web)"]

    T1["🔴 Rejected\ngeneral-bpe also has 16k vocab\nbut fertility = 1.759\n(WORSE than cl100k 1.460)"]
    T2["🔴 Rejected\nboth use byte-level BPE\nsame pretokenizer, same decoder"]
    T3["✅ Confirmed\ncustom-med 1.381  vs  general-bpe 1.759\nOnly training data differs\n→ domain frequency drives merges"]

    Q --> H1
    Q --> H2
    Q --> H3
    H1 --> T1
    H2 --> T2
    H3 --> T3

    style T3 fill:#e8f5e9,stroke:#2e7d32
    style T1 fill:#ffebee,stroke:#c62828
    style T2 fill:#ffebee,stroke:#c62828
```

---
*Back to theory: [04 metrics](../theory/04-metrics.md) · [05 why custom wins](../theory/05-why-custom-wins-in-healthcare.md)*
