# ⚠️ The Pretrained-Model Trap

> **"I'll just swap in my custom tokenizer on Qwen/LLaMA and do LoRA."**
> This sounds reasonable. It is catastrophically wrong.

## The core mismatch

```mermaid
flowchart TD
    TJ["🏥 custom-med tokenizer.json\n16 000 token IDs\nlearned from PubMed"]
    PW["🧠 Qwen-2.5 weights\n152 000 embedding rows\none row per Qwen token ID"]

    TJ -- "❌ ID 4521 in custom-med\n= 'gluc'" --> MISMATCH
    PW -- "❌ Row 4521 in Qwen\n= learned repr of 'the'" --> MISMATCH

    MISMATCH["🔥 Embedding mismatch!\nRow 4521 was trained to mean 'the'\nnow receives signal for 'gluc'\nAll 16 000 IDs point to wrong rows"]

    MISMATCH --> SCRAMBLE["💥 Scrambled model\nLoRA only updates a few adapters\nUnderlying embedding confusion\npersists throughout fine-tuning\nModel outputs nonsense or random text"]

    style MISMATCH fill:#ffebee,stroke:#c62828
    style SCRAMBLE fill:#ffebee,stroke:#c62828
```

## Two valid paths, one fatal path

```mermaid
flowchart LR
    START["🎯 Goal: Medical LLM"]

    subgraph wrong ["❌ WRONG PATH — The Trap"]
        W1["Take Qwen-2.5 weights\n(trained with Qwen tokenizer)"]
        W2["Swap tokenizer.json\nfor custom-med (16k vocab)"]
        W3["LoRA fine-tune on\nmedical text"]
        W4["🔥 FAIL\nIDs don't match embedding rows\nGradients flow into wrong rows\nModel never converges properly"]
        W1 --> W2 --> W3 --> W4
    end

    subgraph right_a ["✅ RIGHT PATH A — Keep the tokenizer"]
        A1["Take Qwen-2.5 weights\n(trained with Qwen tokenizer)"]
        A2["Keep Qwen tokenizer\n(IDs still match rows!)"]
        A3["add_tokens() for\nnew medical terms"]
        A4["resize_token_embeddings()\n→ new rows initialised randomly"]
        A5["LoRA fine-tune\nNew rows learn from scratch\nOld rows adapt via LoRA"]
        A6["✅ Works\nVocab and embeddings stay aligned"]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end

    subgraph right_b ["✅ RIGHT PATH B — Train from scratch"]
        B1["custom-med tokenizer\n(16k, PubMed-trained)"]
        B2["Random weight init\n(new 16k embedding table)"]
        B3["Pre-train LM from scratch\non medical corpus"]
        B4["LoRA or full fine-tune\nfor downstream tasks"]
        B5["✅ Works\nTokenizer and embeddings\nborn together — always aligned"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    START --> wrong
    START --> right_a
    START --> right_b

    style wrong fill:#ffebee,stroke:#c62828
    style right_a fill:#e8f5e9,stroke:#2e7d32
    style right_b fill:#e8f5e9,stroke:#2e7d32
```

## Why the alignment matters: a concrete example

```mermaid
flowchart LR
    subgraph qwen_world ["🧠 Qwen's world (trained)"]
        QT["Token ID 7821\n= Ġmedical\n(Qwen vocab)"]
        QE["Embedding row 7821\n[0.34, -0.12, 0.78, ...]\nEncodes meaning of 'medical'"]
        QT <-->|"✅ match"| QE
    end

    subgraph swapped ["⚠️ After custom-med swap"]
        ST["Token ID 7821\n= 'empagliflozin'\n(custom-med vocab)"]
        SE["Embedding row 7821\n[0.34, -0.12, 0.78, ...]\nStill encodes 'medical' meaning!"]
        ST <-->|"❌ mismatch"| SE
    end

    qwen_world -- "Swap tokenizer\nonly" --> swapped

    style swapped fill:#fff3e0,stroke:#e65100
```

## The golden rule

```mermaid
flowchart TD
    RULE["📏 Golden Rule\nA tokenizer and its model weights are\n**inseparable**. They were trained together.\nYou cannot split them without breaking both."]

    C1["🔑 Keep tokenizer + weights together\n→ extend vocabulary safely with add_tokens"]
    C2["🏗️ Train together from scratch\n→ build your own LM on custom-med tokens"]

    RULE --> C1
    RULE --> C2

    style RULE fill:#fff3e0,stroke:#e65100
```

> **This lab** focuses on tokenizers, not LM training. But understanding this trap helps you see
> why the tokenizer step is not just a preprocessing detail — it **determines what model you can use**.

---
*Back to theory: [06 pretrained model trap](../theory/06-pretrained-model-trap.md)*
