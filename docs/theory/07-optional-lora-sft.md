# Optional homework — LoRA SFT (stock tokenizer)

Not part of notebook 04. Separate session. GPU optional.

**Use the base model’s tokenizer.** MedMCQA LoRA on Qwen. Prove you can SFT. Do not mix in the custom medical BPE.

## Install

```bash
uv sync --extra sft
```

Pulls `torch`, `accelerate>=1.14`, `peft>=0.20`, `trl>=1.12`. Default `uv sync` for class laptops stays CPU-only.

## Shape of the run

```python
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# Stock tokenizer. Do not load artifacts/medical-bpe-hf.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
args = SFTConfig(
    output_dir="artifacts/medmcqa-lora",
    learning_rate=1e-4,
    max_length=1024,
    completion_only_loss=True,
    bf16=True,
)
# Map openlifescienceai/medmcqa to prompt/completion.
# SFTTrainer(..., peft_config=peft_config, processing_class=base_tokenizer)
```

Start with `Qwen/Qwen3-0.6B` on a 16GB GPU. 8B wants QLoRA (`BitsAndBytesConfig` on `SFTTrainer`).

## Why stock tokenizer here

SFT teaches the model to answer exam-style items. It does not need a new alphabet. If you later add 200 drug strings, that is doc `06` move (2), after this homework works.

## Safety

MedMCQA LoRA is an exam-taking adapter. Not medical advice. Not a clinician.
