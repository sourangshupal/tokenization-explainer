# The pretrained-model trap

> ⚠️ **Visual trap diagram:** [Wrong vs right paths → docs/diagrams/04-pretrained-trap.md](../diagrams/04-pretrained-trap.md)

**Board sentence.** Tokenizer IDs must match embedding rows. Custom BPE changes the alphabet. A pretrained model’s embeddings still speak the old alphabet.

## Predict, then reveal

Prompt: *Fertility dropped 20% on discharge summaries. You want to LoRA Llama-8B. Do you replace the tokenizer?*

**No.** Replacing `tokenizer.json` remaps every string to new IDs. Row 128 of Llama’s embedding table is still whatever Llama’s original token 128 was — not your new merge. LoRA on `q_proj` / `v_proj` cannot invent a new language for random IDs. Loss may fall. Quality will not.

That is **not** the same bug as “suboptimal splits.” Biomedical models are somewhat robust to messy pieces **if those pieces were there during pretrain**. Random new IDs are a different failure.

## Legal moves (not this lab)

1. **Train (or continue-pretrain) a model with the custom tokenizer.** Path C. Small LM from scratch, then optional SFT. Vocab and embeddings are born together.
2. **Keep the base tokenizer. Add a few hundred whole medical strings** (`tokenizer.add_tokens`, `resize_token_embeddings`, PEFT `trainable_token_indices`). Path A-ish. Production LoRA path. You did **not** replace the alphabet; you extended it.
3. **Path B (mentioned only):** train a domain BPE, mine tokens that are 1 piece in-domain and 4+ in Llama, then `add_tokens` those strings. Still an extension, not a swap.

## Illegal move (this lab’s anti-pattern)

```python
# WRONG — do not do this
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tok = AutoTokenizer.from_pretrained("artifacts/medical-bpe-hf")  # new vocab
# then LoRA attention and hope
```

IDs and rows no longer mean the same thing.

## What this lab grades

You can explain the chips and the fertility table. You can say the board sentence. You do **not** ship a Qwen adapter with a replaced tokenizer.
