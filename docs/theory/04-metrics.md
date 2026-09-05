# Metrics

> ⚖️ **Fairness design diagram:** [4-tokenizer 2×2 design → docs/diagrams/02-fair-comparison.md](../diagrams/02-fair-comparison.md)  
> 🌱 **Fertility concept diagram:** [What fertility means → docs/diagrams/05-fertility-concept.md](../diagrams/05-fertility-concept.md)

Do not "eyeball chips" and declare a winner. Compute.

## Fertility

```text
fertility = n_tokens / n_whitespace_words
```

Lower is denser. Report it on:
- **held-out domain text** (`pubmed_heldout.jsonl`) — the fair in-domain eval; custom-med should win here
- **held-out general text** (`general_heldout.jsonl`) — cross-domain; custom-med is expected to lose
- `medical_probes.txt` — 20 curated illustration phrases (not the pass bar; covered in `05`)

Also useful: **tokens per 100 characters** — insensitive to how you define a "word."

## Single-token rate

For a list of target domain terms (`empagliflozin`, `creatinine`, `myocardial`, ...):

```text
single_token_rate = count(terms encoded as 1 piece) / count(terms)
```

Custom-med should be higher on the medical term list. General tokenizers often sit in the low
tens of percent. A domain BPE trained on abstracts that mention these words frequently can put
common drugs and lab names at 1–2 pieces each.

## Round-trip

```text
decode(encode(x)) == x
```

Required for production. Byte-level BPE should round-trip all Unicode (`μg`, `β-lactam`, `Naïve`).
If it fails, the pretokenizer/decoder pair is wrong.

## The fair baseline: same-size general BPE

Comparing custom-med (16k vocab, domain) against `cl100k` (100k vocab, general) conflates two
variables: **domain** and **vocab size**. The honest control is a 16k BPE trained on general
text (wikitext-103) — same algorithm, same vocab budget, different training domain.

| tokenizer | vocab size | domain |
|-----------|-----------|--------|
| `cl100k` | ~100k | general |
| `o200k` | ~200k | general |
| `general-bpe` | 16k | general (wikitext) |
| **`custom-med`** | 16k | medical (PubMed) |

If `custom-med` beats `general-bpe` on medical held-out text, the gap is caused by **domain**,
not vocab size. If they tie, domain didn't matter — training corpus quality did.

## Control set

Custom **may lose** on `general_heldout.jsonl`. That is a pass, not a fail. You trained on
abstracts, not novels. If custom also crushes general English, the corpus leaked too much
generic prose — inspect `medical_corpus.txt` or `pubmed_train.jsonl`.

## Winner column

On each probe: fewest pieces wins. Ties are ties. **The lab's golden assert (tests): custom-med
fertility < cl100k fertility on held-out PubMed abstracts.** Not the 20-probe mean.
