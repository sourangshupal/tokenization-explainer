# Why tokenization matters

A language model does not read characters or words. It reads **token IDs**. Each ID indexes one row of the embedding table. That row is the only meaning the network has for that piece of text.

## Split word, split meaning

`empagliflozin` as **one** token → one vector for the drug.

`empagliflozin` as **six** pieces → six vectors. Attention must reassemble “this is one drug” every time the word appears. The model can learn that trick if those splits existed during pretraining. It is wasteful. On a new tokenizer with new IDs, there is nothing to reassemble yet.

## The window is tokens, not words

Context length is counted in tokens. An 800-word discharge summary might be ~1,400 tokens with a general vocab and ~1,000 with a medical vocab. The extra 400 tokens are **fertility tax**: less room for history, labs, and the actual question.

Inference cost and latency scale with token count. Over-segmentation is slower and more expensive even when quality is unchanged.

## Training a tokenizer is not training a model

Tokenizer training is a **deterministic** statistical pass: count pairs, merge, write a vocab. Model training is gradient descent on embeddings and weights.

HuggingFace’s course: you train a new tokenizer on domain data, then you train (or continue-pretrain) a **model** that uses that tokenizer. This lab stops at the tokenizer and the comparison. Doc `06` covers what happens if you skip the model step.
