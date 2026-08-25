(() => {
  /** Classic Sennrich toy set — default for algo labs / WordPiece demo. */
  const SENNRICH = `low
low
low
low
low
lower
lower
newest
newest
newest
widest
widest`;

  /** Richer course corpus — same file as data/tiny_corpus.txt (notebooks HF / SP). */
  const COURSE = `low
low
low
low
low
lower
lower
newest
newest
newest
widest
widest
the cat sat on the mat
the cat sat on the mat
tokenization is the first step of every language model
unhappiness is not a word you see every day
playing players played
walking walks walked walker
low lower lowest
new newer newest
wide wider widest
cats and dogs play in the garden
language models read tokens not characters
byte pair encoding builds a vocabulary from frequent pairs
wordpiece prefers likely pieces not just frequent ones
sentencepiece can tokenize text without splitting on spaces first
hello world
this is a test
the quick brown fox jumps over the lazy dog
machine learning starts with data
neural networks map tokens to vectors
subword units handle rare words
out of vocabulary words break word level tokenizers
cafe
namaste
hello world again
this is another test sentence
rare words like unhappiness still split`;

  window.TokenCorpora = {
    SENNRICH,
    COURSE,
    labels: {
      sennrich: "Sennrich toy (default)",
      course: "Course corpus (notebooks)",
    },
  };
})();
