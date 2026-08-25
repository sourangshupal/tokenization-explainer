(() => {
  const PALETTE = ["#22d3ee", "#a5b4fc", "#fbbf24", "#fb7185", "#34d399", "#c084fc", "#67e8f9"];

  /**
   * Hash a string to a stable chip color.
   * @param {string} token
   * @returns {string}
   */
  function chipColor(token) {
    let h = 0;
    for (const ch of token) {
      h = (h * 31 + (ch.codePointAt(0) ?? 0)) | 0;
    }
    return PALETTE[Math.abs(h) % PALETTE.length];
  }

  /**
   * Word-level tokens: letters, numbers, or punctuation runs.
   * @param {string} text
   * @returns {string[]}
   */
  function wordTokens(text) {
    return text.match(/[\p{L}\p{M}]+|\p{N}+|[^\s\p{L}\p{N}\p{M}]+/gu) ?? [];
  }

  /**
   * Naive subword: first chunk of a word stays bare, later chunks get ##.
   * @param {string} text
   * @param {number} [chunk=3]
   * @returns {string[]}
   */
  function naiveSubwords(text, chunk = 3) {
    const pieces = [];
    for (const w of wordTokens(text)) {
      if (!/^[\p{L}\p{N}\p{M}]+$/u.test(w)) {
        pieces.push(w);
        continue;
      }
      if (w.length <= chunk) {
        pieces.push(w);
        continue;
      }
      pieces.push(w.slice(0, chunk));
      for (let i = chunk; i < w.length; i += chunk) {
        pieces.push(`##${w.slice(i, i + chunk)}`);
      }
    }
    return pieces;
  }

  /**
   * Unicode code-point tokens (not UTF-16 code units).
   * @param {string} text
   * @returns {string[]}
   */
  function codePoints(text) {
    return Array.from(text);
  }

  /**
   * Grapheme clusters when Intl.Segmenter exists.
   * @param {string} text
   * @returns {string[]}
   */
  function graphemes(text) {
    if (typeof Intl !== "undefined" && Intl.Segmenter) {
      return [...new Intl.Segmenter("en", { granularity: "grapheme" }).segment(text)].map(
        (s) => s.segment,
      );
    }
    return codePoints(text);
  }

  /**
   * UTF-8 bytes as 0–255 integers.
   * @param {string} text
   * @returns {number[]}
   */
  function utf8Bytes(text) {
    return [...new TextEncoder().encode(text)];
  }

  /**
   * @param {number} b
   * @returns {string}
   */
  function byteHex(b) {
    return `0x${b.toString(16).padStart(2, "0")}`;
  }

  /**
   * Visible label for a token that may be whitespace.
   * @param {string} token
   * @returns {string}
   */
  function visible(token) {
    if (token === " ") return "␣";
    if (token === "\n") return "\\n";
    if (token === "\t") return "\\t";
    return token;
  }

  /**
   * @param {string[]} tokens
   * @param {HTMLElement} root
   * @param {{unk?: Set<string>}} [opts]
   */
  function renderChips(tokens, root, opts = {}) {
    root.replaceChildren();
    const unk = opts.unk ?? new Set();
    for (const token of tokens) {
      const el = document.createElement("span");
      el.className = "chip";
      if (unk.has(token) || token === "[UNK]") el.classList.add("unk");
      if (token === " " || token === "\n") el.classList.add("space");
      el.textContent = visible(token);
      el.title = `U+${[...token].map((c) => (c.codePointAt(0) ?? 0).toString(16).toUpperCase().padStart(4, "0")).join(" ")}`;
      el.style.borderColor = chipColor(token);
      el.style.color = chipColor(token);
      root.append(el);
    }
  }

  /**
   * @param {string[]} symbols
   * @param {string} a
   * @param {string} b
   * @returns {string[]}
   */
  function mergePair(symbols, a, b) {
    const out = [];
    for (let i = 0; i < symbols.length; ) {
      if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
        out.push(a + b);
        i += 2;
      } else {
        out.push(symbols[i]);
        i += 1;
      }
    }
    return out;
  }

  /**
   * Mini frequency BPE for the live lab. Words end with </w>.
   * @param {string} corpusText
   * @param {number} numMerges
   * @returns {{merges: {pair: [string, string], merged: string, count: number}[], words: {symbols: string[], count: number}[]}}
   */
  function trainMiniBpe(corpusText, numMerges) {
    const counts = new Map();
    const words = corpusText.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
    for (const w of words) {
      counts.set(w, (counts.get(w) ?? 0) + 1);
    }

    let vocabWords = [...counts.entries()].map(([w, count]) => ({
      symbols: [...w, "</w>"],
      count,
    }));

    const merges = [];
    const steps = Math.max(0, Math.floor(numMerges));

    for (let step = 0; step < steps; step += 1) {
      const pairCounts = new Map();
      for (const { symbols, count } of vocabWords) {
        for (let i = 0; i < symbols.length - 1; i += 1) {
          const key = `${symbols[i]}\0${symbols[i + 1]}`;
          pairCounts.set(key, (pairCounts.get(key) ?? 0) + count);
        }
      }

      let best = null;
      let bestCount = 0;
      for (const [k, v] of pairCounts) {
        if (v > bestCount) {
          bestCount = v;
          best = k;
        }
      }
      if (!best || bestCount < 2) break;

      const [a, b] = best.split("\0");
      merges.push({ pair: [a, b], merged: a + b, count: bestCount });
      vocabWords = vocabWords.map(({ symbols, count }) => ({
        symbols: mergePair(symbols, a, b),
        count,
      }));
    }

    return { merges, words: vocabWords };
  }

  /**
   * Apply learned BPE merges to one word.
   * @param {string} word
   * @param {{pair: [string, string]}[]} merges
   * @returns {string[]}
   */
  function applyMerges(word, merges) {
    let symbols = [...word.toLowerCase(), "</w>"];
    for (const { pair } of merges) {
      symbols = mergePair(symbols, pair[0], pair[1]);
    }
    return symbols;
  }

  /**
   * Encode a sentence with trained mini-BPE (word-wise).
   * @param {string} text
   * @param {{pair: [string, string]}[]} merges
   * @returns {string[]}
   */
  function encodeMiniBpe(text, merges) {
    const pieces = [];
    const parts = text.match(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]+|\s+/gu) ?? [];
    for (const part of parts) {
      if (/^[\p{L}\p{N}]+$/u.test(part)) {
        pieces.push(...applyMerges(part, merges));
      } else if (!/^\s+$/u.test(part)) {
        pieces.push(part);
      }
    }
    return pieces;
  }

  /**
   * Glue BPE pieces: strip </w>, put spaces between words.
   * @param {string[]} pieces
   * @returns {string}
   */
  function detokenizeBpe(pieces) {
    let out = "";
    for (const p of pieces) {
      if (p.endsWith("</w>")) {
        out += p.slice(0, -"</w>".length) + " ";
      } else {
        out += p;
      }
    }
    return out.trim();
  }

  /**
   * Glue WordPiece pieces: strip ## and concatenate.
   * @param {string[]} pieces
   * @returns {string}
   */
  function detokenizeWordPiece(pieces) {
    let out = "";
    for (const p of pieces) {
      if (p === "[UNK]") {
        out += (out ? " " : "") + "[UNK]";
        continue;
      }
      if (p.startsWith("##")) out += p.slice(2);
      else out += (out ? " " : "") + p;
    }
    return out;
  }

  /**
   * Glue SentencePiece pieces: ▁ → space.
   * @param {string[]} pieces
   * @returns {string}
   */
  function detokenizeSentencePiece(pieces) {
    return pieces.join("").replace(/\u2581/g, " ").replace(/^ /, "");
  }

  window.TokenLab = {
    chipColor,
    wordTokens,
    naiveSubwords,
    codePoints,
    graphemes,
    utf8Bytes,
    byteHex,
    visible,
    renderChips,
    mergePair,
    trainMiniBpe,
    applyMerges,
    encodeMiniBpe,
    detokenizeBpe,
    detokenizeWordPiece,
    detokenizeSentencePiece,
  };
})();
