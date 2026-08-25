(() => {
  const lab = window.TokenLab;
  if (!lab) return;

  const MARK = "\u2581";

  /**
   * @param {string} text
   * @returns {Map<string, number>}
   */
  function wordCounts(text) {
    const counts = new Map();
    const words = text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
    for (const w of words) {
      counts.set(w, (counts.get(w) ?? 0) + 1);
    }
    return counts;
  }

  /**
   * WordPiece trainer: max score freq(ab)/(freq(a)*freq(b)) among pairs with freq >= 2.
   * @param {string} corpusText
   * @param {number} numMerges
   * @returns {{merges: {pair: [string, string], merged: string, score: number, count: number}[], vocab: Set<string>}}
   */
  function trainWordPiece(corpusText, numMerges) {
    const counts = wordCounts(corpusText);
    let splits = [...counts.entries()].map(([word, count]) => ({
      word,
      symbols: [...word],
      count,
    }));
    const vocab = new Set();
    for (const { symbols } of splits) {
      for (const s of symbols) vocab.add(s);
    }

    const merges = [];
    const steps = Math.max(0, Math.floor(numMerges));

    for (let step = 0; step < steps; step += 1) {
      const pairCounts = new Map();
      const singles = new Map();
      for (const { symbols, count } of splits) {
        for (const s of symbols) {
          singles.set(s, (singles.get(s) ?? 0) + count);
        }
        for (let i = 0; i < symbols.length - 1; i += 1) {
          const key = `${symbols[i]}\0${symbols[i + 1]}`;
          pairCounts.set(key, (pairCounts.get(key) ?? 0) + count);
        }
      }

      let best = null;
      let bestScore = -1;
      let bestFreq = 0;
      for (const [k, freq] of pairCounts) {
        if (freq < 2) continue;
        const [a, b] = k.split("\0");
        const score = freq / ((singles.get(a) ?? 1) * (singles.get(b) ?? 1));
        if (score > bestScore) {
          bestScore = score;
          best = [a, b];
          bestFreq = freq;
        }
      }
      if (!best) break;

      const [a, b] = best;
      const glued = a + b;
      vocab.add(glued);
      merges.push({ pair: [a, b], merged: glued, score: bestScore, count: bestFreq });
      splits = splits.map(({ word, symbols, count }) => ({
        word,
        symbols: lab.mergePair(symbols, a, b),
        count,
      }));
    }

    return { merges, vocab };
  }

  /**
   * Greedy longest-match. Non-initial pieces get ##.
   * @param {string} word
   * @param {Set<string>} vocab
   * @returns {string[]}
   */
  function encodeWordPieceWord(word, vocab) {
    const pieces = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let found = null;
      while (end > start) {
        const piece = word.slice(start, end);
        if (vocab.has(piece)) {
          found = piece;
          break;
        }
        end -= 1;
      }
      if (!found) return ["[UNK]"];
      pieces.push(start === 0 ? found : `##${found}`);
      start += found.length;
    }
    return pieces;
  }

  /**
   * @param {string} text
   * @param {Set<string>} vocab
   * @returns {string[]}
   */
  function encodeWordPiece(text, vocab) {
    const pieces = [];
    const words = text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
    for (const word of words) {
      pieces.push(...encodeWordPieceWord(word, vocab));
    }
    return pieces;
  }

  /**
   * SentencePiece-style raw string: spaces become ▁, string starts with ▁.
   * @param {string} text
   * @returns {string}
   */
  function toSpRaw(text) {
    const t = text.trim().replace(/\s+/g, MARK);
    return t ? MARK + t : "";
  }

  /**
   * Tiny Unigram: keep characters plus frequent n-grams, score = log(count/total).
   * @param {string} corpusText
   * @param {number} keepN
   * @returns {{scores: Map<string, number>, pieces: [string, number][]}}
   */
  function trainUnigram(corpusText, keepN) {
    const ngramCounts = new Map();
    const lines = corpusText.split(/\n/).map((line) => toSpRaw(line)).filter(Boolean);
    for (const line of lines) {
      const chars = [...line];
      for (let i = 0; i < chars.length; i += 1) {
        ngramCounts.set(chars[i], (ngramCounts.get(chars[i]) ?? 0) + 1);
        for (let n = 2; n <= 8 && i + n <= chars.length; n += 1) {
          const g = chars.slice(i, i + n).join("");
          ngramCounts.set(g, (ngramCounts.get(g) ?? 0) + 1);
        }
      }
    }

    const charPieces = [...ngramCounts.entries()].filter(([p]) => [...p].length === 1);
    const longer = [...ngramCounts.entries()]
      .filter(([p, c]) => [...p].length > 1 && c >= 2)
      .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length);

    const budget = Math.max(charPieces.length, Math.floor(keepN));
    const selected = [...charPieces, ...longer].slice(0, budget);
    const total = selected.reduce((sum, [, c]) => sum + c, 0) || 1;
    const scores = new Map(selected.map(([p, c]) => [p, Math.log(c / total)]));
    return { scores, pieces: selected };
  }

  /**
   * @param {string} text
   * @param {Map<string, number>} scores
   * @returns {{pieces: string[], score: number}}
   */
  function viterbiSegment(text, scores) {
    const chars = [...text];
    const n = chars.length;
    if (n === 0) return { pieces: [], score: 0 };

    const best = Array(n + 1).fill(Number.NEGATIVE_INFINITY);
    const back = Array(n + 1).fill(-1);
    const chosen = Array(n + 1).fill("");
    best[0] = 0;

    for (let i = 0; i < n; i += 1) {
      if (best[i] === Number.NEGATIVE_INFINITY) continue;
      for (let j = i + 1; j <= n; j += 1) {
        const piece = chars.slice(i, j).join("");
        if (!scores.has(piece)) continue;
        const cand = best[i] + scores.get(piece);
        if (cand > best[j]) {
          best[j] = cand;
          back[j] = i;
          chosen[j] = piece;
        }
      }
    }

    if (best[n] === Number.NEGATIVE_INFINITY) {
      return { pieces: chars, score: Number.NEGATIVE_INFINITY };
    }

    const pieces = [];
    let idx = n;
    while (idx > 0) {
      pieces.push(chosen[idx]);
      idx = back[idx];
    }
    pieces.reverse();
    return { pieces, score: best[n] };
  }

  /**
   * BPE on the raw ▁-string (no whitespace pre-tokenization).
   * @param {string} corpusText
   * @param {number} numMerges
   * @returns {{merges: {pair: [string, string], merged: string, count: number}[], items: {symbols: string[], count: number}[]}}
   */
  function trainSpBpe(corpusText, numMerges) {
    const counts = new Map();
    for (const line of corpusText.split(/\n/)) {
      const raw = toSpRaw(line);
      if (!raw) continue;
      counts.set(raw, (counts.get(raw) ?? 0) + 1);
    }

    let items = [...counts.entries()].map(([s, count]) => ({
      symbols: [...s],
      count,
    }));
    const merges = [];
    const steps = Math.max(0, Math.floor(numMerges));

    for (let step = 0; step < steps; step += 1) {
      const pairCounts = new Map();
      for (const { symbols, count } of items) {
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
      items = items.map(({ symbols, count }) => ({
        symbols: lab.mergePair(symbols, a, b),
        count,
      }));
    }

    return { merges, items };
  }

  /**
   * @param {string} text
   * @param {{pair: [string, string]}[]} merges
   * @returns {string[]}
   */
  function encodeSpBpe(text, merges) {
    let symbols = [...toSpRaw(text)];
    for (const { pair } of merges) {
      symbols = lab.mergePair(symbols, pair[0], pair[1]);
    }
    return symbols;
  }

  window.TokenAlgos = {
    MARK,
    trainWordPiece,
    encodeWordPiece,
    toSpRaw,
    trainUnigram,
    viterbiSegment,
    trainSpBpe,
    encodeSpBpe,
  };
})();
