(() => {
  const lab = window.TokenLab;
  const algos = window.TokenAlgos;
  if (!lab || !algos) return;

  const fourText = document.querySelector("#four-text");
  const panes = {
    word: document.querySelector("#chips-word"),
    sub: document.querySelector("#chips-sub"),
    char: document.querySelector("#chips-char"),
    byte: document.querySelector("#chips-byte"),
  };
  const counts = {
    word: document.querySelector("#count-word"),
    sub: document.querySelector("#count-sub"),
    char: document.querySelector("#count-char"),
    byte: document.querySelector("#count-byte"),
  };
  const fourStats = document.querySelector("#four-stats");

  function updateFourWay() {
    const text = fourText?.value ?? "";
    const word = lab.wordTokens(text);
    const sub = lab.naiveSubwords(text, 3);
    const chars = lab.codePoints(text);
    const bytes = lab.utf8Bytes(text).map((b) => lab.byteHex(b));

    lab.renderChips(word, panes.word);
    lab.renderChips(sub, panes.sub);
    lab.renderChips(chars, panes.char);
    lab.renderChips(bytes, panes.byte);

    counts.word.textContent = `${word.length}`;
    counts.sub.textContent = `${sub.length}`;
    counts.char.textContent = `${chars.length}`;
    counts.byte.textContent = `${bytes.length}`;

    const charsCount = chars.length || 1;
    fourStats.innerHTML = `
      <span>chars<strong>${chars.length}</strong></span>
      <span>bytes<strong>${bytes.length}</strong></span>
      <span>b/c<strong>${(bytes.length / charsCount).toFixed(2)}</strong></span>
    `;
  }

  document.querySelectorAll("[data-four-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      fourText.value = btn.getAttribute("data-four-preset") ?? "";
      updateFourWay();
    });
  });
  fourText?.addEventListener("input", updateFourWay);

  const vocabSource = document.querySelector("#oov-vocab");
  const oovProbe = document.querySelector("#oov-probe");
  const oovChips = document.querySelector("#oov-chips");
  const oovNote = document.querySelector("#oov-note");

  function updateOov() {
    const vocab = new Set(
      lab.wordTokens((vocabSource?.value ?? "").toLowerCase()).map((w) => w.toLowerCase()),
    );
    const probe = lab.wordTokens(oovProbe?.value ?? "");
    const rendered = probe.map((tok) => {
      const key = tok.toLowerCase();
      if (/^[\p{L}\p{N}\p{M}]+$/u.test(tok) && !vocab.has(key)) return "[UNK]";
      return tok;
    });
    lab.renderChips(rendered, oovChips, { unk: new Set(["[UNK]"]) });
    const unknown = rendered.filter((t) => t === "[UNK]").length;
    oovNote.textContent =
      unknown === 0
        ? `vocab ${vocab.size} types · every probe token in-vocab`
        : `vocab ${vocab.size} types · ${unknown} [UNK] — never seen in training`;
  }

  vocabSource?.addEventListener("input", updateOov);
  oovProbe?.addEventListener("input", updateOov);

  const inspectText = document.querySelector("#inspect-text");
  const inspectBody = document.querySelector("#inspect-body");
  const inspectNote = document.querySelector("#inspect-note");

  function updateInspect() {
    const text = inspectText?.value ?? "";
    const g = lab.graphemes(text);
    inspectBody.replaceChildren();
    let byteTotal = 0;
    for (const grm of g) {
      const cps = lab.codePoints(grm);
      const bytes = lab.utf8Bytes(grm);
      byteTotal += bytes.length;
      const tr = document.createElement("tr");
      const cpLabel = cps
        .map((c) => `U+${(c.codePointAt(0) ?? 0).toString(16).toUpperCase().padStart(4, "0")}`)
        .join(" ");
      tr.innerHTML = `
        <td>${lab.visible(grm)}</td>
        <td>${cpLabel}</td>
        <td>${cps.length}</td>
        <td>${bytes.map(lab.byteHex).join(" ")}</td>
        <td>${bytes.length}</td>
      `;
      inspectBody.append(tr);
    }
    inspectNote.textContent = `${g.length} grapheme(s) · ${lab.codePoints(text).length} code point(s) · ${byteTotal} UTF-8 byte(s)`;
  }

  document.querySelectorAll("[data-inspect-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      inspectText.value = btn.getAttribute("data-inspect-preset") ?? "";
      updateInspect();
    });
  });
  inspectText?.addEventListener("input", updateInspect);

  const lengthText = document.querySelector("#length-text");
  const lengthBars = document.querySelector("#length-bars");
  const lengthLegend = document.querySelector("#length-legend");

  function updateLength() {
    const text = lengthText?.value ?? "";
    const nWord = lab.wordTokens(text).length;
    const nSub = lab.naiveSubwords(text, 3).length;
    const nChar = lab.codePoints(text).length;
    const nByte = lab.utf8Bytes(text).length;
    const max = Math.max(nWord, nSub, nChar, nByte, 1);
    const rows = [
      ["Word", nWord, "#22d3ee"],
      ["Naive subword", nSub, "#a5b4fc"],
      ["Character", nChar, "#fbbf24"],
      ["Byte", nByte, "#fb7185"],
    ];
    lengthBars.replaceChildren();
    for (const [name, n, color] of rows) {
      const wrap = document.createElement("div");
      wrap.className = "bar-wrap";
      wrap.innerHTML = `
        <div class="pane-head"><h4>${name}</h4><span class="count">${n}</span></div>
        <div class="bar" aria-hidden="true"><i style="width:${(n / max) * 100}%;background:${color}"></i></div>
      `;
      lengthBars.append(wrap);
    }
    lengthLegend.textContent =
      "Finer units never go out of vocabulary, but the sequence the model must read gets longer. Compute and context windows care about that length.";
  }

  lengthText?.addEventListener("input", updateLength);

  const state = {
    algo: "bpe",
    spMode: "unigram",
  };

  const algoCorpus = document.querySelector("#algo-corpus");
  const algoStep = document.querySelector("#algo-step");
  const algoStepValue = document.querySelector("#algo-step-value");
  const algoStepCaption = document.querySelector("#algo-step-caption");
  const algoProbe = document.querySelector("#algo-probe");
  const algoRule = document.querySelector("#algo-rule");
  const algoChips = document.querySelector("#algo-chips");
  const algoCount = document.querySelector("#algo-count");
  const algoLog = document.querySelector("#algo-log");
  const algoMetrics = document.querySelector("#algo-metrics");
  const spModes = document.querySelector("#sp-modes");
  const tabButtons = [...document.querySelectorAll("[data-algo]")];
  const spModeButtons = [...document.querySelectorAll("[data-sp-mode]")];

  const RULES = {
    bpe: "BPE · pick the adjacent pair with the highest corpus frequency, then replay those merges on the probe. End-of-word mark is </w>.",
    wordpiece:
      "WordPiece · pick max score freq(ab)/(freq(a)·freq(b)) among pairs with freq ≥ 2. Encode with greedy longest-match. Continuations get ##.",
    sentencepiece_unigram:
      "SentencePiece Unigram · spaces become ▁. Keep characters plus frequent n-grams, score = log(count). Viterbi finds the best segmentation.",
    sentencepiece_bpe:
      "SentencePiece BPE · no whitespace pre-split. BPE runs on the raw ▁-string so the space is just another character.",
  };

  function setTabs() {
    tabButtons.forEach((btn) => {
      btn.setAttribute("aria-selected", btn.getAttribute("data-algo") === state.algo ? "true" : "false");
    });
    spModeButtons.forEach((btn) => {
      btn.setAttribute("aria-pressed", btn.getAttribute("data-sp-mode") === state.spMode ? "true" : "false");
    });
    spModes?.classList.toggle("hidden", state.algo !== "sentencepiece");
    if (state.algo === "sentencepiece" && state.spMode === "unigram") {
      algoStepCaption.textContent = "Keep top N pieces";
    } else {
      algoStepCaption.textContent = "Merges";
    }
  }

  function updateAlgo() {
    setTabs();
    const n = Number(algoStep?.value ?? 0);
    if (algoStepValue) algoStepValue.textContent = String(n);
    const corpus = algoCorpus?.value ?? "";
    const probe = algoProbe?.value ?? "";
    let pieces = [];
    let log = "";
    let merges = 0;
    let vocab = 0;
    let unk = 0;
    let ruleKey = state.algo;

    if (state.algo === "bpe") {
      const trained = lab.trainMiniBpe(corpus, n);
      merges = trained.merges.length;
      pieces = lab.encodeMiniBpe(probe, trained.merges);
      vocab = new Set(trained.words.flatMap((w) => w.symbols)).size;
      const lines = trained.merges.map(
        (m, i) =>
          `${String(i + 1).padStart(2, "0")}  ${m.pair[0]} + ${m.pair[1]}  →  ${m.merged}   freq ${m.count}`,
      );
      log =
        lines.length === 0
          ? "0 merges. Each word is still characters plus </w>."
          : `${lines.join("\n")}\n\nspellings:\n${trained.words.map((w) => `${w.symbols.join(" ")}  ×${w.count}`).join("\n")}`;
    } else if (state.algo === "wordpiece") {
      const trained = algos.trainWordPiece(corpus, n);
      merges = trained.merges.length;
      vocab = trained.vocab.size;
      pieces = algos.encodeWordPiece(probe, trained.vocab);
      unk = pieces.filter((p) => p === "[UNK]").length;
      const lines = trained.merges.map(
        (m, i) =>
          `${String(i + 1).padStart(2, "0")}  ${m.pair[0]} + ${m.pair[1]}  →  ${m.merged}   score=${m.score.toFixed(4)}  freq ${m.count}`,
      );
      log =
        lines.length === 0
          ? "0 merges. Vocabulary is still single characters. Probe encodes as letters with ##."
          : lines.join("\n");
    } else if (state.spMode === "unigram") {
      ruleKey = "sentencepiece_unigram";
      const keepN = 16 + n * 2;
      if (algoStepValue) algoStepValue.textContent = String(keepN);
      const trained = algos.trainUnigram(corpus, keepN);
      vocab = trained.scores.size;
      const raw = algos.toSpRaw(probe);
      const result = algos.viterbiSegment(raw, trained.scores);
      pieces = result.pieces;
      const top = trained.pieces
        .slice()
        .sort((a, b) => b[1] - a[1])
        .slice(0, 18)
        .map(([p, c]) => `${p}  ×${c}`)
        .join("\n");
      log = `raw probe: ${raw || "(empty)"}\nviterbi score: ${Number.isFinite(result.score) ? result.score.toFixed(3) : "n/a (fell back to characters)"}\n\ntop pieces:\n${top}`;
    } else {
      ruleKey = "sentencepiece_bpe";
      const trained = algos.trainSpBpe(corpus, n);
      merges = trained.merges.length;
      pieces = algos.encodeSpBpe(probe, trained.merges);
      vocab = new Set(trained.items.flatMap((w) => w.symbols)).size;
      const lines = trained.merges.map(
        (m, i) =>
          `${String(i + 1).padStart(2, "0")}  ${m.pair[0]} + ${m.pair[1]}  →  ${m.merged}   freq ${m.count}`,
      );
      log =
        lines.length === 0
          ? `0 merges. Probe is still characters of ${algos.toSpRaw(probe) || "(empty)"}.`
          : `${lines.join("\n")}\n\nline spellings:\n${trained.items.map((w) => `${w.symbols.join(" ")}  ×${w.count}`).join("\n")}`;
    }

    if (algoRule) algoRule.textContent = RULES[ruleKey];
    lab.renderChips(pieces, algoChips, { unk: new Set(["[UNK]"]) });
    if (algoCount) algoCount.textContent = `${pieces.length}`;
    if (algoLog) algoLog.textContent = log;
    if (algoMetrics) {
      algoMetrics.innerHTML = `
        <span>merges<strong>${merges}</strong></span>
        <span>vocab<strong>${vocab}</strong></span>
        <span>pieces<strong>${pieces.length}</strong></span>
        <span>unk<strong>${unk}</strong></span>
      `;
    }
  }

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.algo = btn.getAttribute("data-algo") ?? "bpe";
      updateAlgo();
    });
  });
  spModeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.spMode = btn.getAttribute("data-sp-mode") ?? "unigram";
      updateAlgo();
    });
  });
  algoCorpus?.addEventListener("input", updateAlgo);
  algoStep?.addEventListener("input", updateAlgo);
  algoProbe?.addEventListener("input", updateAlgo);

  const navLinks = [...document.querySelectorAll(".nav a[href^='#']")];
  const sections = navLinks
    .map((a) => document.querySelector(a.getAttribute("href") ?? ""))
    .filter(Boolean);

  function setCurrent() {
    const y = window.scrollY + 90;
    let current = sections[0];
    for (const sec of sections) {
      if (sec.offsetTop <= y) current = sec;
    }
    navLinks.forEach((a) => {
      const on = a.getAttribute("href") === `#${current.id}`;
      if (on) a.setAttribute("aria-current", "true");
      else a.removeAttribute("aria-current");
    });
  }

  window.addEventListener("scroll", setCurrent, { passive: true });

  updateFourWay();
  updateOov();
  updateInspect();
  updateLength();
  updateAlgo();
  setCurrent();
})();
