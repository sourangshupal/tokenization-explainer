(() => {
  const lab = window.TokenLab;
  const algos = window.TokenAlgos;
  const corpora = window.TokenCorpora;
  if (!lab || !algos) return;

  const root = document.documentElement;
  const THEME_KEY = "tokenlab-theme";

  function applyTheme(theme) {
    const next = theme === "light" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    const btn = document.querySelector("#theme-toggle");
    if (btn) {
      btn.textContent = next === "light" ? "Dark" : "Light";
      btn.setAttribute("aria-pressed", next === "light" ? "true" : "false");
    }
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* ignore */
    }
  }

  applyTheme((() => {
    try {
      return localStorage.getItem(THEME_KEY) || "dark";
    } catch {
      return "dark";
    }
  })());
  document.querySelector("#theme-toggle")?.addEventListener("click", () => {
    const cur = root.getAttribute("data-theme") === "light" ? "light" : "dark";
    applyTheme(cur === "light" ? "dark" : "light");
  });

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
  const oovReveal = document.querySelector("#oov-reveal");
  const oovPredict = document.querySelector("#oov-predict");
  let oovHidden = true;

  function computeOov() {
    const vocab = new Set(
      lab.wordTokens((vocabSource?.value ?? "").toLowerCase()).map((w) => w.toLowerCase()),
    );
    const probe = lab.wordTokens(oovProbe?.value ?? "");
    const rendered = probe.map((tok) => {
      const key = tok.toLowerCase();
      if (/^[\p{L}\p{N}\p{M}]+$/u.test(tok) && !vocab.has(key)) return "[UNK]";
      return tok;
    });
    return { vocab, rendered };
  }

  function updateOov() {
    const { vocab, rendered } = computeOov();
    const unknown = rendered.filter((t) => t === "[UNK]").length;
    oovNote.textContent =
      unknown === 0
        ? `vocab ${vocab.size} types · every probe token in-vocab`
        : `vocab ${vocab.size} types · ${unknown} [UNK] — never seen in training`;

    if (oovHidden) {
      oovChips?.classList.add("is-hidden");
      if (oovPredict) {
        oovPredict.hidden = false;
        oovPredict.textContent =
          "Predict first: which probe words become [UNK]? Then reveal.";
      }
      if (oovReveal) oovReveal.hidden = false;
      return;
    }
    oovChips?.classList.remove("is-hidden");
    if (oovPredict) oovPredict.hidden = true;
    if (oovReveal) oovReveal.hidden = true;
    lab.renderChips(rendered, oovChips, { unk: new Set(["[UNK]"]) });
  }

  oovReveal?.addEventListener("click", () => {
    oovHidden = false;
    updateOov();
  });
  function hideOovAgain() {
    oovHidden = true;
    updateOov();
  }
  vocabSource?.addEventListener("input", hideOovAgain);
  oovProbe?.addEventListener("input", hideOovAgain);

  const inspectText = document.querySelector("#inspect-text");
  const inspectBody = document.querySelector("#inspect-body");
  const inspectNote = document.querySelector("#inspect-note");

  function setCell(tr, text) {
    const td = document.createElement("td");
    td.textContent = text;
    tr.append(td);
  }

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
      setCell(tr, lab.visible(grm));
      setCell(tr, cpLabel);
      setCell(tr, String(cps.length));
      setCell(tr, bytes.map(lab.byteHex).join(" "));
      setCell(tr, String(bytes.length));
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
      ["Naive 3-char (not BPE)", nSub, "#a5b4fc"],
      ["Character", nChar, "#fbbf24"],
      ["Byte", nByte, "#fb7185"],
    ];
    lengthBars.replaceChildren();
    for (const [name, n, color] of rows) {
      const wrap = document.createElement("div");
      wrap.className = "bar-wrap";
      const head = document.createElement("div");
      head.className = "pane-head";
      const h4 = document.createElement("h4");
      h4.textContent = name;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = String(n);
      head.append(h4, count);
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.setAttribute("aria-hidden", "true");
      const i = document.createElement("i");
      i.style.width = `${(n / max) * 100}%`;
      i.style.background = color;
      bar.append(i);
      wrap.append(head, bar);
      lengthBars.append(wrap);
    }
    lengthLegend.textContent =
      "Bars use word / naive 3-char / character / byte — not the Algorithm labs trainers. Finer units never go OOV, but the sequence the model must read gets longer.";
  }

  lengthText?.addEventListener("input", updateLength);

  const state = {
    algo: "bpe",
    spMode: "unigram",
    predict: true,
    revealed: false,
  };

  const algoCorpus = document.querySelector("#algo-corpus");
  const algoStep = document.querySelector("#algo-step");
  const algoStepValue = document.querySelector("#algo-step-value");
  const algoStepCaption = document.querySelector("#algo-step-caption");
  const algoStepHint = document.querySelector("#algo-step-hint");
  const algoProbe = document.querySelector("#algo-probe");
  const algoRule = document.querySelector("#algo-rule");
  const algoChips = document.querySelector("#algo-chips");
  const algoCount = document.querySelector("#algo-count");
  const algoLog = document.querySelector("#algo-log");
  const algoMetrics = document.querySelector("#algo-metrics");
  const algoDecode = document.querySelector("#algo-decode");
  const algoReveal = document.querySelector("#algo-reveal");
  const algoPredict = document.querySelector("#algo-predict");
  const predictToggle = document.querySelector("#predict-toggle");
  const spModes = document.querySelector("#sp-modes");
  const tabButtons = [...document.querySelectorAll("[data-algo]")];
  const spModeButtons = [...document.querySelectorAll("[data-sp-mode]")];
  const compareRoot = document.querySelector("#compare-panes");

  const RULES = {
    bpe: "BPE · pick the adjacent pair with the highest corpus frequency, then replay those merges on the probe. End-of-word mark is </w>.",
    wordpiece:
      "WordPiece · pick max score freq(ab)/(freq(a)·freq(b)) among pairs with freq ≥ 2. Encode with greedy longest-match. Continuations get ##.",
    sentencepiece_unigram:
      "SentencePiece Unigram · spaces become ▁. Keep characters plus frequent n-grams, score = log(count). Viterbi finds the best segmentation.",
    sentencepiece_bpe:
      "SentencePiece BPE · no whitespace pre-split. BPE runs on the raw ▁-string so the space is just another character.",
  };

  function syncUrl() {
    const params = new URLSearchParams(location.search);
    params.set("tab", state.algo);
    if (state.algo === "sentencepiece") params.set("sp", state.spMode);
    else params.delete("sp");
    const qs = params.toString();
    const hash = location.hash || "#algo-labs";
    history.replaceState(null, "", `${qs ? `?${qs}` : ""}${hash.startsWith("#") ? hash : "#algo-labs"}`);
  }

  function setTabs() {
    tabButtons.forEach((btn) => {
      btn.setAttribute("aria-selected", btn.getAttribute("data-algo") === state.algo ? "true" : "false");
      btn.tabIndex = btn.getAttribute("data-algo") === state.algo ? 0 : -1;
    });
    spModeButtons.forEach((btn) => {
      btn.setAttribute("aria-pressed", btn.getAttribute("data-sp-mode") === state.spMode ? "true" : "false");
    });
    spModes?.classList.toggle("hidden", state.algo !== "sentencepiece");
    if (state.algo === "sentencepiece" && state.spMode === "unigram") {
      algoStepCaption.textContent = "Keep top N pieces";
      if (algoStepHint) algoStepHint.textContent = "N = 16 + 2 × slider (slider 0–20)";
    } else {
      algoStepCaption.textContent = "Merges";
      if (algoStepHint) algoStepHint.textContent = "Slider 0–20 = number of merge steps";
    }
  }

  function runBpe(corpus, n, probe) {
    const trained = lab.trainMiniBpe(corpus, n);
    const pieces = lab.encodeMiniBpe(probe, trained.merges);
    const vocab = new Set(trained.words.flatMap((w) => w.symbols)).size;
    const lines = trained.merges.map(
      (m, i) =>
        `${String(i + 1).padStart(2, "0")}  ${m.pair[0]} + ${m.pair[1]}  →  ${m.merged}   freq ${m.count}`,
    );
    const log =
      lines.length === 0
        ? "0 merges. Each word is still characters plus </w>."
        : `${lines.join("\n")}\n\nspellings:\n${trained.words.map((w) => `${w.symbols.join(" ")}  ×${w.count}`).join("\n")}`;
    return {
      pieces,
      merges: trained.merges.length,
      vocab,
      unk: 0,
      log,
      decode: lab.detokenizeBpe(pieces),
    };
  }

  function runWordPiece(corpus, n, probe) {
    const trained = algos.trainWordPiece(corpus, n);
    const pieces = algos.encodeWordPiece(probe, trained.vocab);
    const lines = trained.merges.map(
      (m, i) =>
        `${String(i + 1).padStart(2, "0")}  ${m.pair[0]} + ${m.pair[1]}  →  ${m.merged}   score=${m.score.toFixed(4)}  freq ${m.count}`,
    );
    const log =
      lines.length === 0
        ? "0 merges. Vocabulary is still single characters. Probe encodes as letters with ##."
        : lines.join("\n");
    return {
      pieces,
      merges: trained.merges.length,
      vocab: trained.vocab.size,
      unk: pieces.filter((p) => p === "[UNK]").length,
      log,
      decode: lab.detokenizeWordPiece(pieces),
      topPairsHint: trained.merges[0]
        ? `first merge ${trained.merges[0].pair[0]}+${trained.merges[0].pair[1]}`
        : "no merges yet",
    };
  }

  function runSpUnigram(corpus, n, probe) {
    const keepN = 16 + n * 2;
    const trained = algos.trainUnigram(corpus, keepN);
    const raw = algos.toSpRaw(probe);
    const result = algos.viterbiSegment(raw, trained.scores);
    const top = trained.pieces
      .slice()
      .sort((a, b) => b[1] - a[1])
      .slice(0, 18)
      .map(([p, c]) => `${p}  ×${c}`)
      .join("\n");
    return {
      pieces: result.pieces,
      merges: 0,
      vocab: trained.scores.size,
      unk: 0,
      keepN,
      log: `raw probe: ${raw || "(empty)"}\nviterbi score: ${Number.isFinite(result.score) ? result.score.toFixed(3) : "n/a (fell back to characters)"}\n\ntop pieces:\n${top}`,
      decode: lab.detokenizeSentencePiece(result.pieces),
    };
  }

  function runSpBpe(corpus, n, probe) {
    const trained = algos.trainSpBpe(corpus, n);
    const pieces = algos.encodeSpBpe(probe, trained.merges);
    const lines = trained.merges.map(
      (m, i) =>
        `${String(i + 1).padStart(2, "0")}  ${m.pair[0]} + ${m.pair[1]}  →  ${m.merged}   freq ${m.count}`,
    );
    const log =
      lines.length === 0
        ? `0 merges. Probe is still characters of ${algos.toSpRaw(probe) || "(empty)"}.`
        : `${lines.join("\n")}\n\nline spellings:\n${trained.items.map((w) => `${w.symbols.join(" ")}  ×${w.count}`).join("\n")}`;
    return {
      pieces,
      merges: trained.merges.length,
      vocab: new Set(trained.items.flatMap((w) => w.symbols)).size,
      unk: 0,
      log,
      decode: lab.detokenizeSentencePiece(pieces),
    };
  }

  function updateCompare(corpus, n, probe) {
    if (!compareRoot) return;
    const bpe = runBpe(corpus, n, probe);
    const wp = runWordPiece(corpus, n, probe);
    const sp =
      state.spMode === "bpe" ? runSpBpe(corpus, n, probe) : runSpUnigram(corpus, n, probe);
    const rows = [
      ["BPE", bpe],
      ["WordPiece", wp],
      [`SentencePiece (${state.spMode})`, sp],
    ];
    compareRoot.replaceChildren();
    for (const [name, result] of rows) {
      const pane = document.createElement("div");
      pane.className = "pane pane-fixed";
      const head = document.createElement("div");
      head.className = "pane-head";
      const h4 = document.createElement("h4");
      h4.textContent = name;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = String(result.pieces.length);
      head.append(h4, count);
      const chips = document.createElement("div");
      chips.className = "chips";
      lab.renderChips(result.pieces, chips, { unk: new Set(["[UNK]"]) });
      const decode = document.createElement("p");
      decode.className = "decode-line";
      decode.textContent = `decode → ${result.decode || "(empty)"}`;
      pane.append(head, chips, decode);
      compareRoot.append(pane);
    }
  }

  function updateAlgo() {
    setTabs();
    const n = Number(algoStep?.value ?? 0);
    const corpus = algoCorpus?.value ?? "";
    const probe = algoProbe?.value ?? "";
    let result;
    let ruleKey = state.algo;

    if (state.algo === "bpe") {
      result = runBpe(corpus, n, probe);
      if (algoStepValue) algoStepValue.textContent = String(n);
    } else if (state.algo === "wordpiece") {
      result = runWordPiece(corpus, n, probe);
      if (algoStepValue) algoStepValue.textContent = String(n);
    } else if (state.spMode === "unigram") {
      ruleKey = "sentencepiece_unigram";
      result = runSpUnigram(corpus, n, probe);
      if (algoStepValue) algoStepValue.textContent = String(result.keepN);
    } else {
      ruleKey = "sentencepiece_bpe";
      result = runSpBpe(corpus, n, probe);
      if (algoStepValue) algoStepValue.textContent = String(n);
    }

    if (algoRule) algoRule.textContent = RULES[ruleKey];
    if (algoCount) algoCount.textContent = `${result.pieces.length}`;
    if (algoLog) algoLog.textContent = result.log;
    if (algoDecode) algoDecode.textContent = `round-trip decode → ${result.decode || "(empty)"}`;
    if (algoMetrics) {
      algoMetrics.innerHTML = `
        <span>merges<strong>${result.merges}</strong></span>
        <span>vocab<strong>${result.vocab}</strong></span>
        <span>pieces<strong>${result.pieces.length}</strong></span>
        <span>unk<strong>${result.unk}</strong></span>
      `;
    }

    const needsPredict =
      state.predict &&
      state.algo === "wordpiece" &&
      /\blowest\b/i.test(probe);

    if (needsPredict && !state.revealed) {
      algoChips?.classList.add("is-hidden");
      if (algoPredict) {
        algoPredict.hidden = false;
        algoPredict.innerHTML =
          "Predict: how does WordPiece encode <code>lowest</code> on the Sennrich toy set?<br>" +
          "A) <code>lowest</code> &nbsp; B) <code>low</code> + <code>##est</code> &nbsp; C) <code>l</code> + <code>##owest</code>";
      }
      if (algoReveal) algoReveal.hidden = false;
    } else if (state.predict && !state.revealed) {
      algoChips?.classList.add("is-hidden");
      if (algoPredict) {
        algoPredict.hidden = false;
        algoPredict.textContent = "Predict the pieces, then reveal.";
      }
      if (algoReveal) algoReveal.hidden = false;
    } else {
      algoChips?.classList.remove("is-hidden");
      if (algoPredict) algoPredict.hidden = true;
      if (algoReveal) algoReveal.hidden = true;
      lab.renderChips(result.pieces, algoChips, { unk: new Set(["[UNK]"]) });
    }

    if (!state.predict || state.revealed) {
      updateCompare(corpus, n, probe);
      compareRoot?.classList.remove("is-hidden");
    } else {
      compareRoot?.classList.add("is-hidden");
      compareRoot?.replaceChildren();
    }
  }

  function selectAlgo(algo) {
    state.algo = algo;
    state.revealed = false;
    syncUrl();
    updateAlgo();
  }

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      selectAlgo(btn.getAttribute("data-algo") ?? "bpe");
    });
  });

  document.querySelector(".tabs")?.addEventListener("keydown", (ev) => {
    const order = ["bpe", "wordpiece", "sentencepiece"];
    const i = order.indexOf(state.algo);
    if (i < 0) return;
    if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
      ev.preventDefault();
      const next =
        ev.key === "ArrowRight"
          ? order[(i + 1) % order.length]
          : order[(i - 1 + order.length) % order.length];
      selectAlgo(next);
      document.querySelector(`[data-algo="${next}"]`)?.focus();
    }
  });

  spModeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.spMode = btn.getAttribute("data-sp-mode") ?? "unigram";
      state.revealed = false;
      syncUrl();
      updateAlgo();
    });
  });

  algoReveal?.addEventListener("click", () => {
    state.revealed = true;
    updateAlgo();
  });
  predictToggle?.addEventListener("click", () => {
    state.predict = !state.predict;
    state.revealed = !state.predict;
    predictToggle.setAttribute("aria-pressed", state.predict ? "true" : "false");
    predictToggle.textContent = state.predict ? "Predict on" : "Predict off";
    updateAlgo();
  });

  function dirtyAlgo() {
    state.revealed = false;
    updateAlgo();
  }
  algoCorpus?.addEventListener("input", dirtyAlgo);
  algoStep?.addEventListener("input", dirtyAlgo);
  algoProbe?.addEventListener("input", dirtyAlgo);

  document.querySelectorAll("[data-corpus]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const kind = btn.getAttribute("data-corpus");
      if (!corpora) return;
      algoCorpus.value = kind === "course" ? corpora.COURSE : corpora.SENNRICH;
      document.querySelectorAll("[data-corpus]").forEach((b) => {
        b.setAttribute("aria-pressed", b === btn ? "true" : "false");
      });
      dirtyAlgo();
    });
  });

  document.querySelectorAll("[data-quiz]").forEach((form) => {
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const answer = form.querySelector("input[name='q']:checked");
      const out = form.querySelector(".quiz-feedback");
      if (!out) return;
      if (!answer) {
        out.textContent = "Pick an option.";
        return;
      }
      const ok = answer.value === form.getAttribute("data-correct");
      out.textContent = ok
        ? form.getAttribute("data-ok") || "Correct."
        : form.getAttribute("data-bad") || "Not quite — try again.";
      out.dataset.tone = ok ? "ok" : "bad";
    });
  });

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

  const params = new URLSearchParams(location.search);
  const tab = params.get("tab");
  const sp = params.get("sp");
  if (tab === "bpe" || tab === "wordpiece" || tab === "sentencepiece") state.algo = tab;
  if (sp === "unigram" || sp === "bpe") state.spMode = sp;
  if (location.hash.startsWith("#algo-labs")) {
    requestAnimationFrame(() => document.querySelector("#algo-labs")?.scrollIntoView());
  }

  if (algoCorpus && corpora && !algoCorpus.value.trim()) {
    algoCorpus.value = corpora.SENNRICH;
  }

  updateFourWay();
  updateOov();
  updateInspect();
  updateLength();
  updateAlgo();
  setCurrent();
})();
