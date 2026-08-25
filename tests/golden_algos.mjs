#!/usr/bin/env node
/**
 * Golden checks for in-browser trainers (Sennrich toy).
 * Run: node tests/golden_algos.mjs
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const sandbox = { console, window: {} };
vm.createContext(sandbox);

for (const name of ["corpora.js", "playgrounds.js", "algorithms.js"]) {
  const src = readFileSync(join(root, "site/js", name), "utf8");
  vm.runInContext(src, sandbox, { filename: name });
}

const { TokenLab: lab, TokenAlgos: algos, TokenCorpora: corpora } = sandbox.window;
if (!lab || !algos || !corpora) {
  console.error("FAIL: TokenLab / TokenAlgos / TokenCorpora missing");
  process.exit(1);
}

const corpus = corpora.SENNRICH;
const n = 8;
let failed = 0;

function check(label, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  console.log(`${ok ? "ok" : "FAIL"}  ${label}`);
  if (!ok) {
    console.log("     got ", got);
    console.log("     want", want);
    failed += 1;
  }
}

const bpe = lab.trainMiniBpe(corpus, 25);
const bpePieces = lab.encodeMiniBpe("lowest", bpe.merges);
check("BPE lowest", bpePieces, ["low", "est</w>"]);

const wp = algos.trainWordPiece(corpus, 25);
const wpPieces = algos.encodeWordPiece("lowest", wp.vocab);
check("WordPiece lowest", wpPieces, ["low", "##est"]);

const raw = algos.toSpRaw("the cat");
check("SP raw mark", raw.startsWith("\u2581") && raw.includes("\u2581"), true);

const uni = algos.trainUnigram(corpus, 32);
const seg = algos.viterbiSegment(algos.toSpRaw("low"), uni.scores);
if (!seg.pieces.length) {
  console.log("FAIL  Unigram empty segmentation");
  failed += 1;
} else {
  console.log("ok  Unigram segments 'low'");
}

const decodedWp = lab.detokenizeWordPiece(["low", "##est"]);
check("detokenize WordPiece", decodedWp, "lowest");

const decodedBpe = lab.detokenizeBpe(["low", "est</w>"]);
check("detokenize BPE", decodedBpe, "lowest");

if (failed) {
  console.error(`\n${failed} failure(s)`);
  process.exit(1);
}
console.log("\nall golden checks passed");
