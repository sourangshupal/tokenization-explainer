# Token Lab diagram tokens

Project source of truth for teaching schematics under `site/diagrams/`.
Do not rely on the Cursor plugin `style-guide.md`.

## Semantic roles

| Role | Hex / value | Use |
|---|---|---|
| `paper` | `#fafafa` | Page + SVG background, label masks |
| `paper-2` | `#f4f4f5` | Alternating layer fills |
| `ink` | `#18181b` | Primary text, primary box stroke |
| `muted` | `#71717a` | Secondary text, default arrows |
| `soft` | `#a1a1aa` | Sublabels, weak boundaries |
| `rule` | `rgba(24,24,27,0.12)` | Hairlines |
| `accent` | `#0891b2` | Focal stroke (≤2 nodes per diagram) |
| `accent-tint` | `rgba(8,145,178,0.10)` | Focal fill |
| `white` | `#ffffff` | Default step fill |

## Typography

| Role | Family |
|---|---|
| Title | Instrument Serif |
| Node names | Geist |
| Technical marks (`</w>`, `##`, `▁`) | Geist Mono |

## Focal rule

Cyan accent on **at most two** elements per diagram. Everything else ink / muted / soft.

## Files

| File | Type |
|---|---|
| `01-pipeline.html` | Flowchart |
| `02-granularity.html` | Layer stack |
| `03-bpe-train.html` | Flowchart |
| `04-bpe-vs-wordpiece.html` | Flowchart |
| `05-detokenize-marks.html` | Architecture |
| `index.html` | Gallery |
