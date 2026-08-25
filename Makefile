.DEFAULT_GOAL := help

PYTHON := uv run python
PYTEST  := uv run pytest

# ── help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "Production Legal-English BPE Tokenizer — make targets"
	@echo "======================================================="
	@echo ""
	@echo "Setup"
	@echo "  make install          Install / sync all dependencies (uv sync)"
	@echo ""
	@echo "Pipeline (run in order)"
	@echo "  make corpus           Stream + shard FreeLaw corpus  (~30-45 min)"
	@echo "  make train            Train byte-level BPE            (~1-2 h CPU)"
	@echo "  make wrap             Wrap into HF PreTrainedTokenizerFast"
	@echo "  make eval             Run TokEval metrics vs gpt2 + cl100k_base"
	@echo "  make test             Run full pytest suite (112 tests)"
	@echo "  make push REPO=user/legal-bpe-50k   Push to Hugging Face Hub"
	@echo ""
	@echo "Shortcuts"
	@echo "  make all              corpus → train → wrap → eval → test"
	@echo "  make smoke            Quick sanity: imports + CLI help + pytest collect"
	@echo "  make clean            Delete data/corpus/ and models/legal-bpe-50k/"
	@echo ""
	@echo "Options (override on command line)"
	@echo "  MAX_SHARDS=10         Number of corpus shards  (default 10, ~5 GB)"
	@echo "  SHARD_MB=500          Shard size in MB         (default 500)"
	@echo "  VOCAB_SIZE=50257      BPE vocabulary size      (default 50,257)"
	@echo "  MIN_FREQ=2            BPE minimum merge freq   (default 2)"
	@echo "  TEST_DOCS=500         Held-out docs for eval   (default 500)"
	@echo ""

# ── config (override on CLI) ──────────────────────────────────────────────────

MAX_SHARDS  ?= 10
SHARD_MB    ?= 500
VOCAB_SIZE  ?= 50257
MIN_FREQ    ?= 2
TEST_DOCS   ?= 500
REPO        ?= YOUR_USERNAME/legal-bpe-50k

# ── setup ─────────────────────────────────────────────────────────────────────

.PHONY: install
install:
	uv sync

# ── pipeline steps ────────────────────────────────────────────────────────────

.PHONY: corpus
corpus:
	$(PYTHON) scripts/build_corpus.py \
		--max-shards $(MAX_SHARDS) \
		--shard-size-mb $(SHARD_MB)

.PHONY: train
train:
	$(PYTHON) scripts/train_tokenizer.py \
		--vocab-size $(VOCAB_SIZE) \
		--min-freq $(MIN_FREQ)

.PHONY: wrap
wrap:
	$(PYTHON) scripts/wrap_tokenizer.py

.PHONY: eval
eval:
	$(PYTHON) scripts/eval_tokenizer.py \
		--test-docs $(TEST_DOCS)

.PHONY: test
test:
	$(PYTEST) tests/test_tokenizer.py -v

.PHONY: push
push:
	$(PYTHON) scripts/push_hub.py --repo $(REPO)

# ── shortcuts ─────────────────────────────────────────────────────────────────

.PHONY: all
all: corpus train wrap eval test

.PHONY: smoke
smoke:
	@echo "--- import check ---"
	$(PYTHON) -c "\
import scripts.build_corpus; \
import scripts.train_tokenizer; \
import scripts.wrap_tokenizer; \
import scripts.eval_tokenizer; \
import scripts.push_hub; \
print('ALL IMPORTS OK')"
	@echo "--- CLI help ---"
	$(PYTHON) scripts/build_corpus.py --help    > /dev/null
	$(PYTHON) scripts/train_tokenizer.py --help > /dev/null
	$(PYTHON) scripts/wrap_tokenizer.py --help  > /dev/null
	$(PYTHON) scripts/eval_tokenizer.py --help  > /dev/null
	$(PYTHON) scripts/push_hub.py --help        > /dev/null
	@echo "CLI help: OK"
	@echo "--- pytest collect ---"
	$(PYTEST) tests/test_tokenizer.py --collect-only -q
	@echo "smoke: PASSED"

.PHONY: clean
clean:
	rm -rf data/corpus/ models/legal-bpe-50k/
	@echo "Cleaned data/corpus/ and models/legal-bpe-50k/"
