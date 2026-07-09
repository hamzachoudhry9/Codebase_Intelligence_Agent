# Convenience targets for the benchmark + actor.
# Real model runs require a local Ollama (https://ollama.com) with the model pulled.

PYTHON ?= python3
MODELS ?= llama3.1:8b

.PHONY: help selfcheck bench-oracle bench leaderboard smoke clean-runs

help:
	@echo "make selfcheck     - validate every task (bug fails before, gold resolves after)"
	@echo "make bench-oracle  - run the benchmark with gold patches (harness self-test, expect 100%)"
	@echo "make bench         - run the benchmark against MODELS (default: $(MODELS))"
	@echo "make leaderboard   - build LEADERBOARD.md from runs/*.json"
	@echo "make smoke         - exercise the actor loop with the deterministic FakeLLM (no Ollama)"
	@echo "make clean-runs    - delete runs/*.json"

selfcheck:
	$(PYTHON) -m bench.selfcheck

bench-oracle:
	$(PYTHON) -m bench.run_bench --oracle

bench:
	$(PYTHON) -m bench.run_bench --models "$(MODELS)"

leaderboard:
	$(PYTHON) -m bench.leaderboard

smoke:
	$(PYTHON) -m bench.smoke

clean-runs:
	rm -f bench/runs/*.json
