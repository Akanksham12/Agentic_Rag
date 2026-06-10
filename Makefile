# Convenience targets. On Windows without `make`, run the commands directly
# (shown to the right of each target).

.PHONY: install ingest serve test eval lint

install:            ## Install the package + dev tools
	pip install -e ".[dev]"

ingest:             ## Download + ingest the demo corpus
	python scripts/ingest_corpus.py

serve:              ## Run the API with autoreload (http://localhost:8000/docs)
	uvicorn app.main:app --reload

test:               ## Run the test suite
	pytest -q

eval:               ## Run the evaluation harness -> table + eval/results.md
	python -m eval.run_eval

lint:               ## Lint with ruff
	ruff check .
