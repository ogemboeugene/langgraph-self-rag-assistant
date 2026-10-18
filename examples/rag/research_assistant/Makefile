.PHONY: install format lint test

TEST ?= tests
PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format src tests examples
	$(PYTHON) -m ruff check --fix src tests examples

lint:
	$(PYTHON) -m ruff check src tests examples

test:
	$(PYTHON) -m pytest $(TEST)
