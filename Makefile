.PHONY: format check-format lint typecheck check

PATH := $(CURDIR)/.venv/bin:$(PATH)
export PATH

format:
	isort .
	black .

check-format:
	isort --check-only .
	black --check .

lint:
	flake8 .

typecheck:
	mypy .

check: check-format lint typecheck