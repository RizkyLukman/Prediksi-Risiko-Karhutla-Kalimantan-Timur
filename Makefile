# Jalankan dari Git Bash (Windows) atau shell POSIX lain.
.DEFAULT_GOAL := help

.PHONY: help setup lint test

help:
	@echo Available targets:
	@echo   setup   Create the virtual environment and install dependencies with uv sync
	@echo   lint    Run ruff check and ruff format --check
	@echo   test    Run pytest

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest
