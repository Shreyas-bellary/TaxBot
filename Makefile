.PHONY: install lint type test test-cov backfill delta migrate evaluate

install:
	poetry install --with dev

lint:
	poetry run ruff check src tests

format:
	poetry run ruff format src tests

type:
	poetry run mypy src

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=src --cov-report=term-missing

migrate:
	poetry run taxbot-migrate

backfill:
	poetry run taxbot-backfill

delta:
	poetry run taxbot-delta

evaluate:
	poetry run taxbot-evaluate
