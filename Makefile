.PHONY: install lint type test test-cov backfill delta migrate evaluate api frontend-install frontend-dev frontend-build docker-build docker-run terraform-fmt terraform-validate

install:
	poetry install --with dev,evaluation

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

evaluate-case:
	poetry run taxbot-evaluate --case-id $(CASE_ID) --debug

api:
	poetry run uvicorn api.main:app --app-dir src --reload --port 8000

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

docker-build:
	docker build --tag taxbot:local .

docker-run:
	docker run --rm --env-file .env \
		-e TAXBOT_STATIC_DIR=/app/static \
		--publish 8080:8080 taxbot:local

terraform-fmt:
	terraform fmt -recursive infra

terraform-validate:
	terraform -chdir=infra/bootstrap init -backend=false
	terraform -chdir=infra/bootstrap validate
	terraform -chdir=infra/terraform init -backend=false
	terraform -chdir=infra/terraform validate
