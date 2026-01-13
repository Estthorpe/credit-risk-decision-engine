.PHONY: setup lint test ingest train evaluate serve serve-dev monitor docker-build

setup:
	python -m venv .venv

lint:
	ruff check .
	black --check .

test:
	pytest -q

ingest:
	python scripts/ingest.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

# Containerless deployment (your daily workflow)
serve:
	python -m uvicorn credit_risk_decision_engine.serving.app:app --host 0.0.0.0 --port 8000

serve-dev:
	python -m uvicorn credit_risk_decision_engine.serving.app:app --reload --host 0.0.0.0 --port 8000

monitor:
	python scripts/monitor.py

# Keep Docker as a portfolio artifact (you may not run this locally)
docker-build:
	docker build -f infra/docker/Dockerfile -t credit-risk-api:local .
