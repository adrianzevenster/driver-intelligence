.PHONY: sync install api simulate regress lint test docker-build docker-up frontend
sync:
	uv sync --extra regression --extra dev
install:
	python -m pip install -e ".[regression,dev]"
api:
	uvicorn f1di.api.main:app --app-dir src --reload --host 0.0.0.0 --port 8080
simulate:
	python scripts/generate_synthetic_race.py --out data/scenarios/synthetic_race.jsonl --laps 12
regress:
	pytest tests/regression
	python scripts/run_replay_regression.py
lint:
	ruff check .
test:
	pytest
frontend:
	cd frontend && npm install && npm run dev
docker-build:
	docker build -t f1di:local .

docker-up:
	docker compose up --build
