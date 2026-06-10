.PHONY: sync install api simulate regress integration lint test docker-build docker-up frontend prod-up prod-down prod-logs
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
	python scripts/run_real_replay_gate.py
integration:
	F1DI_INTEGRATION=1 pytest tests/regression/test_integration_modes.py
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
prod-up:
	docker compose -f docker-compose.prod.yml up -d --build
prod-down:
	docker compose -f docker-compose.prod.yml down
prod-logs:
	docker compose -f docker-compose.prod.yml logs -f api
