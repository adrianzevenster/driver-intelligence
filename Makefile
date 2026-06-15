.PHONY: sync install api simulate regress integration lint test docker-build docker-up frontend prod-up prod-down prod-logs shadow-eval fit-policy fit-thresholds smoketest fit-tire fit-battery fit-weather fit-meta fit-telemetry fit-safety-car fit-fuel fit-classifiers
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
	python scripts/run_llm_judge_eval.py || echo "[llm-judge] skipped — LLM backend unavailable"
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
shadow-eval:
	python scripts/shadow_eval.py
fit-policy:
	python scripts/fit_policy_thresholds.py
fit-thresholds:
	python scripts/fit_thresholds.py
smoketest:
	python scripts/smoketest_flywheel.py
fit-tire:
	python scripts/fit_tire_classifier.py
fit-battery:
	python scripts/fit_battery_classifier.py
fit-weather:
	python scripts/fit_weather_classifier.py
fit-meta:
	python scripts/fit_meta_learner.py
fit-telemetry:
	python scripts/fit_telemetry_classifier.py
fit-safety-car:
	python scripts/fit_safety_car_classifier.py
fit-fuel:
	python scripts/fit_fuel_classifier.py
fit-classifiers: fit-tire fit-battery fit-weather fit-telemetry fit-safety-car fit-fuel fit-meta
