# Operations

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[regression,dev]"
make simulate
make api
```

The API listens on `http://localhost:8080`.

## Operational Endpoints

- `GET /health`: cheap liveness check.
- `GET /ready`: readiness check for configuration and local knowledge loading.
- `GET /version`: package version and runtime mode.
- `GET /metrics`: Prometheus metrics.

## Runtime Modes

Set `F1DI_ENV` to one of:

- `local`: permissive defaults for development.
- `test`: permissive defaults for tests.
- `production`: fail fast when required production configuration is missing.

Production mode requires non-empty `F1DI_STORAGE_URL` and a supported `F1DI_VECTOR_BACKEND`.

## Regression Gate

```bash
make regress
```

The gate writes `data/scenarios/regression_report.json` and fails if grounding, latency, or warning-confidence thresholds regress.

## Metrics To Watch

- `f1di_insights_total`: generated insights by risk, policy, and audience.
- `f1di_insight_latency_ms`: inference latency.
- `f1di_http_requests_total`: HTTP traffic by method, path, and status.
- `f1di_http_request_latency_ms`: request latency.
- `f1di_rag_results`: evidence count per insight.
- `f1di_ready_check_total`: readiness check outcomes.

## Logs

Logs are JSON-formatted and include `event`, `request_id`, and relevant request or insight fields. In production, ship stdout to the platform log collector and use `request_id` to correlate request, inference, and error events.

## Docker

```bash
docker build -t f1di:local .
docker run --rm -p 8080:8080 f1di:local
```

The container runs as a non-root user and includes a healthcheck against `/health`.
