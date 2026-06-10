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

The gate writes `data/scenarios/regression_report.json` and
`data/scenarios/real_replay_report.json`. It fails if grounding, latency,
warning-confidence, labeled replay recall, hard-negative false positives,
expected agent activation, expected evidence-source retrieval, or policy
correctness regress.

## Labeled Replay Evaluation

The offline replay fixture lives at `data/fixtures/real_replay_eval.json`. Each
case stores a complete `TelemetryWindow`, label rationale, source provenance,
expected risk, expected agent activation, expected evidence sources, and expected
policy for positive cases.

Review fixture labels and observed behavior:

```bash
python scripts/review_replay_fixture.py
python scripts/review_replay_fixture.py --failed-only
```

Default CI uses the stored fixture only. Networked capture is an explicit
operator action because FastF1/OpenF1 availability and cache state vary.

Example FastF1 capture:

```bash
python scripts/capture_replay_fixture.py \
  --provider fastf1 \
  --case-id fastf1_silverstone_ver_lap_10 \
  --case-class nominal \
  --year 2024 \
  --round 12 \
  --driver VER \
  --lap 10 \
  --event "British Grand Prix" \
  --label-rationale "Captured baseline lap with no expected intervention." \
  --label-outcome no_action \
  --expected-max-risk WATCH \
  --expected-source silverstone_track
```

Example OpenF1 capture:

```bash
python scripts/capture_replay_fixture.py \
  --provider openf1 \
  --case-id openf1_silverstone_ver_lap_12 \
  --case-class nominal \
  --year 2024 \
  --session-key 9158 \
  --driver VER \
  --driver-number 1 \
  --lap 12 \
  --event "British Grand Prix" \
  --label-rationale "Captured OpenF1 lap with stable telemetry envelope." \
  --label-outcome no_action \
  --expected-max-risk WATCH \
  --expected-source silverstone_track
```

Labeling standard:

- `label.rationale` must describe the telemetry evidence behind the label.
- `label.outcome` should be the race-engineering action or non-action expected.
- Positive cases must include `expected_min_risk`, `expected_agents`, and
  `expected_policy`.
- Hard negatives and nominal cases must include `expected_max_risk`.
- `expected_sources` should name the circuit knowledge document that should be
  retrieved, such as `spa_ers` or `monaco_weather`.

## Integration Gates

Default tests are offline. To run Qdrant and real FastF1 smoke checks:

```bash
make integration
```

This requires a reachable Qdrant instance, embedding model availability, and
FastF1 network/cache access.

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
