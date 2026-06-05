# F1 Driver Intelligence

[![CI](https://github.com/adrianzevenster/driver-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/adrianzevenster/driver-intelligence/actions/workflows/ci.yml)

Real-time driver coaching platform for Formula 1. Turns telemetry windows into confidence-calibrated insights for driver and engineer audiences using a multi-agent RAG pipeline grounded in a 24-circuit knowledge base.

---

## Architecture

```
OpenF1 live / synthetic simulator
        │
        ▼
  TelemetryWindow  ──────────────────────────────┐
        │                                         │
        ▼                                         ▼
  Feature extraction              Hybrid RAG retrieval
        │                      (Qdrant dense + BM25 sparse)
        │                      circuit docs · FastF1 · Jolpica
        │                                         │
        ▼                                         │
  ┌─────────────────────────────────────────────┐ │
  │           Specialist agents                  │◄┘
  │  Tire Strategy · Telemetry · Weather · ERS   │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
            Decision fusion layer
          (confidence calibration,
           uncertainty estimation,
           driver-safety policy gate)
                     │
                     ▼
              DriverInsight
          risk · recommendation
          evidence · confidence
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     FastAPI REST          React cockpit UI
   /v1/insights            live + synthetic
   /v1/chat                session replay
   /v1/live/*
```

---

## Quick start (Docker)

```bash
git clone https://github.com/adrianzevenster/driver-intelligence.git
cd driver-intelligence
docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8080 |
| Frontend | run `make frontend` in a second terminal |
| Grafana | http://localhost:3000 |
| Qdrant dashboard | http://localhost:6333/dashboard |

After the API is up, ingest circuit knowledge and historical race data:

```bash
# Static circuit knowledge (24 circuits, instant)
curl -s http://localhost:8080/v1/knowledge/status

# Jolpica race results (2024, last 8 rounds)
curl -X POST "http://localhost:8080/v1/knowledge/ingest/jolpica?years=2024&n=8"

# FastF1 lap analysis + qualifying (2024, last 5 rounds — downloads ~500MB)
curl -X POST "http://localhost:8080/v1/knowledge/ingest/fastf1?years=2024&n=5"
```

---

## Manual setup

**Requirements:** Python 3.11+, Node 18+, [uv](https://docs.astral.sh/uv/), [Qdrant](https://qdrant.tech/documentation/quick-start/), [Ollama](https://ollama.com) (optional)

```bash
# Python environment
uv sync --extra regression --extra dev

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant:v1.9.0

# (Optional) Start Ollama with llama3.1 for LLM recommendations
ollama pull llama3.1
ollama serve

# Copy and edit environment
cp .env.example .env          # edit F1DI_LLM_BACKEND, F1DI_QDRANT_URL as needed

# Start API
make api                       # http://localhost:8080

# Start frontend
make frontend                  # http://localhost:5173
```

---

## LLM backends

| `F1DI_LLM_BACKEND` | Requires | Notes |
|---|---|---|
| `rules` | nothing | Deterministic, works fully offline, fastest |
| `openai_compatible` | Ollama running locally | `llama3.1` recommended; set `F1DI_LLM_BASE_URL` |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude Sonnet; highest quality recommendations |

---

## API reference

```
GET  /health                           liveness probe
GET  /ready                            readiness + knowledge check
GET  /version                          build info

POST /v1/insights                      analyse a TelemetryWindow → DriverInsight
POST /v1/chat                          RAG-grounded free-form chat

GET  /v1/knowledge/status              doc counts by source
POST /v1/knowledge/ingest              OpenF1 session data
POST /v1/knowledge/ingest/fastf1       FastF1 race + qualifying
POST /v1/knowledge/ingest/jolpica      Jolpica race results

GET  /v1/live/sessions                 recent OpenF1 sessions
GET  /v1/live/drivers/{session_key}    drivers in a session
POST /v1/live/insight                  live telemetry → DriverInsight

GET  /metrics                          Prometheus metrics
```

### Example: analyse a synthetic window

```bash
python scripts/generate_synthetic_race.py --out data/scenarios/synthetic_race.jsonl --laps 4

curl -X POST 'http://localhost:8080/v1/insights?audience=DRIVER' \
  -H 'content-type: application/json' \
  --data-binary @<(head -n 1 data/scenarios/synthetic_race.jsonl)
```

### Example: live insight from OpenF1

```bash
# List available sessions
curl http://localhost:8080/v1/live/sessions?year=2024

# Fetch a live or recent insight for Verstappen (driver 1) in session 9158
curl -X POST "http://localhost:8080/v1/live/insight?session_key=9158&driver_number=1"
```

---

## Knowledge sources

The RAG retrieval layer is grounded in three source types:

| Source | Content | Ingest endpoint |
|---|---|---|
| Circuit docs | Track layout, ERS deployment zones, weather patterns — 24 circuits × 3 docs | loaded at startup |
| FastF1 | Compound performance, stint structure, qualifying laps, sector times | `/v1/knowledge/ingest/fastf1` |
| Jolpica | Historical race results, finishing positions, DNFs, fastest laps | `/v1/knowledge/ingest/jolpica` |

---

## Configuration

All settings are set via environment variables (prefix `F1DI_`) or `.env`:

| Variable | Default | Description |
|---|---|---|
| `F1DI_LLM_BACKEND` | `rules` | `rules` · `openai_compatible` · `anthropic` |
| `F1DI_VECTOR_BACKEND` | `memory` | `memory` · `qdrant` |
| `F1DI_QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `F1DI_LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama / OpenAI-compatible URL |
| `F1DI_LLM_OPEN_SOURCE_MODEL` | `llama3.1` | Model name for openai_compatible |
| `F1DI_ANTHROPIC_API_KEY` | _(empty)_ | Required for `anthropic` backend |
| `F1DI_KNOWLEDGE_PATH` | `data/knowledge` | Path to circuit markdown docs |
| `F1DI_LOG_LEVEL` | `INFO` | Structured JSON log level |

---

## Development

```bash
make test          # pytest
make regress       # regression gates + replay
make lint          # ruff check
make simulate      # regenerate synthetic race scenario
make docker-up     # docker compose up --build
```

Regression gates (all must pass on main):
- retrieval grounding presence
- p95 inference latency < 250ms (rules-only path)
- warning/critical confidence discrimination
- deterministic replay stability
