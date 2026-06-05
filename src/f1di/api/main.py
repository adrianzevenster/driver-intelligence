from __future__ import annotations

import logging
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from pydantic import BaseModel

from f1di.config.settings import settings
from f1di.domain.schemas import DriverInsight, InsightAudience, RetrievedEvidence, TelemetryWindow
from f1di.inference.fusion import InferenceOrchestrator
from f1di.observability.logging import configure_logging, log_event
from f1di.observability.metrics import (
    CONFIDENCE_GAUGE,
    HTTP_REQUEST_LATENCY,
    HTTP_REQUESTS_TOTAL,
    INSIGHT_LATENCY,
    INSIGHTS_TOTAL,
    RAG_RESULTS,
    READY_CHECK_TOTAL,
)

try:
    APP_VERSION = version("f1-driver-intelligence")
except PackageNotFoundError:
    APP_VERSION = "0.1.0"

configure_logging(settings.log_level)
logger = logging.getLogger("f1di.api")

if settings.env == "production":
    settings.validate_runtime()

app = FastAPI(title="F1 Driver Intelligence", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FastAPIInstrumentor.instrument_app(app)
orchestrator = InferenceOrchestrator()


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=path,
            status=str(status_code),
        ).inc()
        HTTP_REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration_ms)
        log_event(
            logger,
            logging.INFO,
            "http_request",
            request_id=request_id,
            method=request.method,
            path=path,
            status=status_code,
            latency_ms=round(duration_ms, 3),
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, object]:
    import json as _json
    calibration_quality: object = None
    _q = Path("data/calibration/quality.json")
    if _q.exists():
        try:
            calibration_quality = _json.loads(_q.read_text())
        except Exception:
            pass

    checks: dict[str, object] = {
        "config": settings.runtime_errors(),
        "knowledge_path": Path(settings.knowledge_path).exists(),
        "retriever_documents": len(orchestrator.retriever.documents),
        "vector_backend": settings.vector_backend,
        "calibration_quality": calibration_quality,
    }
    ready_status = (
        not checks["config"]
        and bool(checks["knowledge_path"])
        and int(checks["retriever_documents"]) > 0
    )
    READY_CHECK_TOTAL.labels(status="ready" if ready_status else "not_ready").inc()
    return {"status": "ready" if ready_status else "not_ready", "checks": checks}


@app.get("/version")
def app_version() -> dict[str, str]:
    return {
        "name": "f1-driver-intelligence",
        "version": APP_VERSION,
        "env": settings.env,
        "model_backend": settings.llm_backend,
        "vector_backend": settings.vector_backend,
        "llm_advice_model": settings.llm_advice_model,
        "llm_open_source_model": settings.llm_open_source_model,
    }


@app.post("/v1/insights", response_model=DriverInsight)
def create_insight(
    window: TelemetryWindow,
    audience: InsightAudience = InsightAudience.DRIVER,
) -> DriverInsight:
    insight = orchestrator.analyze(window, audience=audience)
    INSIGHT_LATENCY.observe(insight.latency_ms)
    INSIGHTS_TOTAL.labels(
        risk=insight.risk.value,
        policy=insight.policy,
        audience=audience.value,
    ).inc()
    CONFIDENCE_GAUGE.set(insight.confidence)
    RAG_RESULTS.observe(len(insight.evidence))
    log_event(
        logger,
        logging.INFO,
        "insight_generated",
        session_id=insight.session_id,
        driver_id=insight.driver_id,
        risk=insight.risk.value,
        policy=insight.policy,
        audience=audience.value,
        confidence=round(insight.confidence, 4),
        latency_ms=round(insight.latency_ms, 3),
        evidence_count=len(insight.evidence),
    )
    return insight


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    evidence: list[RetrievedEvidence]
    latency_ms: float


@app.get("/v1/knowledge/status")
def knowledge_status() -> dict:
    total = len(orchestrator.retriever.documents)
    by_source = orchestrator.retriever.source_counts()
    return {
        "documents": total,
        "by_source": by_source,
        "vector_backend": settings.vector_backend,
    }


@app.post("/v1/knowledge/ingest")
def knowledge_ingest(years: str = "", n: int = 8) -> dict:
    from f1di.knowledge.openf1_ingester import ingest
    year_list = [int(y) for y in years.split(",") if y.strip()] or None
    start = time.perf_counter()
    ingested = ingest(orchestrator.retriever, years=year_list, n_per_year=n)
    return {
        "ingested": len(ingested),
        "sessions": ingested,
        "documents_total": len(orchestrator.retriever.documents),
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


@app.post("/v1/knowledge/ingest/fastf1")
def knowledge_ingest_fastf1(years: str = "", n: int = 5, qualifying: bool = True) -> dict:
    from f1di.knowledge.fastf1_ingester import ingest
    year_list = [int(y) for y in years.split(",") if y.strip()] or None
    start = time.perf_counter()
    ingested = ingest(orchestrator.retriever, years=year_list, n_per_year=n, include_qualifying=qualifying)
    return {
        "ingested": len(ingested),
        "sessions": ingested,
        "documents_total": len(orchestrator.retriever.documents),
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


@app.post("/v1/knowledge/ingest/jolpica")
def knowledge_ingest_jolpica(years: str = "", n: int = 8) -> dict:
    from f1di.knowledge.jolpica_ingester import ingest
    year_list = [int(y) for y in years.split(",") if y.strip()] or None
    start = time.perf_counter()
    ingested = ingest(orchestrator.retriever, years=year_list, n_per_year=n)
    return {
        "ingested": len(ingested),
        "sessions": ingested,
        "documents_total": len(orchestrator.retriever.documents),
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


@app.post("/v1/chat", response_model=ChatResponse)
def create_chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    evidence = orchestrator.retriever.search(req.message, top_k=4)
    context_snippets = [f"{e.title}: {e.text[:300]}" for e in evidence]
    history = [{"role": m.role, "content": m.content} for m in req.history]

    from f1di.llm.chat import chat
    response_text = chat(req.message, history, context_snippets)

    if response_text is None:
        response_text = "LLM backend unavailable — set F1DI_LLM_BACKEND to anthropic or openai_compatible."

    return ChatResponse(
        response=response_text,
        evidence=evidence,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


@app.get("/v1/live/sessions")
def live_sessions(year: int = 2024, session_type: str = "Race") -> list[dict]:
    from fastapi import HTTPException
    from f1di.knowledge.openf1_live import OpenF1Blocked, get_sessions
    try:
        return get_sessions(year=year, session_type=session_type)
    except OpenF1Blocked as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/v1/live/drivers/{session_key}")
def live_drivers(session_key: int) -> list[dict]:
    from f1di.knowledge.openf1_live import get_drivers
    return get_drivers(session_key=session_key)


@app.get("/v1/live/laps/{session_key}/{driver_number}")
def live_laps(session_key: int, driver_number: int) -> list[dict]:
    from f1di.knowledge.openf1_live import get_laps
    return get_laps(session_key=session_key, driver_number=driver_number)


@app.post("/v1/live/insight", response_model=DriverInsight)
def live_insight(
    session_key: int,
    driver_number: int,
    audience: InsightAudience = InsightAudience.DRIVER,
    lap_number: int | None = None,
) -> DriverInsight:
    from f1di.knowledge.openf1_live import build_window
    window = build_window(session_key=session_key, driver_number=driver_number, lap_number=lap_number)
    return orchestrator.analyze(window, audience=audience)


@app.get("/v1/session/races")
def session_races(year: int = 2024) -> list[dict]:
    from f1di.knowledge.fastf1_session import get_races
    return get_races(year=year)


@app.get("/v1/session/drivers/{year}/{round_num}")
def session_drivers(year: int, round_num: int) -> list[dict]:
    from f1di.knowledge.fastf1_session import get_drivers
    return get_drivers(year=year, round_num=round_num)


@app.get("/v1/session/laps/{year}/{round_num}/{driver}")
def session_laps(year: int, round_num: int, driver: str) -> list[dict]:
    from f1di.knowledge.fastf1_session import get_laps
    return get_laps(year=year, round_num=round_num, driver=driver)


@app.post("/v1/session/insight", response_model=DriverInsight)
def session_insight(
    year: int,
    round_num: int,
    driver: str,
    audience: InsightAudience = InsightAudience.DRIVER,
    lap_number: int | None = None,
) -> DriverInsight:
    from f1di.knowledge.fastf1_session import build_window
    window = build_window(year=year, round_num=round_num, driver=driver, lap_number=lap_number)
    return orchestrator.analyze(window, audience=audience)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
