from __future__ import annotations

import json as _json
import logging
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from f1di.config.settings import settings
from f1di.domain.schemas import (
    DriverInsight,
    InsightAudience,
    RaceProjection,
    RetrievedEvidence,
    RiskLevel,
    StrategyComparison,
    TelemetryWindow,
)
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
    latency_percentiles,
    record_insight_latency,
)

try:
    APP_VERSION = version("f1-driver-intelligence")
except PackageNotFoundError:
    APP_VERSION = "0.1.0"

configure_logging(settings.log_level)
logger = logging.getLogger("f1di.api")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if not settings.api_key_enabled:
        return
    if not key or key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ---------------------------------------------------------------------------
# Lifespan: background scheduler + DB initialisation
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB tables exist.
    try:
        from f1di.storage.database import get_engine
        get_engine()
        logger.info("Storage DB initialised at %s", settings.storage_url)
    except ImportError:
        logger.info("SQLAlchemy not installed — persistence disabled.")
    except Exception as exc:
        logger.warning("Storage DB init failed: %s", exc)

    # Seed drift tracker baseline from existing telemetry.
    try:
        from f1di.observability.drift import get_tracker
        get_tracker().seed_from_db()
    except Exception as exc:
        logger.warning("Drift baseline seed failed: %s", exc)

    # Seed calibration metrics from quality.json so Prometheus sees current ECE on startup.
    try:
        import json as _json
        from f1di.observability.metrics import CALIBRATION_ECE_GAUGE, CALIBRATION_REGRESSION_BLOCKED
        _q = Path("data/calibration/quality.json")
        if _q.exists():
            _qdata = _json.loads(_q.read_text())
            CALIBRATION_ECE_GAUGE.set(_qdata.get("ece") or 0)
            CALIBRATION_REGRESSION_BLOCKED.set(1 if _qdata.get("regression_detected") else 0)
    except Exception as exc:
        logger.debug("Could not seed calibration metrics: %s", exc)

    # Start background ingestion if enabled.
    scheduler = None
    if settings.ingestion_auto_enabled:
        try:
            from f1di.ingestion.scheduler import IngestionScheduler
            years = (
                [int(y) for y in settings.ingestion_years.split(",") if y.strip()]
                if settings.ingestion_years
                else None
            )
            scheduler = IngestionScheduler(
                orchestrator=get_orchestrator(),
                interval_hours=settings.ingestion_interval_hours,
                years=years,
            )
            await scheduler.start()
            logger.info("Background ingestion scheduler started.")
        except Exception as exc:
            logger.warning("Ingestion scheduler failed to start: %s", exc)

    yield

    if scheduler:
        await scheduler.stop()


if settings.env == "production":
    settings.validate_runtime()

app = FastAPI(title="F1 Driver Intelligence", version=APP_VERSION, lifespan=lifespan)


@app.middleware("http")
async def strip_api_prefix(request: Request, call_next):
    if request.scope["path"].startswith("/api/"):
        request.scope["path"] = request.scope["path"][4:]
        request.scope["raw_path"] = request.scope["path"].encode()
    return await call_next(request)


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


@lru_cache(maxsize=1)
def get_orchestrator() -> InferenceOrchestrator:
    return InferenceOrchestrator()


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
        HTTP_REQUESTS_TOTAL.labels(method=request.method, path=path, status=str(status_code)).inc()
        HTTP_REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration_ms)
        log_event(
            logger, logging.INFO, "http_request",
            request_id=request_id, method=request.method, path=path,
            status=status_code, latency_ms=round(duration_ms, 3),
        )


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

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

    orchestrator = get_orchestrator()
    db_ok = False
    try:
        from f1di.storage.database import check_connection
        db_ok = check_connection()
    except ImportError:
        db_ok = None  # type: ignore[assignment]

    checks: dict[str, object] = {
        "config": settings.runtime_errors(),
        "knowledge_path": Path(settings.knowledge_path).exists(),
        "retriever_documents": len(orchestrator.retriever.documents),
        "vector_backend": settings.vector_backend,
        "calibration_quality": calibration_quality,
        "database": db_ok,
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


class SessionCooldown:
    """Suppresses duplicate WARNING/CRITICAL alerts for the same session+driver+risk
    within a configurable lap window to prevent recommendation fatigue.

    State is in-process only — resets on restart, which is acceptable since each
    race weekend is a fresh session.  Not thread-safe across multiple uvicorn workers;
    single-worker deployments (the current default) are unaffected.
    """

    def __init__(self, cooldown_laps: int = 3) -> None:
        self._cooldown_laps = cooldown_laps
        # (session_id, driver_id, risk_value) -> last lap where that risk fired
        self._last_fired: dict[tuple[str, str, str], int] = {}

    def apply(self, insight: DriverInsight, current_lap: int) -> DriverInsight:
        """Return the insight unchanged, or with policy=SUPPRESS if within cooldown."""
        if self._cooldown_laps <= 0:
            return insight
        if insight.risk.value not in ("WARNING", "CRITICAL"):
            return insight
        if insight.policy == "SUPPRESS":
            return insight

        key = (insight.session_id, insight.driver_id, insight.risk.value)
        last = self._last_fired.get(key)
        if last is not None and current_lap - last < self._cooldown_laps:
            logger.debug(
                "cooldown: suppressing %s %s risk=%s (last_lap=%d current=%d)",
                insight.driver_id, insight.session_id, insight.risk.value, last, current_lap,
            )
            return insight.model_copy(update={"policy": "SUPPRESS"})

        self._last_fired[key] = current_lap
        return insight


_cooldown = SessionCooldown(settings.alert_cooldown_laps)


def _run_shadow_v2(production_insight: DriverInsight, window: TelemetryWindow) -> None:
    """Re-score the production insight using the v2 challenger weights and save as shadow."""
    try:
        import uuid as _uuid
        from f1di.confidence.calibration import ChallengerCalibrator
        from f1di.storage.database import db_session
        from f1di.storage.repository import save_insight

        challenger = ChallengerCalibrator()
        v2_conf, v2_unc, _, v2_raw = challenger.calibrate(production_insight.findings)
        shadow = production_insight.model_copy(update={
            "insight_id": str(_uuid.uuid4()),
            "confidence": v2_conf,
            "uncertainty": v2_unc,
            "raw_score": v2_raw,
        })
        with db_session() as session:
            save_insight(session, shadow, window, shadow=True, challenger_version="weights-v2")
        logger.debug(
            "Shadow v2 saved for %s: conf %.3f→%.3f",
            production_insight.insight_id, production_insight.confidence, v2_conf,
        )
    except Exception as exc:
        logger.debug("Shadow v2 pass failed: %s", exc)


def _run_judge_background(insight_id: str, recommendation: str, risk: str, audience: str) -> None:
    """Score a recommendation with the LLM judge and persist the result."""
    try:
        from f1di.evaluation.llm_judge import evaluate_recommendation
        from f1di.storage.database import db_session
        from f1di.storage.repository import save_judge_score

        score = evaluate_recommendation(recommendation, risk=risk, audience=audience)
        if score is None:
            return
        with db_session() as session:
            save_judge_score(
                session,
                insight_id=insight_id,
                safety=score.safety,
                actionability=score.actionability,
                register=score.register,
                calibration=score.calibration,
                mean_score=score.mean,
                rationale=score.rationale,
            )
        logger.debug("Judge scored insight %s: mean=%.3f", insight_id, score.mean)
    except Exception as exc:
        logger.warning("Background judge failed for %s: %s", insight_id, exc)


@app.post("/v1/insights", response_model=DriverInsight)
def create_insight(
    window: TelemetryWindow,
    audience: InsightAudience = InsightAudience.DRIVER,
) -> DriverInsight:
    orchestrator = get_orchestrator()
    insight = orchestrator.analyze(window, audience=audience)

    # Suppress duplicate alerts within the cooldown window.
    insight = _cooldown.apply(insight, window.latest.lap)

    # Persist to DB (non-blocking — failure doesn't affect the response).
    _persist_insight(insight, window)

    # Shadow challenger: re-score with v2 weights in a background thread.
    if settings.shadow_challenger_enabled:
        import threading
        threading.Thread(
            target=_run_shadow_v2, args=(insight, window), daemon=True
        ).start()

    # Push notification only for non-suppressed high-priority insights.
    if insight.policy != "SUPPRESS":
        try:
            from f1di.delivery.notifier import notify_if_configured
            notify_if_configured(insight)
        except Exception as exc:
            logger.debug("Notification skipped: %s", exc)

    INSIGHT_LATENCY.observe(insight.latency_ms)
    record_insight_latency(insight.latency_ms)
    INSIGHTS_TOTAL.labels(risk=insight.risk.value, policy=insight.policy, audience=audience.value).inc()
    CONFIDENCE_GAUGE.set(insight.confidence)
    RAG_RESULTS.observe(len(insight.evidence))
    log_event(
        logger, logging.INFO, "insight_generated",
        session_id=insight.session_id, driver_id=insight.driver_id,
        risk=insight.risk.value, policy=insight.policy, audience=audience.value,
        confidence=round(insight.confidence, 4), latency_ms=round(insight.latency_ms, 3),
        evidence_count=len(insight.evidence),
    )
    return insight


def _persist_insight(insight: DriverInsight, window: TelemetryWindow | None = None) -> None:
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import save_insight, save_telemetry_bulk
        with db_session() as session:
            if window:
                save_telemetry_bulk(session, window)
            save_insight(session, insight, window, shadow=False)
    except ImportError:
        pass  # persistence extra not installed
    except Exception as exc:
        logger.warning("Failed to persist insight %s: %s", insight.insight_id, exc)
        return

    import threading
    t = threading.Thread(
        target=_run_judge_background,
        args=(insight.insight_id, insight.recommendation, insight.risk.value, insight.audience.value),
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# Insight history
# ---------------------------------------------------------------------------

@app.get("/v1/insights/history")
def insight_history(
    driver_id: str | None = None,
    track_id: str | None = None,
    risk: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import get_judge_scores_bulk, list_insights
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")
    with db_session() as session:
        records = list_insights(session, driver_id=driver_id, track_id=track_id, risk=risk, limit=limit, offset=offset)
        judge_means = get_judge_scores_bulk(session, [r.insight_id for r in records])
        return [
            {
                "insight_id": r.insight_id,
                "session_id": r.session_id,
                "driver_id": r.driver_id,
                "track_id": r.track_id,
                "lap": r.lap,
                "compound": r.compound,
                "risk": r.risk,
                "confidence": r.confidence,
                "policy": r.policy,
                "recommendation": r.recommendation,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat(),
                "judge_mean": judge_means.get(r.insight_id),
            }
            for r in records
        ]


@app.get("/v1/insights/{insight_id}/judge")
def get_judge_score(insight_id: str) -> dict[str, Any]:
    """Return the LLM judge score for a single insight, or 404 if not yet scored."""
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import get_judge_score as _get
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")
    with db_session() as session:
        record = _get(session, insight_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Judge score not yet available.")
    return {
        "insight_id": record.insight_id,
        "safety": record.safety,
        "actionability": record.actionability,
        "register": record.register,
        "calibration": record.calibration,
        "mean_score": record.mean_score,
        "rationale": record.rationale,
        "scored_at": record.scored_at.isoformat(),
    }


@app.get("/v1/insights/trend/{driver_id}")
def driver_trend(driver_id: str, days: int = 30) -> dict:
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import driver_trend as _trend
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")
    with db_session() as session:
        return _trend(session, driver_id, days=days)


@app.get("/v1/insights/circuit/{track_id}")
def circuit_heatmap(track_id: str) -> dict:
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import circuit_heatmap as _heatmap
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")
    with db_session() as session:
        return _heatmap(session, track_id)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    insight_id: str
    rating: int
    correct: bool | None = None
    comment: str | None = None
    submitted_by: str | None = None


@app.get("/v1/insights/review-queue")
def insight_review_queue(limit: int = 50) -> list[dict[str, Any]]:
    """Return insights that have no feedback yet, ordered by risk then uncertainty."""
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import review_queue
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")
    with db_session() as session:
        records = review_queue(session, limit=min(limit, 200))
        return [
            {
                "insight_id": r.insight_id,
                "driver_id": r.driver_id,
                "track_id": r.track_id,
                "lap": r.lap,
                "risk": r.risk,
                "confidence": r.confidence,
                "uncertainty": r.uncertainty,
                "recommendation": r.recommendation,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]


@app.get("/v1/ml/judge-correlation")
def judge_correlation() -> dict[str, Any]:
    """Pearson r between LLM judge mean_score and human correct ratings.

    A strong positive r means the judge is a reliable proxy for human judgement
    and its scores can be used to gate calibrator retrains.
    """
    try:
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, JudgeScoreRecord
        from sqlalchemy import select
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")

    with db_session() as session:
        rows = session.execute(
            select(JudgeScoreRecord.mean_score, FeedbackRecord.correct)
            .join(FeedbackRecord, JudgeScoreRecord.insight_id == FeedbackRecord.insight_id)
            .where(FeedbackRecord.correct.isnot(None))
        ).all()

    n = len(rows)
    if n < 3:
        return {"r": None, "n": n, "message": f"Need ≥3 rated+judged insights, have {n}."}

    scores = [r.mean_score for r in rows]
    correct = [1.0 if r.correct else 0.0 for r in rows]
    mean_s = sum(scores) / n
    mean_c = sum(correct) / n
    num = sum((s - mean_s) * (c - mean_c) for s, c in zip(scores, correct))
    den_s = sum((s - mean_s) ** 2 for s in scores) ** 0.5
    den_c = sum((c - mean_c) ** 2 for c in correct) ** 0.5
    r = round(num / (den_s * den_c), 4) if den_s * den_c > 1e-9 else 0.0

    return {
        "r": r,
        "n": n,
        "interpretation": (
            "strong signal" if abs(r) >= 0.5
            else "moderate signal" if abs(r) >= 0.3
            else "weak signal" if abs(r) >= 0.1
            else "no signal — judge scores don't correlate with human correctness"
        ),
        "judge_mean": round(mean_s, 4),
        "human_accuracy": round(mean_c, 4),
    }


@app.get("/v1/ml/precision-degradation")
def precision_degradation() -> list[dict[str, Any]]:
    """Return agents whose recent (7-day) precision has degraded vs the 30-day baseline."""
    from f1di.confidence.online import check_precision_degradation
    return check_precision_degradation()


@app.get("/v1/ml/synthetic-audit")
def synthetic_audit_result() -> dict:
    """Return the most recent synthetic label quality audit, or trigger one if stale/missing."""
    from f1di.evaluation.synthetic_audit import load_last_audit, run_audit
    existing = load_last_audit()
    if existing is None:
        try:
            agents = run_audit()
            existing = load_last_audit() or {"agents": agents}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Audit failed: {exc}")
    return existing


@app.get("/v1/ml/meta-weights")
def meta_learner_weights() -> dict[str, Any]:
    """Return the meta-learner feature importances and model metadata."""
    meta_path = Path("data/calibration/meta_learner.pkl")
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Meta-learner not yet trained — need ≥20 real labels.")
    try:
        from f1di.inference.meta_learner import MetaLearner
        meta = MetaLearner.load(meta_path)
        return {
            "feature_importances": meta.get_feature_importances(),
            "n_train": meta.n_train,
            "n_real": meta.n_real,
            "accuracy": round(meta.accuracy, 4),
            "active_in_inference": meta.n_real >= 20,
            "model_type": getattr(meta, "model_type", "unknown"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/ml/latency")
def insight_latency_percentiles() -> dict[str, Any]:
    """Return p50/p95/p99 insight latency from the rolling window (last 200 requests)."""
    return latency_percentiles()


@app.get("/v1/ml/quality")
def ml_quality() -> dict[str, Any]:
    """Current calibrator quality metrics (ECE, Brier score, fit timestamp)."""
    import json as _json
    q = Path("data/calibration/quality.json")
    if not q.exists():
        raise HTTPException(status_code=404, detail="Calibrator not yet fitted — run scripts/fit_calibrator.py")
    try:
        return _json.loads(q.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read quality.json: {exc}")


@app.post("/v1/feedback")
def submit_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import save_feedback
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")
    with db_session() as session:
        save_feedback(
            session,
            insight_id=req.insight_id,
            rating=req.rating,
            correct=req.correct,
            comment=req.comment,
            submitted_by=req.submitted_by,
        )
    from f1di.agents.auto_retrain import maybe_retrain_all
    background_tasks.add_task(maybe_retrain_all)
    return {"status": "recorded", "insight_id": req.insight_id}


# ---------------------------------------------------------------------------
# Analytics (Text-to-SQL via DuckDB + Ollama)
# ---------------------------------------------------------------------------

class AnalyticsQuery(BaseModel):
    question: str


@app.post("/v1/analytics/query")
def analytics_query(req: AnalyticsQuery) -> dict:
    try:
        from f1di.analytics.sql_agent import SQLAgent
        from f1di.analytics.warehouse import TelemetryWarehouse
        agent = SQLAgent(TelemetryWarehouse())
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Install the analytics extra: pip install 'f1-driver-intelligence[analytics]'",
        )
    return agent.answer(req.question)


@app.get("/v1/analytics/schema")
def analytics_schema() -> dict:
    try:
        from f1di.analytics.warehouse import TelemetryWarehouse
        wh = TelemetryWarehouse()
    except ImportError:
        raise HTTPException(status_code=503, detail="Install the analytics extra.")
    return {"schema": wh.schema_info(), "sample_queries": wh.sample_queries()}


# ---------------------------------------------------------------------------
# Document ingestion (PDF, image, text)
# ---------------------------------------------------------------------------

@app.post("/v1/documents/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    track_id: str = "",
    season: str = "",
    _auth: None = Depends(_require_api_key),
) -> dict:
    try:
        from f1di.ingestion.document_processor import DocumentProcessor
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Install the ocr extra: pip install 'f1-driver-intelligence[ocr]'",
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50 MB guard
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    processor = DocumentProcessor()
    metadata = {}
    if track_id:
        metadata["track_id"] = track_id
    if season:
        metadata["year"] = season

    try:
        docs = processor.process(content, file.filename or "document", metadata=metadata)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Processing failed: {exc}")

    if not docs:
        raise HTTPException(status_code=422, detail="No extractable content found in document.")

    orchestrator = get_orchestrator()
    # Convert to store.KnowledgeDocument (same dataclass shape).
    from f1di.rag.store import KnowledgeDocument as StoreDoc
    store_docs = [StoreDoc(source_id=d.source_id, title=d.title, text=d.text, metadata=d.metadata) for d in docs]
    orchestrator.retriever.add_documents(store_docs)

    # Optionally persist to disk for durability.
    kb_path = Path(settings.knowledge_path)
    if kb_path.exists():
        from f1di.rag.store import save_document_as_markdown
        for doc in store_docs:
            try:
                save_document_as_markdown(doc, kb_path)
            except Exception:
                pass

    return {
        "filename": file.filename,
        "chunks": len(docs),
        "chunks_indexed": len(docs),
        "documents_total": len(orchestrator.retriever.documents),
        "source_ids": [d.source_id for d in docs],
    }


@app.post("/v1/documents/analyse")
async def analyse_document(
    file: UploadFile = File(...),
    track_id: str = "",
    season: str = "",
    _auth: None = Depends(_require_api_key),
) -> dict:
    """Ingest a document into the knowledge base AND return an LLM analysis of its content."""
    start = time.perf_counter()
    try:
        from f1di.ingestion.document_processor import DocumentProcessor
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Install the ocr extra: pip install 'f1-driver-intelligence[ocr]'",
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    processor = DocumentProcessor()
    metadata: dict = {}
    if track_id:
        metadata["track_id"] = track_id
    if season:
        metadata["year"] = season

    try:
        docs = processor.process(content, file.filename or "document", metadata=metadata)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Processing failed: {exc}")

    if not docs:
        raise HTTPException(status_code=422, detail="No extractable content found in document.")

    orchestrator = get_orchestrator()
    from f1di.rag.store import KnowledgeDocument as StoreDoc
    store_docs = [StoreDoc(source_id=d.source_id, title=d.title, text=d.text, metadata=d.metadata) for d in docs]
    orchestrator.retriever.add_documents(store_docs)

    kb_path = Path(settings.knowledge_path)
    if kb_path.exists():
        from f1di.rag.store import save_document_as_markdown
        for doc in store_docs:
            try:
                save_document_as_markdown(doc, kb_path)
            except Exception:
                pass

    # Build combined text for analysis (first ~4000 chars across chunks)
    combined = "\n\n".join(d.text for d in docs)[:4000]
    analysis = _analyse_text(combined, file.filename or "document")

    return {
        "filename": file.filename,
        "chunks_indexed": len(docs),
        "documents_total": len(orchestrator.retriever.documents),
        "source_ids": [d.source_id for d in docs],
        "analysis": analysis,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


_DOC_ANALYSE_SYSTEM = (
    "You are an expert Formula 1 race-engineering analyst. "
    "Analyse the provided document and return ONLY valid JSON with these exact keys:\n"
    '  "summary": string (2-3 sentences),\n'
    '  "key_findings": array of 3-5 strings,\n'
    '  "risk_signal": one of "INFO", "WARNING", "CRITICAL",\n'
    '  "recommended_action": string (one concrete action)\n'
    "Do not add any text outside the JSON object."
)


def _analyse_text(text: str, filename: str) -> dict:
    """Run an LLM structured analysis over extracted document text."""
    import json as _json

    user_msg = f"Document: {filename}\n\n{text}"
    raw: str | None = None

    if settings.llm_backend == "openai_compatible":
        try:
            import httpx
            headers = {"Content-Type": "application/json"}
            if settings.llm_api_key:
                headers["Authorization"] = f"Bearer {settings.llm_api_key}"
            r = httpx.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                json={
                    "model": settings.llm_open_source_model,
                    "messages": [
                        {"role": "system", "content": _DOC_ANALYSE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                headers=headers,
                timeout=settings.llm_timeout_ms / 1000,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("doc_analyse_openai_failed: %s", exc)

    elif settings.llm_backend == "anthropic" and settings.anthropic_api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            resp = client.messages.create(
                model=settings.llm_advice_model,
                max_tokens=512,
                system=_DOC_ANALYSE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = next((b.text for b in resp.content if b.type == "text"), None)
        except Exception as exc:
            logger.warning("doc_analyse_anthropic_failed: %s", exc)

    if raw:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            parsed = _json.loads(cleaned)
            return {
                "summary": str(parsed.get("summary", "")),
                "key_findings": [str(f) for f in parsed.get("key_findings", [])],
                "risk_signal": str(parsed.get("risk_signal", "INFO")),
                "recommended_action": str(parsed.get("recommended_action", "")),
            }
        except Exception:
            pass

    word_count = len(text.split())
    return {
        "summary": (
            f"Document '{filename}' indexed into the knowledge base ({word_count} words extracted). "
            "LLM analysis requires F1DI_LLM_BACKEND=openai_compatible or anthropic."
        ),
        "key_findings": [
            f"{word_count} words extracted across document chunks",
            "Document is now searchable in Chat",
        ],
        "risk_signal": "INFO",
        "recommended_action": "Ask a question in Chat to query the document's content.",
    }


# ---------------------------------------------------------------------------
# Ingestion management
# ---------------------------------------------------------------------------

@app.get("/v1/ingestion/status")
def ingestion_status() -> dict:
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import list_ingestion_runs
        with db_session() as session:
            runs = list_ingestion_runs(session)
        return {
            "total_runs": len(runs),
            "by_source": {
                src: len([r for r in runs if r.source == src])
                for src in {r.source for r in runs}
            },
            "latest": [
                {
                    "source": r.source,
                    "year": r.year,
                    "round_num": r.round_num,
                    "track_id": r.track_id,
                    "documents_added": r.documents_added,
                    "completed_at": r.completed_at.isoformat(),
                }
                for r in runs[:10]
            ],
        }
    except ImportError:
        return {"error": "Persistence not installed — run tracking unavailable."}


@app.post("/v1/ingestion/trigger")
async def trigger_ingestion(
    source: str = "fastf1",
    years: str = "",
    n: int = 5,
    _auth: None = Depends(_require_api_key),
) -> dict:
    """Manually trigger a background ingestion pull."""
    import asyncio

    orchestrator = get_orchestrator()
    year_list = [int(y) for y in years.split(",") if y.strip()] or None

    async def _pull():
        try:
            if source == "fastf1":
                from f1di.knowledge.fastf1_ingester import ingest
            elif source == "openf1":
                from f1di.knowledge.openf1_ingester import ingest
            elif source == "jolpica":
                from f1di.knowledge.jolpica_ingester import ingest
            else:
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: ingest(orchestrator.retriever, years=year_list, n_per_year=n))
        except Exception as exc:
            logger.warning("Triggered ingestion failed: %s", exc)

    asyncio.create_task(_pull())
    return {"status": "ingestion_triggered", "source": source, "years": year_list}


@app.get("/v1/flywheel/status")
def flywheel_status() -> dict:
    """Fast health check for the data flywheel pipeline (no FastF1 network calls)."""
    import json as _json
    import pickle

    # DB
    db_ok = False
    try:
        from f1di.storage.database import check_connection
        db_ok = bool(check_connection())
    except Exception:
        pass

    # Isotonic calibration artifact
    cal_path = Path("data/calibration/isotonic.pkl")
    quality_path = Path("data/calibration/quality.json")
    cal_exists = cal_path.exists()
    ece: float | None = None
    ece_ok = False
    if quality_path.exists():
        try:
            q = _json.loads(quality_path.read_text())
            raw_ece = q.get("ece")
            if isinstance(raw_ece, (int, float)):
                ece = round(float(raw_ece), 4)
                ece_ok = ece <= 0.15
        except Exception:
            pass

    # Outcome-labeled cache
    labeled_path = Path("data/calibration/outcome_labeled.json")
    rounds_labeled = 0
    outcome_cache_exists = labeled_path.exists()
    if outcome_cache_exists:
        try:
            rounds_labeled = len(_json.loads(labeled_path.read_text()))
        except Exception:
            pass

    def _clf_info(pkl_path: Path) -> dict:
        if not pkl_path.exists():
            return {"exists": False, "accuracy": None, "brier_score": None, "n_real": None, "n_train": None, "model_version": None, "model_type": None, "per_class": {}}
        try:
            obj = pickle.loads(pkl_path.read_bytes())
            return {
                "exists": True,
                "accuracy": round(float(obj.accuracy), 4),
                "brier_score": round(float(obj.brier_score), 4) if hasattr(obj, "brier_score") else None,
                "cv_n_splits": getattr(obj, "cv_n_splits", 0),
                "cv_accuracy_std": round(s, 4) if (s := getattr(obj, "cv_accuracy_std", None)) is not None else None,
                "real_sample_weight": round(w, 4) if (w := getattr(obj, "real_sample_weight", None)) is not None else None,
                "prior_cv_accuracy": round(p, 4) if (p := getattr(obj, "prior_cv_accuracy", None)) is not None else None,
                "transfer_lift": round(float(obj.accuracy) - p, 4) if (p := getattr(obj, "prior_cv_accuracy", None)) is not None else None,
                "n_real": int(obj.n_real),
                "n_train": int(obj.n_train),
                "model_version": getattr(obj, "model_version", None),
                "model_type": getattr(obj, "model_type", None),
                "per_class": getattr(obj, "cv_per_class", None) or {},
            }
        except Exception:
            return {"exists": True, "accuracy": None, "brier_score": None, "cv_n_splits": None, "cv_accuracy_std": None, "real_sample_weight": None, "prior_cv_accuracy": None, "transfer_lift": None, "n_real": None, "n_train": None, "model_version": None, "model_type": None, "per_class": {}}

    classifiers = {
        "tire":        _clf_info(Path("data/calibration/tire_classifier.pkl")),
        "battery":     _clf_info(Path("data/calibration/battery_classifier.pkl")),
        "weather":     _clf_info(Path("data/calibration/weather_classifier.pkl")),
        "telemetry":   _clf_info(Path("data/calibration/telemetry_classifier.pkl")),
        "safety_car":  _clf_info(Path("data/calibration/safety_car_classifier.pkl")),
        "fuel":        _clf_info(Path("data/calibration/fuel_classifier.pkl")),
        "meta":        _clf_info(Path("data/calibration/meta_learner.pkl")),
    }
    meta_active = (
        classifiers["meta"]["exists"]
        and classifiers["meta"]["n_real"] is not None
        and classifiers["meta"]["n_real"] >= 20
    )
    classifiers["meta"]["active_in_inference"] = meta_active

    ingestion_enabled = settings.ingestion_auto_enabled
    overall_ok = db_ok and cal_exists and ece_ok and ingestion_enabled

    return {
        "overall_ok": overall_ok,
        "ingestion_enabled": ingestion_enabled,
        "db_ok": db_ok,
        "calibrator_exists": cal_exists,
        "calibrator_ece": ece,
        "ece_ok": ece_ok,
        "rounds_labeled": rounds_labeled,
        "outcome_cache_exists": outcome_cache_exists,
        "shadow_challenger_enabled": settings.shadow_challenger_enabled,
        "alert_cooldown_laps": settings.alert_cooldown_laps,
        "classifiers": classifiers,
        "auto_retrain": _auto_retrain_status(),
    }


def _auto_retrain_status() -> dict:
    try:
        from f1di.agents.auto_retrain import retrain_status
        return retrain_status()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Classifier model management
# ---------------------------------------------------------------------------

_CLASSIFIER_AGENTS = {
    "tire":       Path("data/calibration/tire_classifier.pkl"),
    "battery":    Path("data/calibration/battery_classifier.pkl"),
    "weather":    Path("data/calibration/weather_classifier.pkl"),
    "telemetry":  Path("data/calibration/telemetry_classifier.pkl"),
    "safety_car": Path("data/calibration/safety_car_classifier.pkl"),
    "fuel":       Path("data/calibration/fuel_classifier.pkl"),
    "meta":       Path("data/calibration/meta_learner.pkl"),
}

_HISTORY_PATH = Path("data/calibration/model_history.json")


def _read_classifier_history() -> list[dict]:
    if not _HISTORY_PATH.exists():
        return []
    try:
        return [e for e in _json.loads(_HISTORY_PATH.read_text()) if "agent" in e]
    except Exception:
        return []


@app.get("/v1/model/history")
def model_history(agent: str | None = None, limit: int = 100) -> list[dict]:
    """Return classifier fit history entries from model_history.json."""
    entries = _read_classifier_history()
    if agent:
        entries = [e for e in entries if e.get("agent") == agent]
    return entries[-limit:]


@app.get("/v1/model/snapshots/{agent}")
def model_snapshots(agent: str) -> list[dict]:
    """List versioned snapshot pkls for a given agent with their stored metrics."""
    if agent not in _CLASSIFIER_AGENTS:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent}")
    import hashlib
    cal_dir = Path("data/calibration")
    prefix = "meta_learner_" if agent == "meta" else f"{agent}_classifier_"
    snaps = sorted(cal_dir.glob(f"{prefix}*.pkl"), reverse=True)
    live_path = _CLASSIFIER_AGENTS[agent]
    live_hash = hashlib.md5(live_path.read_bytes()).hexdigest() if live_path.exists() else ""
    result = []
    for p in snaps:
        try:
            raw = p.read_bytes()
            obj = pickle.loads(raw)
            is_live = (hashlib.md5(raw).hexdigest() == live_hash)
            result.append({
                "path": str(p),
                "filename": p.name,
                "fitted_at": p.stem.split("_")[-1],
                "is_live": is_live,
                "accuracy": round(float(obj.accuracy), 4),
                "brier_score": round(float(obj.brier_score), 4) if hasattr(obj, "brier_score") else None,
                "cv_n_splits": getattr(obj, "cv_n_splits", 0),
                "cv_accuracy_std": round(s, 4) if (s := getattr(obj, "cv_accuracy_std", None)) is not None else None,
                "cv_brier_std": round(s, 4) if (s := getattr(obj, "cv_brier_std", None)) is not None else None,
                "cv_fold_accuracies": [round(v, 4) for v in fa] if (fa := getattr(obj, "cv_fold_accuracies", None)) else None,
                "cv_fold_briers": [round(v, 4) for v in fb] if (fb := getattr(obj, "cv_fold_briers", None)) else None,
                "real_sample_weight": round(w, 4) if (w := getattr(obj, "real_sample_weight", None)) is not None else None,
                "prior_cv_accuracy": round(p, 4) if (p := getattr(obj, "prior_cv_accuracy", None)) is not None else None,
                "transfer_lift": round(float(obj.accuracy) - p, 4) if (p := getattr(obj, "prior_cv_accuracy", None)) is not None else None,
                "n_real": int(obj.n_real),
                "n_train": int(obj.n_train),
                "model_version": getattr(obj, "model_version", None),
                "model_type": getattr(obj, "model_type", None),
                "classes": getattr(obj, "classes_", []),
                "cv_per_class": getattr(obj, "cv_per_class", None) or {},
            })
        except Exception:
            result.append({"path": str(p), "filename": p.name, "error": "unreadable"})
    return result


@app.post("/v1/model/test")
def model_test(body: dict) -> dict:
    """Evaluate a snapshot pkl on a fresh synthetic test set and return metrics."""
    agent = body.get("agent", "")
    snapshot_path = body.get("snapshot_path", "")
    if agent not in _CLASSIFIER_AGENTS:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}")
    p = Path(snapshot_path)
    if not p.exists() or not p.is_relative_to(Path("data/calibration")):
        raise HTTPException(status_code=400, detail="Invalid snapshot path")
    try:
        obj = pickle.loads(p.read_bytes())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load snapshot: {exc}")

    # Generate a held-out synthetic test set using each classifier's generator.
    try:
        from sklearn.metrics import accuracy_score
        if agent == "tire":
            from f1di.agents.tire_classifier import generate_synthetic
            X_test, y_test = generate_synthetic(n=300, seed=99)
        elif agent == "battery":
            from f1di.agents.battery_classifier import generate_synthetic
            X_test, y_test = generate_synthetic(n=300, seed=99)
        elif agent == "weather":
            from f1di.agents.weather_classifier import generate_synthetic
            X_test, y_test = generate_synthetic(n=300, seed=99)
        elif agent == "telemetry":
            from f1di.agents.telemetry_classifier import generate_synthetic
            X_test, y_test = generate_synthetic(n=300, seed=99)
        elif agent == "safety_car":
            from f1di.agents.safety_car_classifier import generate_synthetic
            X_test, y_test = generate_synthetic(n=300, seed=99)
        elif agent == "fuel":
            from f1di.agents.fuel_classifier import generate_synthetic
            X_test, y_test = generate_synthetic(n=300, seed=99)
        elif agent == "meta":
            from f1di.inference.meta_learner import generate_synthetic
            X_test, y_test = generate_synthetic(n=300, seed=99)
        else:
            raise HTTPException(status_code=400, detail="No test generator for this agent")

        X_s = obj._scaler.transform(X_test)
        proba = obj._model.predict_proba(X_s)
        preds = obj._model.predict(X_s)
        acc = float(accuracy_score(y_test, preds))

        if agent == "meta":
            # Binary Brier: mean((p_correct - y)^2), matching MetaLearner.fit's own scoring.
            import numpy as _np
            p_correct_idx = int(_np.where(obj._model.classes_ == 1)[0][0])
            brier = float(_np.mean((proba[:, p_correct_idx] - y_test.astype(_np.float64)) ** 2))
        else:
            from f1di.agents.classifier_utils import multiclass_brier as _mb
            brier = float(_mb(proba, y_test, obj._model.classes_))

        from sklearn.metrics import confusion_matrix as _cm
        cm = _cm(y_test, preds).tolist()
        classes_list = [str(c) for c in obj._model.classes_]
        if hasattr(obj, "classes_") and obj.classes_:
            class_labels = obj.classes_  # human-readable e.g. ["INFO","WATCH",...]
        else:
            class_labels = classes_list

        return {
            "agent": agent,
            "snapshot": p.name,
            "model_version": getattr(obj, "model_version", None),
            "model_type": getattr(obj, "model_type", None),
            "test_n": len(y_test),
            "test_accuracy": round(acc, 4),
            "test_brier": round(brier, 4),
            # cv_accuracy/cv_brier come from the k-fold CV done at fit time (see
            # classifier_utils.cross_val_eval) — already a held-out estimate, not a
            # train-set score. test_accuracy above re-checks against a fresh
            # synthetic draw as a second, independent sanity check.
            "cv_accuracy": round(float(obj.accuracy), 4),
            "cv_brier": round(float(obj.brier_score), 4) if hasattr(obj, "brier_score") else None,
            "cv_n_splits": getattr(obj, "cv_n_splits", 0),
            "n_real": int(obj.n_real),
            "confusion_matrix": cm,
            "confusion_labels": class_labels,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Test failed: {exc}")


@app.post("/v1/model/promote")
def model_promote(body: dict) -> dict:
    """Copy a versioned snapshot to the live classifier path."""
    agent = body.get("agent", "")
    snapshot_path = body.get("snapshot_path", "")
    if agent not in _CLASSIFIER_AGENTS:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}")
    src = Path(snapshot_path)
    if not src.exists() or not src.is_relative_to(Path("data/calibration")):
        raise HTTPException(status_code=400, detail="Invalid snapshot path")
    dst = _CLASSIFIER_AGENTS[agent]
    try:
        prev_accuracy: float | None = None
        if dst.exists():
            try:
                prev = pickle.loads(dst.read_bytes())
                prev_accuracy = float(prev.accuracy)
            except Exception:
                pass
        import shutil
        shutil.copy2(src, dst)
        obj = pickle.loads(src.read_bytes())
        logger.info(
            "model_promote: agent=%s snapshot=%s acc=%.4f prev_acc=%s",
            agent, src.name, obj.accuracy, prev_accuracy,
        )
        return {
            "promoted": True,
            "agent": agent,
            "snapshot": src.name,
            "accuracy": round(float(obj.accuracy), 4),
            "prev_accuracy": round(prev_accuracy, 4) if prev_accuracy is not None else None,
            "model_version": getattr(obj, "model_version", None),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Promote failed: {exc}")


@app.get("/v1/model/types")
def model_types() -> dict:
    """Return available model types and their display labels."""
    from f1di.agents.classifier_utils import MODEL_TYPES, _MODEL_DISPLAY
    return {
        "types": MODEL_TYPES,
        "labels": _MODEL_DISPLAY,
        "defaults": {
            "tire": "hgbc", "meta": "hgbc",
            "battery": "logistic", "weather": "logistic",
            "telemetry": "logistic", "safety_car": "logistic", "fuel": "logistic",
        },
    }


@app.post("/v1/model/retrain")
def model_retrain(body: dict) -> dict:
    """Trigger a full fit for one classifier agent and return training metrics."""
    agent = body.get("agent", "")
    model_type: str | None = body.get("model_type") or None
    if agent not in _CLASSIFIER_AGENTS:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}")
    try:
        kwargs = {}
        if agent == "tire":
            from f1di.agents.tire_classifier import train_from_labels, DEFAULT_MODEL_TYPE
        elif agent == "battery":
            from f1di.agents.battery_classifier import train_from_labels, DEFAULT_MODEL_TYPE
        elif agent == "weather":
            from f1di.agents.weather_classifier import train_from_labels, DEFAULT_MODEL_TYPE
        elif agent == "telemetry":
            from f1di.agents.telemetry_classifier import train_from_labels, DEFAULT_MODEL_TYPE
        elif agent == "safety_car":
            from f1di.agents.safety_car_classifier import train_from_labels, DEFAULT_MODEL_TYPE
        elif agent == "fuel":
            from f1di.agents.fuel_classifier import train_from_labels, DEFAULT_MODEL_TYPE
        elif agent == "meta":
            from f1di.inference.meta_learner import train_from_labels, DEFAULT_MODEL_TYPE
        else:
            raise HTTPException(status_code=400, detail=f"No trainer for agent: {agent}")
        kwargs["model_type"] = model_type or DEFAULT_MODEL_TYPE
        result = train_from_labels(**kwargs)
        result["agent"] = agent
        result["model_type_used"] = kwargs["model_type"]
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("model_retrain failed for agent=%s", agent)
        raise HTTPException(status_code=500, detail=f"Retrain failed: {exc}")


@app.post("/v1/model/tune")
def model_tune(body: dict) -> dict:
    """Run Optuna hyperparameter search for one HGBC classifier agent.

    Saves best params to data/calibration/{agent}_best_params.json.
    Subsequent retrains pick them up automatically.
    """
    agent    = body.get("agent", "")
    n_trials = min(int(body.get("n_trials", 30)), 150)
    if agent not in _CLASSIFIER_AGENTS:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}")
    try:
        from f1di.agents.tuner import tune_agent
        return tune_agent(agent, n_trials=n_trials)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("model_tune failed for agent=%s", agent)
        raise HTTPException(status_code=500, detail=f"Tune failed: {exc}")


@app.get("/v1/model/best-params/{agent}")
def model_best_params(agent: str) -> dict:
    """Return saved Optuna best-params for one agent, or {} if not yet tuned."""
    if agent not in _CLASSIFIER_AGENTS:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent}")
    path = Path("data/calibration") / f"{agent}_best_params.json"
    if not path.exists():
        return {"agent": agent, "tuned": False}
    try:
        import json as _json
        data = _json.loads(path.read_text())
        return {"agent": agent, "tuned": True, **data}
    except Exception:
        return {"agent": agent, "tuned": False}


@app.get("/v1/model/feature-importance/{agent}")
def model_feature_importance(agent: str) -> dict:
    """Return permutation-importance scores for the live classifier of an agent."""
    if agent not in _CLASSIFIER_AGENTS:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent}")
    path = _CLASSIFIER_AGENTS[agent]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No trained model for agent: {agent}")
    try:
        obj = pickle.loads(path.read_bytes())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model: {exc}")

    try:
        from sklearn.inspection import permutation_importance as _pi

        if agent == "tire":
            from f1di.agents.tire_classifier import generate_synthetic, FEATURE_NAMES
        elif agent == "battery":
            from f1di.agents.battery_classifier import generate_synthetic, FEATURE_NAMES
        elif agent == "weather":
            from f1di.agents.weather_classifier import generate_synthetic, FEATURE_NAMES
        elif agent == "telemetry":
            from f1di.agents.telemetry_classifier import generate_synthetic, FEATURE_NAMES
        elif agent == "safety_car":
            from f1di.agents.safety_car_classifier import generate_synthetic, FEATURE_NAMES
        elif agent == "fuel":
            from f1di.agents.fuel_classifier import generate_synthetic, FEATURE_NAMES
        elif agent == "meta":
            from f1di.inference.meta_learner import generate_synthetic, FEATURE_NAMES
        else:
            raise HTTPException(status_code=400, detail="No generator for this agent")

        X, y = generate_synthetic(n=400, seed=77)
        X_s = obj._scaler.transform(X)
        result = _pi(obj._model, X_s, y, n_repeats=8, random_state=42, scoring="accuracy")
        importances = result.importances_mean.tolist()
        std = result.importances_std.tolist()
        order = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        return {
            "agent": agent,
            "model_version": getattr(obj, "model_version", None),
            "features": [FEATURE_NAMES[i] for i in order],
            "importances": [round(importances[i], 4) for i in order],
            "importances_std": [round(std[i], 4) for i in order],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {exc}")


# ---------------------------------------------------------------------------
# Shadow / A-B mode
# ---------------------------------------------------------------------------


@app.post("/v1/shadow/analyze", response_model=DriverInsight)
def shadow_analyze(
    window: TelemetryWindow,
    challenger_version: str = "challenger",
    audience: InsightAudience = InsightAudience.DRIVER,
) -> DriverInsight:
    """Run inference in shadow mode: result is stored but not shown to drivers.

    Use this to safely evaluate a new model/config version alongside production
    without affecting live race recommendations.
    """
    orchestrator = get_orchestrator()
    insight = orchestrator.analyze(window, audience=audience)

    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import save_insight
        with db_session() as session:
            save_insight(session, insight, window, shadow=True, challenger_version=challenger_version)
    except Exception as exc:
        logger.warning("Failed to persist shadow insight: %s", exc)

    return insight


@app.get("/v1/shadow/compare")
def shadow_compare(challenger_version: str = "challenger") -> dict[str, Any]:
    """Compare shadow vs. production insight distributions for a challenger version."""
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import shadow_compare as _compare
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")
    with db_session() as session:
        return _compare(session, challenger_version)


@app.get("/v1/shadow/evaluate")
def shadow_evaluate(challenger_version: str = "challenger") -> dict[str, Any]:
    """Statistical evaluation of a shadow challenger vs production.

    Runs a Mann-Whitney U test on confidence distributions and compares
    risk escalation rates. Returns a promote=True/False recommendation.
    A promote recommendation means the challenger shows statistically significant
    improvement (p<0.05) without increased risk escalation.
    """
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import shadow_evaluate as _evaluate
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")
    with db_session() as session:
        return _evaluate(session, challenger_version)


@app.post("/v1/shadow/promote")
def shadow_promote(
    challenger_version: str = "challenger",
    force: bool = False,
    _auth: None = Depends(_require_api_key),
) -> dict[str, Any]:
    """Promote a shadow challenger if evaluation passes.

    Runs shadow_evaluate; if promote=True (or force=True), appends a timestamped
    record to data/calibration/promotions.json and returns the evaluation summary.
    """
    try:
        from f1di.storage.database import db_session
        from f1di.storage.repository import shadow_evaluate as _evaluate
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")

    with db_session() as session:
        evaluation = _evaluate(session, challenger_version)

    if not evaluation.get("promote") and not force:
        return {
            "promoted": False,
            "reason": evaluation.get("recommendation", "evaluation_failed"),
            **evaluation,
        }

    import json as _json
    from datetime import datetime, timezone

    promotions_path = Path("data/calibration/promotions.json")
    promotions_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = _json.loads(promotions_path.read_text()) if promotions_path.exists() else []
    except Exception:
        existing = []

    record: dict[str, Any] = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "challenger_version": challenger_version,
        "forced": force,
        **evaluation,
    }
    existing.append(record)
    promotions_path.write_text(_json.dumps(existing, indent=2))
    logger.info("shadow_promoted challenger=%s forced=%s", challenger_version, force)

    return {"promoted": True, **record}


@app.get("/v1/shadow/promotion-history")
def shadow_promotion_history() -> list[dict[str, Any]]:
    """Return the log of past shadow challenger promotions (manual and auto)."""
    import json as _json
    promotions_path = Path("data/calibration/promotions.json")
    if not promotions_path.exists():
        return []
    try:
        return _json.loads(promotions_path.read_text())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


@app.post("/v1/calibrator/fit-thresholds")
def fit_thresholds_from_telemetry(
    min_rows: int = 30,
    _auth: None = Depends(_require_api_key),
) -> dict:
    """Refit per-circuit agent thresholds from stored telemetry percentiles."""
    try:
        from f1di.storage.database import db_session
        from f1di.storage.models import TelemetrySampleRecord
        import f1di.agents.thresholds as _t
        from f1di.agents.thresholds import CircuitThresholds, save
        from sqlalchemy import select
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")

    with db_session() as session:
        rows = list(session.scalars(select(TelemetrySampleRecord)))

    by_track: dict[str, list] = {}
    for row in rows:
        by_track.setdefault(row.track_id, []).append(row)

    registry: dict[str, CircuitThresholds] = {}
    fitted: list[str] = []
    skipped: list[str] = []

    for track_id, track_rows in sorted(by_track.items()):
        if len(track_rows) < min_rows:
            skipped.append(f"{track_id} ({len(track_rows)} rows, need {min_rows})")
            continue

        fl = sorted(r.tire_wear_fl for r in track_rows)
        soc = sorted(r.battery_soc for r in track_rows)
        def pct(vals: list, p: float) -> float:
            idx = (len(vals) - 1) * p
            lo = int(idx)
            hi = min(lo + 1, len(vals) - 1)
            return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)

        registry[track_id] = CircuitThresholds(
            wear_warning=round(max(0.55, min(0.84, pct(fl, 0.75))), 4),
            wear_critical=round(max(0.70, min(0.95, pct(fl, 0.90))), 4),
            battery_soc_warning=round(max(0.15, min(0.35, pct(soc, 0.10))), 4),
        )
        fitted.append(track_id)

    if registry:
        save(registry)
        _t._LOADED = False
        _t._REGISTRY.clear()

    return {"fitted": fitted, "skipped": skipped, "total_rows": len(rows)}


@app.post("/v1/calibrator/retrain")
def retrain_calibrator(
    min_feedback: int = 20,
    _auth: None = Depends(_require_api_key),
) -> dict:
    """Retrain the isotonic calibrator from human feedback + synthetic base data."""
    try:
        from f1di.confidence.online import retrain
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")

    try:
        result = retrain(min_feedback=min_feedback)
    except Exception as exc:
        logger.exception("Calibrator retrain failed")
        raise HTTPException(status_code=500, detail=f"Calibrator retrain failed: {exc}") from exc

    if not result.get("skipped"):
        try:
            from f1di.observability.metrics import CALIBRATION_ECE_GAUGE, CALIBRATION_REGRESSION_BLOCKED
            CALIBRATION_ECE_GAUGE.set(result.get("ece") or 0)
            CALIBRATION_REGRESSION_BLOCKED.set(1 if result.get("regression_detected") else 0)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------

@app.get("/v1/drift/status")
def drift_status() -> dict:
    """Return current feature drift Z-scores and alert state."""
    from f1di.observability.drift import get_tracker
    return get_tracker().status()


@app.get("/v1/live/performance")
def live_performance() -> dict:
    """Bundle per-agent accuracy, calibration ECE history, feature drift, judge correlation,
    rolling precision series, and reliability diagram data for the live-performance card."""
    import datetime as _datetime
    import json as _json
    from f1di.confidence.online import per_agent_accuracy, reliability_diagram_data, rolling_precision_series
    from f1di.observability.drift import get_tracker

    ece_history: list[dict] = []
    if _QUALITY_HISTORY_PATH.exists():
        try:
            history = _json.loads(_QUALITY_HISTORY_PATH.read_text())
            ece_history = [
                {
                    "recorded_at": h.get("recorded_at"),
                    "ece": h.get("calibration", {}).get("ece"),
                    "brier_score": h.get("calibration", {}).get("brier_score"),
                    "n_feedback": h.get("calibration", {}).get("n_feedback"),
                }
                for h in history[-20:]
                if h.get("calibration", {}).get("ece") is not None
            ]
        except Exception:
            pass

    # Inline judge correlation — avoids an internal HTTP round-trip.
    judge_corr: dict = {"r": None, "n": 0}
    try:
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, JudgeScoreRecord
        from sqlalchemy import select as _select
        with db_session() as _session:
            _rows = _session.execute(
                _select(JudgeScoreRecord.mean_score, FeedbackRecord.correct)
                .join(FeedbackRecord, JudgeScoreRecord.insight_id == FeedbackRecord.insight_id)
                .where(FeedbackRecord.correct.isnot(None))
            ).all()
        _n = len(_rows)
        if _n >= 3:
            _scores = [r.mean_score for r in _rows]
            _correct = [1.0 if r.correct else 0.0 for r in _rows]
            _ms = sum(_scores) / _n
            _mc = sum(_correct) / _n
            _num = sum((s - _ms) * (c - _mc) for s, c in zip(_scores, _correct))
            _ds = sum((s - _ms) ** 2 for s in _scores) ** 0.5
            _dc = sum((c - _mc) ** 2 for c in _correct) ** 0.5
            _r = round(_num / (_ds * _dc), 4) if _ds * _dc > 1e-9 else 0.0
            judge_corr = {
                "r": _r,
                "n": _n,
                "interpretation": (
                    "strong" if abs(_r) >= 0.5
                    else "moderate" if abs(_r) >= 0.3
                    else "weak" if abs(_r) >= 0.1
                    else "no signal"
                ),
            }
        else:
            judge_corr = {"r": None, "n": _n}
    except Exception:
        pass

    since_7d = _datetime.datetime.utcnow() - _datetime.timedelta(days=7)
    from f1di.confidence.online import alert_rate_series, per_driver_precision
    from f1di.evaluation.synthetic_audit import load_last_audit
    return {
        "agent_accuracy": per_agent_accuracy(),
        "agent_accuracy_7d": per_agent_accuracy(since=since_7d),
        "rolling_precision": rolling_precision_series(days=14),
        "ece_history": ece_history,
        "drift": get_tracker().status(),
        "judge_correlation": judge_corr,
        "reliability": reliability_diagram_data(),
        "alert_rate": alert_rate_series(days=30),
        "per_driver_precision": per_driver_precision(since=since_7d),
        "synthetic_audit": load_last_audit(),
        "latency": latency_percentiles(),
    }


# ---------------------------------------------------------------------------
# Feedback stats + calibration health
# ---------------------------------------------------------------------------

@app.get("/v1/feedback/stats")
def feedback_stats() -> dict:
    """Return feedback quality metrics and calibration health for the retrain loop."""
    import json as _json
    stats: dict = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "with_rating": 0,
        "avg_rating": None,
        "min_for_retrain": 20,
        "ready_to_retrain": False,
        "current_ece": None,
        "current_brier": None,
        "last_retrain": None,
        "retrain_dataset": None,
    }

    try:
        from sqlalchemy import func, select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord
        with db_session() as session:
            correct = session.execute(
                select(func.count()).select_from(FeedbackRecord).where(FeedbackRecord.correct == True)  # noqa: E712
            ).scalar_one()
            incorrect = session.execute(
                select(func.count()).select_from(FeedbackRecord).where(FeedbackRecord.correct == False)  # noqa: E712
            ).scalar_one()
            rating_agg = session.execute(
                select(func.count(FeedbackRecord.rating), func.avg(FeedbackRecord.rating))
                .where(FeedbackRecord.rating.isnot(None))
            ).one()
            # Count usable feedback records (same logic as _feedback_pairs: LEFT JOIN,
            # all records with correct or rating are usable).
            paired = session.execute(
                select(func.count())
                .select_from(FeedbackRecord)
                .where(
                    (FeedbackRecord.correct.isnot(None)) | (FeedbackRecord.rating.isnot(None))
                )
            ).scalar_one()
        null_outcome = session.execute(
            select(func.count())
            .select_from(FeedbackRecord)
            .where(FeedbackRecord.submitted_by == "null_outcome")
        ).scalar_one()
        stats["total"] = paired   # show paired count so progress bar matches retrain gate
        stats["correct"] = correct
        stats["incorrect"] = incorrect
        stats["with_rating"] = rating_agg[0] or 0
        stats["avg_rating"] = round(float(rating_agg[1]), 2) if rating_agg[1] else None
        stats["ready_to_retrain"] = paired >= stats["min_for_retrain"]
        stats["null_outcome_labels"] = null_outcome
    except Exception as exc:
        logger.debug("feedback_stats DB query failed: %s", exc)

    quality_path = Path("data/calibration/quality.json")
    if quality_path.exists():
        try:
            q = _json.loads(quality_path.read_text())
            stats["current_ece"] = q.get("ece")
            stats["current_brier"] = q.get("brier_score")
            stats["last_retrain"] = q.get("fitted_at")
            stats["retrain_dataset"] = q.get("calibration_dataset")
            stats["regression_detected"] = q.get("regression_detected", False)
        except Exception:
            pass

    return stats


# ---------------------------------------------------------------------------
# Delivery status
# ---------------------------------------------------------------------------

@app.get("/v1/delivery/status")
def delivery_status() -> dict:
    from f1di.delivery.notifier import get_notifier, get_recipients
    has_telegram = bool(settings.telegram_bot_token and settings.telegram_chat_id)
    has_slack = bool(settings.slack_webhook_url)
    has_email = bool(settings.smtp_username and settings.smtp_password)
    notifier = get_notifier()
    return {
        "email": has_email,
        "email_recipients": get_recipients(),
        "smtp_host": settings.smtp_host,
        "smtp_username": settings.smtp_username,
        "telegram": has_telegram,
        "slack": has_slack,
        "notify_min_risk": notifier.get_min_risk(),
        "any_configured": has_email or has_telegram or has_slack,
    }


@app.post("/v1/delivery/recipients")
def update_recipients(body: dict) -> dict:
    from f1di.delivery.notifier import set_recipients, get_recipients
    recipients = body.get("recipients", [])
    if not isinstance(recipients, list):
        raise HTTPException(status_code=422, detail="recipients must be a list of email strings")
    set_recipients(recipients)
    return {"recipients": get_recipients()}


@app.post("/v1/delivery/min-risk")
def update_min_risk(body: dict) -> dict:
    from f1di.delivery.notifier import get_notifier
    risk = body.get("risk", "WARNING")
    valid = {"INFO", "WATCH", "WARNING", "CRITICAL"}
    if risk not in valid:
        raise HTTPException(status_code=422, detail=f"risk must be one of {sorted(valid)}")
    get_notifier().set_min_risk(risk)
    return {"notify_min_risk": risk}


@app.post("/v1/delivery/test")
async def test_delivery() -> dict:
    from f1di.delivery.notifier import get_notifier
    result = get_notifier().send_test()
    return {"result": result}


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------

@app.post("/v1/predictions/race-projection")
def race_projection(window: TelemetryWindow) -> RaceProjection:
    from f1di.features.extractor import extract_features
    from f1di.simulator.monte_carlo import MonteCarloSimulator

    features = extract_features(window)
    sim = MonteCarloSimulator(iterations=200)
    return sim.project_race(window, features)


@app.post("/v1/predictions/strategy-comparison")
def strategy_comparison(window: TelemetryWindow) -> StrategyComparison:
    from f1di.features.extractor import extract_features
    from f1di.simulator.monte_carlo import MonteCarloSimulator

    features = extract_features(window)
    sim = MonteCarloSimulator(iterations=200)
    return sim.compare_strategies(window, features)


@app.get("/v1/knowledge/status")
def knowledge_status() -> dict:
    orchestrator = get_orchestrator()
    retriever = orchestrator.retriever
    total = len(retriever.documents)
    by_source = retriever.source_counts()

    extra: dict = {}
    if hasattr(retriever, "hot_document_count"):
        extra["hot_documents"] = retriever.hot_document_count
        extra["cold_documents"] = retriever.cold_document_count

    return {
        "documents": total,
        "by_source": by_source,
        "vector_backend": settings.vector_backend,
        **extra,
    }


@app.post("/v1/knowledge/ingest")
def knowledge_ingest(years: str = "", n: int = 8, _auth: None = Depends(_require_api_key)) -> dict:
    from f1di.knowledge.openf1_ingester import ingest
    orchestrator = get_orchestrator()
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
def knowledge_ingest_fastf1(years: str = "", n: int = 5, qualifying: bool = True, _auth: None = Depends(_require_api_key)) -> dict:
    from f1di.knowledge.fastf1_ingester import ingest
    orchestrator = get_orchestrator()
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
def knowledge_ingest_jolpica(years: str = "", n: int = 8, _auth: None = Depends(_require_api_key)) -> dict:
    from f1di.knowledge.jolpica_ingester import ingest
    orchestrator = get_orchestrator()
    year_list = [int(y) for y in years.split(",") if y.strip()] or None
    start = time.perf_counter()
    ingested = ingest(orchestrator.retriever, years=year_list, n_per_year=n)
    return {
        "ingested": len(ingested),
        "sessions": ingested,
        "documents_total": len(orchestrator.retriever.documents),
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

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


@app.post("/v1/chat", response_model=ChatResponse)
def create_chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    orchestrator = get_orchestrator()
    evidence = orchestrator.retriever.search(req.message, top_k=4)
    context_snippets = [f"{e.title}: {e.text[:300]}" for e in evidence]
    history = [{"role": m.role, "content": m.content} for m in req.history]

    from f1di.llm.chat import chat
    response_text = chat(req.message, history, context_snippets)

    if response_text is None:
        response_text = (
            "LLM backend unavailable. Set F1DI_LLM_BACKEND=openai_compatible and "
            "start Ollama (e.g. ollama run llama3.1) to enable chat."
        )

    return ChatResponse(
        response=response_text,
        evidence=evidence,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


# ---------------------------------------------------------------------------
# Live session endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/live/sessions")
def live_sessions(year: int = 2024, session_type: str = "Race") -> list[dict]:
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
    try:
        window = build_window(session_key=session_key, driver_number=driver_number, lap_number=lap_number)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return get_orchestrator().analyze(window, audience=audience)


# ---------------------------------------------------------------------------
# FastF1 session replay
# ---------------------------------------------------------------------------

@app.get("/v1/session/races")
def session_races(year: int = 2024) -> list[dict]:
    from f1di.knowledge.fastf1_session import get_races
    return get_races(year=year)


@app.get("/v1/session/drivers/{year}/{round_num}")
def session_drivers(year: int, round_num: int, session_type: str = "R") -> list[dict]:
    from f1di.knowledge.fastf1_session import get_drivers
    return get_drivers(year=year, round_num=round_num, session_type=session_type)


@app.get("/v1/session/laps/{year}/{round_num}/{driver}")
def session_laps(year: int, round_num: int, driver: str, session_type: str = "R") -> list[dict]:
    from f1di.knowledge.fastf1_session import get_laps
    return get_laps(year=year, round_num=round_num, driver=driver, session_type=session_type)


@app.get("/v1/session/trace/{year}/{round_num}/{driver}/{lap_number}")
def session_trace(
    year: int, round_num: int, driver: str, lap_number: int, session_type: str = "R",
) -> list[dict]:
    from f1di.knowledge.fastf1_session import get_lap_trace
    return get_lap_trace(
        year=year, round_num=round_num, driver=driver, lap_number=lap_number, session_type=session_type,
    )


@app.post("/v1/session/insight", response_model=DriverInsight)
def session_insight(
    year: int,
    round_num: int,
    driver: str,
    audience: InsightAudience = InsightAudience.DRIVER,
    lap_number: int | None = None,
    session_type: str = "R",
) -> DriverInsight:
    from f1di.knowledge.fastf1_session import build_window
    try:
        window = build_window(
            year=year, round_num=round_num, driver=driver, lap_number=lap_number, session_type=session_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Session data error: {exc}")
    try:
        insight = get_orchestrator().analyze(window, audience=audience)
    except Exception as exc:
        logger.exception("session_insight analyze failed driver=%s year=%s round=%s", driver, year, round_num)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    _persist_insight(insight, window)
    return insight


@app.get("/v1/session/strategy/{year}/{round_num}/{driver}")
def session_strategy(year: int, round_num: int, driver: str, session_type: str = "R") -> dict:
    """Replay every lap of a race through the inference pipeline and compare
    the system's calculated risk/pit-window calls against the driver's real
    strategy (FastF1 stints) — a single overview of "what we said" vs "what
    actually happened" for one driver in one race.

    skip_llm=True / record_drift=False on every analyze() call: this is a
    batch historical replay, not live traffic, so it should be fast and must
    not feed the drift tracker's live-traffic baseline.
    """
    from f1di.confidence.calibration import RISK_WEIGHT
    from f1di.knowledge.fastf1_session import actual_strategy, build_all_lap_windows

    try:
        actual = actual_strategy(year=year, round_num=round_num, driver=driver, session_type=session_type)
        windows = build_all_lap_windows(year=year, round_num=round_num, driver=driver, session_type=session_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not windows:
        raise HTTPException(status_code=404, detail=f"No laps found for {driver} in {year} R{round_num}")

    orchestrator = get_orchestrator()
    calculated: list[dict] = []
    for lap in sorted(windows):
        insight = orchestrator.analyze(windows[lap], skip_llm=True, record_drift=False)
        top = next((f for f in insight.findings if f.risk == insight.risk), insight.findings[0])
        tire = next((f for f in insight.findings if f.agent == "tire_strategy"), None)
        calculated.append({
            "lap": lap,
            "risk": insight.risk.value,
            "confidence": round(insight.confidence, 4),
            "top_agent": top.agent,
            "summary": top.summary,
            "tire_risk": tire.risk.value if tire else None,
            "tire_summary": tire.summary if tire else None,
        })

    # Model "pit calls": laps where the tire_strategy agent specifically — not
    # whichever agent happens to be highest risk that lap — first rises to
    # WARNING/CRITICAL after the previous lap was below that level. safety_car
    # and fuel findings can outrank tire risk in `insight.risk` but they're
    # not a pit-timing signal, so deriving pit calls from the overall highest
    # risk would misattribute e.g. a safety-car warning as a tire pit call.
    pit_threshold = RISK_WEIGHT[RiskLevel.WARNING]
    model_pit_calls: list[dict] = []
    prev_above = False
    for row in calculated:
        if row["tire_risk"] is None:
            continue
        above = RISK_WEIGHT[RiskLevel(row["tire_risk"])] >= pit_threshold
        if above and not prev_above:
            model_pit_calls.append({"lap": row["lap"], "risk": row["tire_risk"], "summary": row["tire_summary"]})
        prev_above = above

    return {
        "year": year,
        "round_num": round_num,
        "driver": driver.upper(),
        "session_type": session_type.upper(),
        "actual_strategy": actual,
        "calculated": calculated,
        "model_pit_calls": model_pit_calls,
    }


@app.get("/v1/strategy/undercut/{year}/{round_num}/{driver}/{rival}/{lap_number}")
def strategy_undercut(
    year: int, round_num: int, driver: str, rival: str, lap_number: int,
    session_type: str = "R",
    gap_s: float = 0.0,
    pit_loss_s: float | None = None,
) -> dict:
    """Heuristic undercut-window estimate comparing `driver` and `rival` at a
    given lap. See f1di.strategy.undercut module docstring for the model's
    assumptions and limitations — `model_caveat` in the response is the
    short version.

    gap_s: current on-track time gap from driver to rival in seconds (positive
    means rival is ahead). Defaults to 0.0 (side-by-side). Passing the real
    gap raises the success threshold, making the estimate more realistic.

    pit_loss_s: override the circuit-specific pit-lane time loss in seconds.
    When omitted the calibrated per-circuit value from thresholds.json is used.
    """
    from f1di.strategy.undercut import undercut_window
    try:
        return undercut_window(
            year=year, round_num=round_num, driver=driver, rival=rival,
            lap_number=lap_number, session_type=session_type,
            gap_s=gap_s, pit_loss_s=pit_loss_s,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/v1/strategy/cliff/{year}/{round_num}/{driver}/{lap_number}")
def strategy_cliff(
    year: int, round_num: int, driver: str, lap_number: int,
    session_type: str = "R",
) -> dict:
    """Monte Carlo tire-cliff projection for a single driver at a given lap.

    Returns the median first-crossing lap (eta_laps), a per-lap cumulative
    crossing probability distribution, and the circuit wear threshold used.
    eta_laps is None when fewer than half the simulated trajectories cross
    within the horizon — no confident cliff call yet.

    Cheaper than POST /v1/session/insight: skips RAG retrieval and the full
    agent pipeline, returning only the projection numbers.
    """
    from f1di.agents.thresholds import get as get_thresholds
    from f1di.agents.tire_projection import project_cliff_for_window
    from f1di.features.extractor import extract_features
    from f1di.knowledge.fastf1_session import build_window

    try:
        window = build_window(
            year=year, round_num=round_num, driver=driver,
            lap_number=lap_number, session_type=session_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    features = extract_features(window)
    t = get_thresholds(window.track_id)
    cliff = project_cliff_for_window(window, features, t.wear_critical)

    return {
        "year": year,
        "round_num": round_num,
        "session_type": session_type.upper(),
        "driver": driver.upper(),
        "lap": lap_number,
        "track_id": window.track_id,
        "compound": window.latest.compound.value,
        "wear_critical": t.wear_critical,
        "fl_wear": round(features.fl_wear, 4),
        "fr_wear": round(features.fr_wear, 4),
        "stint_fraction": round(features.stint_fraction, 3),
        "eta_laps": cliff["eta_laps"],
        "probability_by_lap": cliff["probability_by_lap"],
        "horizon_laps": cliff["horizon_laps"],
        "n_sims": cliff["n_sims"],
    }


# ---------------------------------------------------------------------------
# Retrieval evaluation (RAGAS-style)
# ---------------------------------------------------------------------------

@app.get("/v1/eval/retrieval")
def retrieval_eval(save: bool = False) -> dict:
    """Run RAGAS-style retrieval quality evaluation against the gold QA set.

    Returns precision@k, recall@k, MRR, and NDCG@5 aggregated and per-topic.
    Pass ?save=true to persist the report to data/calibration/retrieval_eval.json.
    """
    from f1di.evaluation.retrieval_eval import evaluate_retriever, save_eval_report
    orchestrator = get_orchestrator()
    metrics = evaluate_retriever(orchestrator.retriever)
    if save:
        save_eval_report(metrics)
    return metrics.to_dict()


# ---------------------------------------------------------------------------
# Quality history
# ---------------------------------------------------------------------------

_QUALITY_HISTORY_PATH = Path("data/calibration/quality_history.json")
_RETRIEVAL_EVAL_PATH  = Path("data/calibration/retrieval_eval.json")
_QUALITY_PATH         = Path("data/calibration/quality.json")


@app.post("/v1/quality/record")
def record_quality_snapshot(trigger: str = "manual") -> dict[str, Any]:
    """Append a quality snapshot to quality_history.json.

    Captures calibration ECE/Brier and retrieval P@1/MRR/NDCG from the most
    recent saved reports. Call after each flywheel retrain or eval run.
    """
    import json as _json
    from datetime import datetime, timezone

    snapshot: dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
    }

    if _QUALITY_PATH.exists():
        try:
            q = _json.loads(_QUALITY_PATH.read_text())
            snapshot["calibration"] = {
                "ece": q.get("ece"),
                "brier_score": q.get("brier_score"),
                "n_feedback": q.get("calibration_dataset", {}).get("n_feedback"),
                "fitted_at": q.get("fitted_at"),
            }
        except Exception:
            pass

    if _RETRIEVAL_EVAL_PATH.exists():
        try:
            r = _json.loads(_RETRIEVAL_EVAL_PATH.read_text())
            snapshot["retrieval"] = {
                "precision_at_1": r.get("precision_at_1"),
                "mrr": r.get("mrr"),
                "ndcg_at_5": r.get("ndcg_at_5"),
                "n_queries": r.get("n_queries"),
            }
        except Exception:
            pass

    _QUALITY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = _json.loads(_QUALITY_HISTORY_PATH.read_text()) if _QUALITY_HISTORY_PATH.exists() else []
    except Exception:
        history = []
    history.append(snapshot)
    _QUALITY_HISTORY_PATH.write_text(_json.dumps(history, indent=2))

    logger.info("quality_snapshot_recorded trigger=%s", trigger)
    return snapshot


@app.get("/v1/quality/history")
def get_quality_history(limit: int = 50) -> list[dict]:
    """Return the last `limit` quality snapshots from quality_history.json."""
    import json as _json
    if not _QUALITY_HISTORY_PATH.exists():
        return []
    try:
        history = _json.loads(_QUALITY_HISTORY_PATH.read_text())
        return history[-limit:]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Threshold fitter
# ---------------------------------------------------------------------------

@app.post("/v1/calibrator/fit-thresholds/fastf1")
def fit_thresholds_from_fastf1(
    years: str = "",
    n_per_year: int = 8,
    _auth: None = Depends(_require_api_key),
) -> dict:
    """Fit per-circuit wear/brake thresholds from FastF1 historical stint data.

    This replaces the hand-coded global defaults in thresholds.json with
    circuit-specific estimates derived from empirical pit-stop distributions.
    Results are blended toward the global prior via Bayesian shrinkage.

    Requires FastF1 to be installed and internet access for first run
    (results are cached in /tmp/f1di_fastf1_cache).
    """
    from f1di.agents.threshold_fitter import fit_and_save
    year_list = [int(y.strip()) for y in years.split(",") if y.strip()] or None
    return fit_and_save(years=year_list, n_per_year=n_per_year)


# ---------------------------------------------------------------------------
# Incident dataset builder
# ---------------------------------------------------------------------------

@app.post("/v1/data/build-incident-dataset")
async def build_incident_dataset_endpoint(
    years: str = "",
    n_per_year: int = 6,
    _auth: None = Depends(_require_api_key),
) -> dict:
    """Build a labeled incident dataset from FastF1 historical race data.

    Identifies forced pit stops, retirements, safety cars, and degradation
    cliffs, and labels the preceding laps as high-risk. The dataset is saved
    to data/incidents/labeled_dataset.jsonl and automatically incorporated
    into the calibration retraining pipeline.

    Runs in the background; returns immediately with a confirmation.
    """
    import asyncio
    from f1di.data.incident_dataset import build_dataset

    year_list = [int(y.strip()) for y in years.split(",") if y.strip()] or None

    async def _run():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: build_dataset(years=year_list, n_per_year=n_per_year))

    asyncio.create_task(_run())
    return {"status": "building", "years": year_list, "n_per_year": n_per_year}


# ---------------------------------------------------------------------------
# Race outcome labeling (closes the data flywheel)
# ---------------------------------------------------------------------------

@app.post("/v1/outcomes/label")
def label_race_outcomes(
    year: int,
    round_num: int,
    dry_run: bool = False,
    background_tasks: BackgroundTasks = None,
    key: str | None = Security(_api_key_header),
) -> dict:
    """Label stored insights for one race by comparing against actual outcomes.

    Downloads FastF1 data for the given race, extracts incidents (retirements,
    safety cars, forced pits), then marks each WARNING/CRITICAL insight as
    correct or incorrect based on whether a matching incident occurred within
    the look-ahead window. Labels are written as FeedbackRecord rows.

    Pass ?dry_run=true to compute labels without writing to the database.
    dry_run is unauthenticated (read-only preview); actual writes require auth.
    """
    if not dry_run:
        _require_api_key(key)
    from f1di.data.outcome_labeler import label_race
    from dataclasses import asdict
    try:
        result = label_race(year=year, round_num=round_num, dry_run=dry_run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"outcome_labeler error: {exc}") from exc
    if not dry_run and background_tasks is not None:
        from f1di.agents.auto_retrain import maybe_retrain_all
        background_tasks.add_task(maybe_retrain_all)
    return asdict(result)


@app.get("/v1/outcomes/summary")
def outcome_summary() -> dict:
    """Return a summary of outcome-labeled feedback records."""
    try:
        from sqlalchemy import func, select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord
        with db_session() as session:
            outcome_rows = session.execute(
                select(
                    FeedbackRecord.correct,
                    func.count().label("n"),
                )
                .where(FeedbackRecord.submitted_by == "outcome_labeler")
                .group_by(FeedbackRecord.correct)
            ).all()

        by_label = {str(row.correct): row.n for row in outcome_rows}
        total = sum(by_label.values())
        return {
            "total": total,
            "correct": by_label.get("True", 0),
            "incorrect": by_label.get("False", 0),
            "accuracy": (
                round(by_label.get("True", 0) / total, 3) if total > 0 else None
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/v1/outcomes/predictions")
def outcome_predictions(
    outcome: str | None = None,   # "correct" | "incorrect" | "unlabeled"
    agent: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Recent WARNING/CRITICAL insights joined with their outcome labels."""
    import json as _json
    try:
        from sqlalchemy import select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    with db_session() as session:
        rows = session.execute(
            select(InsightRecord, FeedbackRecord)
            .outerjoin(FeedbackRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
            .where(InsightRecord.risk.in_(["WARNING", "CRITICAL"]))
            .where(InsightRecord.shadow == False)  # noqa: E712
            .order_by(InsightRecord.created_at.desc())
            .limit(limit * 4)
        ).all()

    result = []
    for ins, fb in rows:
        correct: bool | None = None
        if fb is not None:
            if fb.correct is not None:
                correct = fb.correct
            elif fb.rating is not None:
                correct = fb.rating >= 4

        label = "unlabeled" if correct is None else ("correct" if correct else "incorrect")

        if outcome and label != outcome:
            continue

        try:
            findings = _json.loads(ins.findings_json or "[]")
        except Exception:
            findings = []

        agents_present = sorted({f.get("agent") for f in findings if f.get("agent")})

        if agent and agent not in agents_present:
            continue

        result.append({
            "insight_id": ins.insight_id,
            "driver_id": ins.driver_id,
            "track_id": ins.track_id,
            "lap": ins.lap,
            "compound": ins.compound,
            "risk": ins.risk,
            "confidence": round(ins.confidence, 3),
            "created_at": ins.created_at.isoformat(),
            "outcome": label,
            "correct": correct,
            "agents": agents_present,
            "recommendation": (ins.recommendation or "")[:140],
            "findings": [
                {
                    "agent": f.get("agent"),
                    "risk": f.get("risk"),
                    "message": (f.get("message") or "")[:120],
                }
                for f in findings
                if f.get("risk") in ("WARNING", "CRITICAL")
            ],
        })

        if len(result) >= limit:
            break

    return result


# ---------------------------------------------------------------------------
# Regression gates
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parents[3] / "data" / "fixtures"


@app.get("/v1/regression/fixtures")
def regression_fixtures() -> list[str]:
    if not _FIXTURES_DIR.exists():
        return []
    return sorted(p.name for p in _FIXTURES_DIR.glob("*.json"))


@app.post("/v1/regression/run")
def regression_run(fixture: str) -> dict:
    from f1di.regression.real_replay import evaluate_cases, load_cases
    path = (_FIXTURES_DIR / fixture).resolve()
    if not path.is_relative_to(_FIXTURES_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail=f"Fixture {fixture!r} not found")
    start = time.perf_counter()
    report = evaluate_cases(load_cases(path), get_orchestrator())
    report["latency_ms"] = round((time.perf_counter() - start) * 1000)
    return report


@app.post("/v1/regression/capture-from-feedback")
def capture_fixtures_from_feedback(
    max_cases: int = 50,
    _auth: None = Depends(_require_api_key),
) -> dict:
    """Build regression fixtures from insights marked correct=False.

    For each incorrect prediction, looks up the stored telemetry window and
    writes a fixture entry to data/fixtures/feedback_corrections_<date>.json.
    Entries are tagged needs_labeling=true — manually review to set
    expected_min_risk / expected_max_risk before adding to the gate suite.
    """
    try:
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord, TelemetrySampleRecord
        from sqlalchemy import select
    except ImportError:
        raise HTTPException(status_code=503, detail="Persistence layer not installed.")

    with db_session() as session:
        rows = session.execute(
            select(FeedbackRecord, InsightRecord)
            .join(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
            .where(FeedbackRecord.correct == False)  # noqa: E712
            .order_by(InsightRecord.created_at.desc())
            .limit(max_cases)
        ).all()

        cases = []
        for fb, ins in rows:
            telemetry = list(session.scalars(
                select(TelemetrySampleRecord)
                .where(
                    TelemetrySampleRecord.session_id == ins.session_id,
                    TelemetrySampleRecord.driver_id == ins.driver_id,
                )
                .order_by(TelemetrySampleRecord.timestamp_ms)
                .limit(24)
            ))
            if not telemetry:
                continue

            samples = []
            for i, t in enumerate(telemetry):
                braking = t.brake_pressure > 60
                samples.append({
                    "session_id": t.session_id,
                    "driver_id": t.driver_id,
                    "track_id": t.track_id,
                    "timestamp_ms": t.timestamp_ms,
                    "lap": t.lap,
                    "sector": min(3, max(1, 1 + i // (max(len(telemetry), 3) // 3))),
                    "distance_m": 5891.0 * (t.lap - 1) + i * 491.0,
                    "corner_id": f"T{1 + (i % 18)}",
                    "speed_kph": t.speed_kph,
                    "acceleration_g": -0.8 if braking else 0.3,
                    "throttle_pct": t.throttle_pct,
                    "brake_pressure_bar": t.brake_pressure,
                    "steering_angle_deg": 7.0,
                    "yaw_rate_deg_s": 7.0 * t.speed_kph / 190,
                    "slip_angle_deg": 0.3,
                    "wheel_speed_fl": t.speed_kph,
                    "wheel_speed_fr": t.speed_kph,
                    "wheel_speed_rl": t.speed_kph,
                    "wheel_speed_rr": t.speed_kph,
                    "compound": t.compound,
                    "stint_lap": t.stint_lap,
                    "tire_temp_fl_c": 88.0 + t.tire_wear_fl * 30,
                    "tire_temp_fr_c": 86.0 + t.tire_wear_fr * 30,
                    "tire_temp_rl_c": 84.0 + t.tire_wear_rl * 30,
                    "tire_temp_rr_c": 83.0 + t.tire_wear_rr * 30,
                    "tire_wear_fl": t.tire_wear_fl,
                    "tire_wear_fr": t.tire_wear_fr,
                    "tire_wear_rl": t.tire_wear_rl,
                    "tire_wear_rr": t.tire_wear_rr,
                    "grip_estimate": t.grip_estimate,
                    "lockup_event": False,
                    "battery_soc": t.battery_soc,
                    "ers_deploy_kw": 120.0 if t.throttle_pct > 80 else 20.0,
                    "ers_regen_kw": 70.0 if braking else 5.0,
                    "pu_thermal_state": 0.55,
                    "track_temp_c": t.track_temp_c,
                    "ambient_temp_c": 24.0,
                    "humidity_pct": min(100.0, 55.0 + t.rain_intensity * 45),
                    "wind_speed_kph": 14.0,
                    "wind_direction_deg": 245.0,
                    "rain_intensity": t.rain_intensity,
                    "evolving_grip": max(0.4, 0.88 - t.rain_intensity * 0.45),
                    "brake_temp_fl_c": 420.0 + (300.0 if braking else 0.0),
                    "brake_temp_fr_c": 415.0 + (295.0 if braking else 0.0),
                    "brake_temp_rl_c": 310.0,
                    "brake_temp_rr_c": 305.0,
                })

            # Tentative gate: if the model fired WARNING/CRITICAL and human said wrong
            # → likely false positive → guard with expected_max_risk=WATCH.
            # If model said INFO/WATCH → false negative → needs human labeling.
            risk_val = ins.risk
            is_false_positive = risk_val in ("WARNING", "CRITICAL")
            cases.append({
                "case_id": f"feedback_correction_{ins.insight_id[:12]}",
                "class": "false_positive" if is_false_positive else "false_negative",
                "needs_labeling": True,
                "source": {
                    "type": "feedback_correction",
                    "insight_id": ins.insight_id,
                    "original_risk": risk_val,
                    "driver_id": ins.driver_id,
                    "track_id": ins.track_id,
                    "lap": ins.lap,
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "label": {
                    "rationale": f"Human marked this {risk_val} prediction as incorrect.",
                    "outcome": "incorrect_prediction",
                },
                **({"expected_max_risk": "WATCH"} if is_false_positive else {}),
                "window": {
                    "session_id": ins.session_id,
                    "driver_id": ins.driver_id,
                    "track_id": ins.track_id,
                    "samples": samples,
                },
            })

    if not cases:
        return {"captured": 0, "message": "No incorrect predictions with stored telemetry found."}

    import json
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    date_str = time.strftime("%Y%m%d")
    out_path = _FIXTURES_DIR / f"feedback_corrections_{date_str}.json"
    existing: list = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            pass
    existing_ids = {c["case_id"] for c in existing}
    new_cases = [c for c in cases if c["case_id"] not in existing_ids]
    out_path.write_text(json.dumps(existing + new_cases, indent=2))

    return {
        "captured": len(new_cases),
        "skipped_duplicates": len(cases) - len(new_cases),
        "fixture_file": out_path.name,
        "needs_labeling": sum(1 for c in new_cases if c.get("needs_labeling")),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# SPA fallback
# ---------------------------------------------------------------------------

_DIST = Path(__file__).parents[3] / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        return FileResponse(_DIST / "index.html")
