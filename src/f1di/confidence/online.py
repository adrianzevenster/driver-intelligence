from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("f1di.confidence.online")

from f1di.agents.classifier_utils import _CALIBRATION_DIR
_CALIBRATOR_PATH = _CALIBRATION_DIR / "isotonic.pkl"
_QUALITY_PATH = _CALIBRATION_DIR / "quality.json"
_HISTORY_PATH = _CALIBRATION_DIR / "model_history.json"


def _file_op(action: str, path: Path, fn: Callable[[], object]) -> object:
    try:
        return fn()
    except OSError as exc:
        raise RuntimeError(f"{action} {path}: {exc}") from exc


def _promote_live_model(versioned_path: Path, live_path: Path) -> None:
    tmp_path = live_path.with_name(f".{live_path.name}.tmp")

    def _copy_and_replace() -> None:
        shutil.copyfile(versioned_path, tmp_path)
        tmp_path.replace(live_path)

    try:
        _file_op("promote live calibrator", live_path, _copy_and_replace)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


_INFERENCE_AGENTS = frozenset({
    "telemetry", "tire_strategy", "weather", "battery", "safety_car", "fuel",
})


def per_agent_accuracy(since: _dt.datetime | None = None) -> dict[str, dict]:
    """Per-agent precision computed from labeled WARNING/CRITICAL insights in the DB.

    For each agent, 'precision' is the fraction of WARNING/CRITICAL findings
    that were confirmed correct by human feedback or outcome labels.
    All six inference agents are always included; agents with no labeled data
    at WARNING/CRITICAL level get precision=None so the UI denominator stays
    stable at 6 rather than collapsing to however many happened to fire.
    """
    import json as _json
    try:
        from sqlalchemy import select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except Exception:
        return {}

    agent_stats: dict[str, dict[str, int]] = {}
    try:
        with db_session() as session:
            stmt = (
                select(FeedbackRecord, InsightRecord)
                .outerjoin(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
                .where(InsightRecord.risk.in_(["WARNING", "CRITICAL"]))
            )
            if since is not None:
                stmt = stmt.where(InsightRecord.created_at >= since)
            for fb, ins in session.execute(stmt).all():
                if ins is None:
                    continue
                if fb.correct is not None:
                    is_correct = fb.correct
                elif fb.rating is not None:
                    is_correct = fb.rating >= 4
                else:
                    continue
                try:
                    findings = _json.loads(ins.findings_json or "[]")
                except Exception:
                    continue
                for finding in findings:
                    if finding.get("risk") not in ("WARNING", "CRITICAL"):
                        continue
                    agent = finding.get("agent", "unknown")
                    stats = agent_stats.setdefault(agent, {"n_correct": 0, "n_total": 0})
                    stats["n_total"] += 1
                    if is_correct:
                        stats["n_correct"] += 1
    except Exception as exc:
        logger.warning("per_agent_accuracy query failed: %s", exc)
        return {}

    result = {
        agent: {
            "precision": round(s["n_correct"] / s["n_total"], 4) if s["n_total"] > 0 else None,
            "n_correct": s["n_correct"],
            "n_total": s["n_total"],
        }
        for agent, s in sorted(agent_stats.items())
    }
    # Pad with null entries for any inference agent not yet seen at WARNING+ level
    for agent in sorted(_INFERENCE_AGENTS):
        if agent not in result:
            result[agent] = {"precision": None, "n_correct": 0, "n_total": 0}
    return result


def rolling_precision_series(days: int = 14) -> list[dict]:
    """Per-agent precision bucketed by calendar day for the past N days.

    Returns list of {date, agent, precision, n} sorted by date, suitable
    for drawing per-agent trend lines on the live-performance card.
    """
    import json as _json
    try:
        from sqlalchemy import select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except Exception:
        return []

    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    by_date_agent: dict[tuple[str, str], dict[str, int]] = {}
    try:
        with db_session() as session:
            stmt = (
                select(FeedbackRecord, InsightRecord)
                .outerjoin(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
                .where(InsightRecord.risk.in_(["WARNING", "CRITICAL"]))
                .where(InsightRecord.created_at >= cutoff)
                .order_by(InsightRecord.created_at)
            )
            for fb, ins in session.execute(stmt).all():
                if ins is None:
                    continue
                if fb.correct is not None:
                    is_correct = fb.correct
                elif fb.rating is not None:
                    is_correct = fb.rating >= 4
                else:
                    continue
                date_str = ins.created_at.strftime("%Y-%m-%d")
                try:
                    findings = _json.loads(ins.findings_json or "[]")
                except Exception:
                    continue
                for finding in findings:
                    if finding.get("risk") not in ("WARNING", "CRITICAL"):
                        continue
                    agent = finding.get("agent", "unknown")
                    key = (date_str, agent)
                    s = by_date_agent.setdefault(key, {"n_correct": 0, "n_total": 0})
                    s["n_total"] += 1
                    if is_correct:
                        s["n_correct"] += 1
    except Exception as exc:
        logger.warning("rolling_precision_series failed: %s", exc)
        return []

    return [
        {
            "date": date_str,
            "agent": agent,
            "precision": round(s["n_correct"] / s["n_total"], 4) if s["n_total"] > 0 else None,
            "n": s["n_total"],
        }
        for (date_str, agent), s in sorted(by_date_agent.items())
    ]


def reliability_diagram_data(n_bins: int = 10) -> list[dict]:
    """Confidence bins vs. actual accuracy for a reliability (calibration) diagram.

    Returns list of {bucket_min, bucket_max, mean_confidence, actual_accuracy, n}
    for each non-empty decile bucket. Buckets with n=0 are omitted.
    """
    try:
        pairs = _feedback_pairs()
    except Exception:
        return []
    if not pairs:
        return []

    bins: list[dict] = [
        {"bucket_min": i / n_bins, "bucket_max": (i + 1) / n_bins, "confs": [], "labels": []}
        for i in range(n_bins)
    ]
    for conf, label in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx]["confs"].append(conf)
        bins[idx]["labels"].append(label)

    result = []
    for b in bins:
        n = len(b["labels"])
        if n == 0:
            continue
        result.append({
            "bucket_min": round(b["bucket_min"], 2),
            "bucket_max": round(b["bucket_max"], 2),
            "mean_confidence": round(sum(b["confs"]) / n, 4),
            "actual_accuracy": round(sum(b["labels"]) / n, 4),
            "n": n,
        })
    return result


def per_driver_precision(since: _dt.datetime | None = None) -> dict[str, dict[str, dict]]:
    """Per-driver, per-agent precision from labeled WARNING/CRITICAL insights.

    Returns {driver_id: {agent: {precision, n_correct, n_total}}}
    Only drivers with n_total >= 3 across all agents are included.
    """
    import json as _json
    try:
        from sqlalchemy import select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except Exception:
        return {}

    raw: dict[str, dict[str, dict[str, int]]] = {}
    try:
        with db_session() as session:
            stmt = (
                select(FeedbackRecord, InsightRecord)
                .outerjoin(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
                .where(InsightRecord.risk.in_(["WARNING", "CRITICAL"]))
            )
            if since is not None:
                stmt = stmt.where(InsightRecord.created_at >= since)
            for fb, ins in session.execute(stmt).all():
                if ins is None:
                    continue
                if fb.correct is not None:
                    is_correct = fb.correct
                elif fb.rating is not None:
                    is_correct = fb.rating >= 4
                else:
                    continue
                driver = ins.driver_id or "unknown"
                try:
                    findings = _json.loads(ins.findings_json or "[]")
                except Exception:
                    continue
                for finding in findings:
                    if finding.get("risk") not in ("WARNING", "CRITICAL"):
                        continue
                    agent = finding.get("agent", "unknown")
                    s = raw.setdefault(driver, {}).setdefault(agent, {"n_correct": 0, "n_total": 0})
                    s["n_total"] += 1
                    if is_correct:
                        s["n_correct"] += 1
    except Exception as exc:
        logger.warning("per_driver_precision query failed: %s", exc)
        return {}

    result = {}
    for driver, agent_map in raw.items():
        total_n = sum(s["n_total"] for s in agent_map.values())
        if total_n < 3:
            continue
        result[driver] = {
            agent: {
                "precision": round(s["n_correct"] / s["n_total"], 4) if s["n_total"] > 0 else None,
                "n_correct": s["n_correct"],
                "n_total": s["n_total"],
            }
            for agent, s in sorted(agent_map.items())
        }
    return dict(sorted(result.items(), key=lambda kv: -sum(s["n_total"] for s in kv[1].values()))[:20])


def alert_rate_series(days: int = 30) -> list[dict]:
    """Daily count of WARNING/CRITICAL/WATCH insights for the past N days.

    Returns [{date, risk, n}] sorted by date, skipping shadow rows.
    """
    try:
        from sqlalchemy import func, select
        from f1di.storage.database import db_session
        from f1di.storage.models import InsightRecord
    except Exception:
        return []

    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    try:
        with db_session() as session:
            rows = session.execute(
                select(
                    func.date(InsightRecord.created_at).label("date"),
                    InsightRecord.risk,
                    func.count().label("n"),
                )
                .where(InsightRecord.shadow == False)  # noqa: E712
                .where(InsightRecord.created_at >= cutoff)
                .where(InsightRecord.risk.in_(["WARNING", "CRITICAL", "WATCH"]))
                .group_by(func.date(InsightRecord.created_at), InsightRecord.risk)
                .order_by(func.date(InsightRecord.created_at))
            ).all()
        return [{"date": str(row.date), "risk": row.risk, "n": row.n} for row in rows]
    except Exception as exc:
        logger.warning("alert_rate_series failed: %s", exc)
        return []


def check_precision_degradation(
    threshold: float = 0.60,
    drop_pp: float = 0.10,
    lookback_days: int = 7,
    baseline_days: int = 30,
) -> list[dict]:
    """Detect agents whose recent precision has dropped significantly.

    An agent is flagged when its lookback_days precision is both below
    `threshold` AND has dropped by at least `drop_pp` from the baseline.
    Agents with fewer than 5 recent samples are skipped.
    """
    now = _dt.datetime.utcnow()
    recent   = per_agent_accuracy(since=now - _dt.timedelta(days=lookback_days))
    baseline = per_agent_accuracy(since=now - _dt.timedelta(days=baseline_days))

    alerts = []
    for agent, r in recent.items():
        prec_7d = r.get("precision")
        if prec_7d is None or r.get("n_total", 0) < 5:
            continue
        prec_base = baseline.get(agent, {}).get("precision")
        if prec_base is None:
            continue
        drop = prec_base - prec_7d
        if prec_7d < threshold and drop >= drop_pp:
            alerts.append({
                "agent": agent,
                "precision_recent": round(prec_7d, 4),
                "precision_baseline": round(prec_base, 4),
                "drop_pp": round(drop * 100, 1),
                "n_recent": r["n_total"],
            })
    return alerts


def _feedback_pairs() -> list[tuple[float, float]]:
    from sqlalchemy import select
    from f1di.storage.database import db_session
    from f1di.storage.models import FeedbackRecord, InsightRecord

    pairs: list[tuple[float, float]] = []
    with db_session() as session:
        # LEFT JOIN: include all feedback; orphaned records (no matching insight)
        # use confidence=0.5 (maximum uncertainty) as a conservative default.
        stmt = (
            select(FeedbackRecord, InsightRecord)
            .outerjoin(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
        )
        for fb, ins in session.execute(stmt).all():
            if fb.correct is not None:
                label = 1.0 if fb.correct else 0.0
            elif fb.rating is not None:
                label = (fb.rating - 1) / 4.0
            else:
                continue
            confidence = (ins.raw_score if ins is not None and ins.raw_score is not None
                          else ins.confidence if ins is not None else 0.5)
            pairs.append((confidence, label))
    return pairs


def retrain(
    *,
    min_feedback: int = 20,
    calibrator_path: Path = _CALIBRATOR_PATH,
    quality_path: Path = _QUALITY_PATH,
    history_path: Path | None = None,
) -> dict:
    """Retrain isotonic calibrator augmenting synthetic base with human feedback.

    Uses calibrated confidence as a proxy for raw score — reasonable approximation
    since the isotonic mapping is near-identity and we're only fine-tuning.
    """
    history_path = history_path or calibrator_path.parent / _HISTORY_PATH.name
    try:
        pairs = _feedback_pairs()
    except Exception as exc:
        logger.warning("Could not load feedback from DB: %s", exc)
        return {"skipped": True, "reason": str(exc)}

    if len(pairs) < min_feedback:
        logger.info(
            "Retrain skipped — %d feedback records available, need %d", len(pairs), min_feedback
        )
        return {"skipped": True, "n_feedback": len(pairs), "reason": "insufficient_feedback"}

    from f1di.confidence.calibration import ConfidenceCalibrator
    from f1di.confidence.fitting import (
        calibration_brier,
        calibration_ece,
        generate_calibration_dataset,
    )

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    _file_op(
        "create calibration directory",
        calibrator_path.parent,
        lambda: calibrator_path.parent.mkdir(parents=True, exist_ok=True),
    )

    # --- read previous ECE for lineage record ---
    prev_ece: float | None = None
    prev_model: str | None = None
    if quality_path.exists():
        try:
            prev = json.loads(quality_path.read_text())
            prev_ece = prev.get("ece")
            prev_model = prev.get("model_path")
        except Exception:
            pass

    # --- snapshot training data ---
    snapshot_path = calibrator_path.parent / f"feedback_snapshot_{ts}.jsonl"
    def _write_snapshot() -> None:
        with snapshot_path.open("w") as fh:
            for confidence, label in pairs:
                fh.write(json.dumps({"confidence": confidence, "label": label}) + "\n")

    _file_op("write feedback snapshot", snapshot_path, _write_snapshot)

    # --- train ---
    X_syn, y_syn = generate_calibration_dataset(n_races=30, seed=42)
    X_fb = [p[0] for p in pairs]
    y_fb = [p[1] for p in pairs]
    X = X_syn + X_fb * 3
    y = y_syn + y_fb * 3

    calibrator = ConfidenceCalibrator.fit(X, y)

    # --- save versioned copy; live copy only written if quality passes ---
    versioned_path = calibrator_path.parent / f"isotonic_{ts}.pkl"
    _file_op("write versioned calibrator", versioned_path, lambda: calibrator.save(versioned_path))

    ece = calibration_ece(calibrator, n_races=15, seed=999)
    brier = calibration_brier(calibrator, n_races=15, seed=999)

    # --- quality regression guard ---
    # Block the live copy if ECE degrades more than 1 pp vs the previous run.
    # The versioned pkl is always written for audit; only the live copy is held back.
    regression_detected = prev_ece is not None and ece > prev_ece + 0.01
    if not regression_detected:
        _promote_live_model(versioned_path, calibrator_path)
    else:
        logger.warning(
            "Calibrator retrain BLOCKED — ECE %.4f regressed from %.4f (delta=%.4f > 0.01); "
            "versioned model saved but live model NOT updated.",
            ece, prev_ece, ece - prev_ece,
        )

    try:
        from f1di.observability.metrics import CALIBRATION_ECE_GAUGE, CALIBRATION_REGRESSION_BLOCKED
        CALIBRATION_ECE_GAUGE.set(ece)
        CALIBRATION_REGRESSION_BLOCKED.set(1 if regression_detected else 0)
    except Exception:
        pass

    fitted_at = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}Z"
    quality = {
        "ece": ece,
        "brier_score": brier,
        "fitted_at": fitted_at,
        "model_path": str(versioned_path),
        "previous_ece": prev_ece,
        "previous_model_path": prev_model,
        "feedback_snapshot": str(snapshot_path),
        "regression_detected": regression_detected,
        "calibration_dataset": {
            "generator": "synthetic+feedback",
            "n_synthetic": len(X_syn),
            "n_feedback": len(pairs),
            "feedback_weight": 3,
        },
        "per_agent_accuracy": per_agent_accuracy(),
    }
    _file_op(
        "write calibration quality",
        quality_path,
        lambda: quality_path.write_text(json.dumps(quality, indent=2)),
    )

    # --- append to rolling model history ---
    history: list = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            pass
    history.append({
        "fitted_at": fitted_at,
        "ece": ece,
        "brier_score": brier,
        "previous_ece": prev_ece,
        "n_feedback": len(pairs),
        "model_path": str(versioned_path),
        "feedback_snapshot": str(snapshot_path),
        "regression_detected": regression_detected,
    })
    _file_op(
        "write calibration history",
        history_path,
        lambda: history_path.write_text(json.dumps(history, indent=2)),
    )

    if regression_detected:
        return {
            "skipped": False,
            "regression_detected": True,
            "n_feedback": len(pairs),
            "ece": ece,
            "brier_score": brier,
            "previous_ece": prev_ece,
            "versioned_model_path": str(versioned_path),
            "live_model_unchanged": True,
            "feedback_snapshot": str(snapshot_path),
        }

    logger.info(
        "Calibrator retrained — ECE %.4f (was %.4f)  Brier %.4f  (n_feedback=%d)",
        ece, prev_ece or 0.0, brier, len(pairs),
    )
    return {
        "skipped": False,
        "regression_detected": False,
        "n_feedback": len(pairs),
        "ece": ece,
        "brier_score": brier,
        "previous_ece": prev_ece,
        "model_path": str(versioned_path),
        "feedback_snapshot": str(snapshot_path),
    }


