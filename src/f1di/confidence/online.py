from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("f1di.confidence.online")

_CALIBRATOR_PATH = Path("data/calibration/isotonic.pkl")
_QUALITY_PATH = Path("data/calibration/quality.json")
_HISTORY_PATH = Path("data/calibration/model_history.json")


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


def per_agent_accuracy() -> dict[str, dict]:
    """Per-agent precision computed from labeled WARNING/CRITICAL insights in the DB.

    For each agent, 'precision' is the fraction of WARNING/CRITICAL findings
    that were confirmed correct by human feedback or outcome labels.
    Returns an empty dict when no labeled data exists yet.
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

    return {
        agent: {
            "precision": round(s["n_correct"] / s["n_total"], 4) if s["n_total"] > 0 else None,
            "n_correct": s["n_correct"],
            "n_total": s["n_total"],
        }
        for agent, s in sorted(agent_stats.items())
    }


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
