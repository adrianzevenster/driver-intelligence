"""Auto-retrain: trigger classifier refit when enough new real labels have accumulated.

Called from BackgroundTasks after submit_feedback and label_race_outcomes so
retraining is non-blocking and happens automatically without manual intervention.

Logic per agent:
  db_n_real  = labels currently in the DB for this agent
  pkl_n_real = n_real baked into the live classifier pkl at last fit time
  delta      = db_n_real - pkl_n_real
  if delta >= THRESHOLD and no retrain in progress → retrain in this thread

The threshold is intentionally small (5) so the model improves promptly after
each race weekend's labelling run. A per-agent lock prevents concurrent retrains
of the same classifier.
"""
from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path

logger = logging.getLogger("f1di.agents.auto_retrain")

RETRAIN_THRESHOLD = 5

_AGENT_PATHS: dict[str, Path] = {
    "tire":      Path("data/calibration/tire_classifier.pkl"),
    "battery":   Path("data/calibration/battery_classifier.pkl"),
    "weather":   Path("data/calibration/weather_classifier.pkl"),
    "telemetry": Path("data/calibration/telemetry_classifier.pkl"),
}

_lock = threading.Lock()
_in_progress: set[str] = set()


def _pkl_n_real(agent: str) -> int:
    p = _AGENT_PATHS.get(agent)
    if not p or not p.exists():
        return 0
    try:
        return int(pickle.loads(p.read_bytes()).n_real)
    except Exception:
        return 0


def _db_n_real(agent: str) -> int:
    """Count labeled feedback rows available for this agent."""
    try:
        if agent == "tire":
            from f1di.agents.tire_classifier import _load_labeled_from_db
        elif agent == "battery":
            from f1di.agents.battery_classifier import _load_labeled_from_db
        elif agent == "weather":
            from f1di.agents.weather_classifier import _load_labeled_from_db
        elif agent == "telemetry":
            from f1di.agents.telemetry_classifier import _load_labeled_from_db
        else:
            return 0
        _, y = _load_labeled_from_db()
        return len(y)
    except Exception as exc:
        logger.warning("auto_retrain: label count failed for %s: %s", agent, exc)
        return 0


def _do_retrain(agent: str) -> None:
    try:
        if agent == "tire":
            from f1di.agents.tire_classifier import train_from_labels
        elif agent == "battery":
            from f1di.agents.battery_classifier import train_from_labels
        elif agent == "weather":
            from f1di.agents.weather_classifier import train_from_labels
        elif agent == "telemetry":
            from f1di.agents.telemetry_classifier import train_from_labels
        else:
            return
        report = train_from_labels()
        logger.info(
            "auto_retrain: %s complete — acc=%.4f brier=%.4f n_real=%d blocked=%s",
            agent, report["accuracy"], report.get("brier_score", float("nan")),
            report["n_real"], report.get("snapshot_blocked", False),
        )
    except Exception as exc:
        logger.error("auto_retrain: %s retrain raised: %s", agent, exc, exc_info=True)
    finally:
        with _lock:
            _in_progress.discard(agent)


def maybe_retrain(agent: str, threshold: int = RETRAIN_THRESHOLD) -> None:
    """Check delta and retrain if warranted. Safe to call from BackgroundTasks."""
    if agent not in _AGENT_PATHS:
        return
    db_n = _db_n_real(agent)
    pkl_n = _pkl_n_real(agent)
    delta = db_n - pkl_n
    if delta < threshold:
        logger.debug("auto_retrain: %s skip (delta=%d < %d)", agent, delta, threshold)
        return
    with _lock:
        if agent in _in_progress:
            logger.debug("auto_retrain: %s retrain already running", agent)
            return
        _in_progress.add(agent)
    logger.info(
        "auto_retrain: triggering %s (db_n=%d pkl_n=%d delta=%d threshold=%d)",
        agent, db_n, pkl_n, delta, threshold,
    )
    _do_retrain(agent)


def maybe_retrain_all(threshold: int = RETRAIN_THRESHOLD) -> None:
    """Retrain check for all agents. Used after batch labelling (label_race_outcomes)."""
    for agent in _AGENT_PATHS:
        maybe_retrain(agent, threshold)


def retrain_status() -> dict:
    """Return current retrain state (used by /model-health)."""
    with _lock:
        in_progress = list(_in_progress)
    result = {}
    for agent, path in _AGENT_PATHS.items():
        pkl_n = _pkl_n_real(agent)
        result[agent] = {
            "pkl_n_real": pkl_n,
            "retrain_in_progress": agent in in_progress,
        }
    return {"agents": result, "threshold": RETRAIN_THRESHOLD}
