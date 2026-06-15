"""Shared utilities for agent classifiers.

Provides save_with_snapshot() — mirrors the isotonic calibrator's versioning
pattern: always writes a timestamped copy; only promotes to the live path when
accuracy hasn't regressed, so a bad retrain can't silently clobber a good model.

Provides record_history() — appends a structured entry to model_history.json
so every classifier fit is traceable alongside calibrator retrains.
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
import time
from pathlib import Path

logger = logging.getLogger("f1di.agents.classifier_utils")

_HISTORY_PATH = Path("data/calibration/model_history.json")


def record_history(clf, agent: str, versioned_path: str, blocked: bool, history_path: Path = _HISTORY_PATH) -> None:
    """Append one classifier fit entry to model_history.json."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        entries: list = json.loads(history_path.read_text()) if history_path.exists() else []
    except Exception:
        entries = []
    entries.append({
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": agent,
        "model_version": getattr(clf, "model_version", "unknown"),
        "model_type": getattr(clf, "model_type", "unknown"),
        "accuracy": round(clf.accuracy, 4),
        "brier_score": round(clf.brier_score, 4),
        "n_train": clf.n_train,
        "n_real": clf.n_real,
        "classes": clf.classes_,
        "versioned_path": versioned_path,
        "blocked": blocked,
    })
    history_path.write_text(json.dumps(entries, indent=2))
    logger.info("model_history updated: agent=%s version=%s acc=%.4f", agent, getattr(clf, "model_version", "?"), clf.accuracy)


def save_with_snapshot(
    clf,
    live_path: Path,
    min_accuracy_delta: float = 0.02,
) -> dict:
    """Save *clf* with a versioned snapshot and an accuracy regression guard.

    Args:
        clf: Any classifier with an `.accuracy` and `.n_real` attribute.
        live_path: Path of the canonical live pkl (e.g. `data/calibration/tire_classifier.pkl`).
        min_accuracy_delta: Block the live update if new accuracy drops more than
            this many points below the previous live model's accuracy.

    Returns:
        Dict with keys: blocked, versioned_path, accuracy, prev_accuracy.
    """
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stem = live_path.stem
    versioned_path = live_path.parent / f"{stem}_{ts}.pkl"
    live_path.parent.mkdir(parents=True, exist_ok=True)

    prev_accuracy: float | None = None
    if live_path.exists():
        try:
            prev = pickle.loads(live_path.read_bytes())
            prev_accuracy = float(prev.accuracy)
        except Exception:
            pass

    # Always write the versioned copy for audit.
    versioned_path.write_bytes(pickle.dumps(clf))

    blocked = (
        prev_accuracy is not None
        and clf.accuracy < prev_accuracy - min_accuracy_delta
    )

    if not blocked:
        shutil.copy2(versioned_path, live_path)
        logger.info(
            "%s saved: acc=%.4f n_real=%d versioned=%s",
            stem, clf.accuracy, clf.n_real, versioned_path.name,
        )
    else:
        logger.warning(
            "%s retrain BLOCKED — new acc %.4f regressed from %.4f (delta=%.4f > %.2f); "
            "versioned copy saved, live model unchanged.",
            stem, clf.accuracy, prev_accuracy,
            prev_accuracy - clf.accuracy, min_accuracy_delta,
        )

    return {
        "blocked": blocked,
        "versioned_path": str(versioned_path),
        "accuracy": round(clf.accuracy, 4),
        "prev_accuracy": round(prev_accuracy, 4) if prev_accuracy is not None else None,
    }
