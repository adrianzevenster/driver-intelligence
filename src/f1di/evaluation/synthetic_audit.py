"""Synthetic label quality audit — callable for scheduled runs.

Compares classifier accuracy when trained on synthetic-only data vs. synthetic +
real flywheel labels evaluated on a held-out real-label test fold.  A large
negative acc_delta means synthetic labels are teaching patterns that conflict
with real race outcomes.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("f1di.evaluation.synthetic_audit")

_AUDIT_PATH = Path("data/calibration/synthetic_audit.json")

_AGENTS = {
    "tire":      "f1di.agents.tire_classifier",
    "battery":   "f1di.agents.battery_classifier",
    "weather":   "f1di.agents.weather_classifier",
    "telemetry": "f1di.agents.telemetry_classifier",
}


def _stratified_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.20, seed: int = 0
):
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_frac))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _acc(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(scaler.fit_transform(X_train), y_train)
    return float(accuracy_score(y_test, clf.predict(scaler.transform(X_test))))


def run_audit(min_real: int = 8) -> dict[str, dict]:
    """Run synthetic vs. real label alignment audit for all agents.

    Returns {agent: result} and persists to data/calibration/synthetic_audit.json.
    Each result has: skipped, n_real, acc_synth, acc_blend, acc_delta, aligned.
    aligned=False means blending real labels HURTS by >3pp — synthetic is conflicting.
    """
    import importlib

    results: dict[str, dict] = {}
    for agent, mod_name in _AGENTS.items():
        try:
            mod = importlib.import_module(mod_name)
            X_real, y_real = mod._load_labeled_from_db()
            n_real = len(y_real)
            if n_real < min_real:
                results[agent] = {"skipped": True, "n_real": n_real, "reason": f"< {min_real} real labels"}
                continue

            X_tr, y_tr, X_te, y_te = _stratified_split(X_real, y_real)
            X_syn, y_syn = mod.generate_synthetic(n=600, seed=42)

            acc_synth = _acc(X_syn, y_syn, X_te, y_te)
            X_blend = np.vstack([X_syn, np.repeat(X_tr, 5, axis=0)])
            y_blend = np.concatenate([y_syn, np.repeat(y_tr, 5)])
            acc_blend = _acc(X_blend, y_blend, X_te, y_te)

            acc_delta = acc_blend - acc_synth
            results[agent] = {
                "skipped": False,
                "n_real": n_real,
                "n_test": len(y_te),
                "acc_synth": round(acc_synth, 4),
                "acc_blend": round(acc_blend, 4),
                "acc_delta": round(acc_delta, 4),
                "aligned": acc_delta >= -0.03,
            }
            logger.info(
                "synthetic_audit %s: n_real=%d acc_synth=%.3f acc_blend=%.3f delta=%+.3f aligned=%s",
                agent, n_real, acc_synth, acc_blend, acc_delta, results[agent]["aligned"],
            )
        except Exception as exc:
            logger.warning("synthetic_audit failed for %s: %s", agent, exc)
            results[agent] = {"skipped": True, "n_real": 0, "reason": str(exc)}

    record = {
        "audited_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agents": results,
    }
    _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _AUDIT_PATH.write_text(json.dumps(record, indent=2))
    return results


def load_last_audit() -> dict | None:
    """Load the most recent audit result from disk."""
    if not _AUDIT_PATH.exists():
        return None
    try:
        return json.loads(_AUDIT_PATH.read_text())
    except Exception:
        return None
