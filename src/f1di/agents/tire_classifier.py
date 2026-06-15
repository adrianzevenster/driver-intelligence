"""Logistic regression classifier for tire strategy risk level.

Trained on flywheel-labeled insights + synthetic rule-distilled data for cold start.
At runtime TireStrategyAgent loads this lazily (mtime-aware) and uses it in place of
the hand-written threshold cascade.  Falls back to rules if no pkl exists.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger("f1di.agents.tire_classifier")

_CLASSIFIER_PATH = Path("data/calibration/tire_classifier.pkl")

FEATURE_NAMES: list[str] = [
    "wear_pressure",       # max(fl_wear, fr_wear, rear_wear_mean)
    "grip_estimate",
    "fl_wear_slope",
    "fr_wear_slope",
    "rear_wear_slope",
    "axle_imbalance_fl_rl",
    "laps_remaining",
    "stint_fraction",
    "race_phase",
]

_LABEL_MAP: dict[int, str] = {0: "INFO", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"}
_LABEL_INV: dict[str, int] = {v: k for k, v in _LABEL_MAP.items()}

MODEL_VERSION = "lr-v1"
MODEL_TYPE = "LogisticRegression"


def _multiclass_brier(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between probability vector and one-hot target."""
    n = len(y)
    n_c = len(classes)
    cls_idx = {int(c): i for i, c in enumerate(classes)}
    Y_oh = np.zeros((n, n_c), dtype=np.float64)
    for i, yi in enumerate(y):
        Y_oh[i, cls_idx[int(yi)]] = 1.0
    return float(np.mean(np.sum((proba - Y_oh) ** 2, axis=1)))


def features_to_array(features, wear_pressure: float) -> np.ndarray:
    return np.array([
        wear_pressure,
        features.grip_estimate,
        features.fl_wear_slope,
        features.fr_wear_slope,
        features.rear_wear_slope,
        features.axle_imbalance_fl_rl,
        features.laps_remaining,
        features.stint_fraction,
        features.race_phase,
    ], dtype=np.float64)


class TireClassifier:
    """Multinomial LR classifier for tire strategy risk level (INFO/WATCH/WARNING/CRITICAL)."""

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            C=1.0, max_iter=1000,
            solver="lbfgs", random_state=42,
        )
        self.classes_: list[str] = []
        self.n_train: int = 0
        self.n_real: int = 0
        self.accuracy: float = 0.0
        self.brier_score: float = 1.0
        self.model_version: str = MODEL_VERSION
        self.model_type: str = MODEL_TYPE

    def fit(self, X: np.ndarray, y: np.ndarray, n_real: int = 0) -> "TireClassifier":
        from sklearn.metrics import accuracy_score
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self.classes_ = [_LABEL_MAP[int(c)] for c in self._model.classes_]
        self.n_train = int(len(y))
        self.n_real = n_real
        proba = self._model.predict_proba(X_scaled)
        self.accuracy = float(accuracy_score(y, self._model.predict(X_scaled)))
        self.brier_score = float(_multiclass_brier(proba, y, self._model.classes_))
        logger.info(
            "TireClassifier fitted: n_total=%d n_real=%d acc=%.3f brier=%.4f classes=%s",
            self.n_train, self.n_real, self.accuracy, self.brier_score, self.classes_,
        )
        return self

    def ood_score(self, features, wear_pressure: float) -> float:
        """Max absolute Z-score of features vs training distribution. >4.0 = OOD."""
        x = features_to_array(features, wear_pressure)
        z = np.abs((x - self._scaler.mean_) / np.maximum(self._scaler.scale_, 1e-8))
        return float(z.max())

    def predict(self, features, wear_pressure: float) -> tuple[str, float, np.ndarray]:
        """Return (risk_label, confidence, proba_array) where proba is over all classes."""
        x = features_to_array(features, wear_pressure).reshape(1, -1)
        x_scaled = self._scaler.transform(x)
        proba = self._model.predict_proba(x_scaled)[0]
        # proba is ordered by self._model.classes_ (int labels sorted ascending)
        class_idx = int(np.argmax(proba))
        return _LABEL_MAP[class_idx], float(proba[class_idx]), proba

    def save(self, path: Path = _CLASSIFIER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))
        logger.info(
            "TireClassifier saved: %s  n=%d  real=%d  acc=%.3f",
            path, self.n_train, self.n_real, self.accuracy,
        )

    @staticmethod
    def load(path: Path = _CLASSIFIER_PATH) -> "TireClassifier":
        return pickle.loads(path.read_bytes())


# ── Synthetic data generation (cold-start) ─────────────────────────────────

def _synthetic_label(
    wear_pressure: float,
    grip: float,
    fl_slope: float,
    fr_slope: float,
    rear_slope: float,
    axle_imbalance: float,
) -> int:
    """Apply the rule cascade to a synthetic observation to get a training label."""
    from f1di.agents.thresholds import CircuitThresholds
    t = CircuitThresholds()

    if wear_pressure > t.wear_critical and grip < 0.62:
        return 3  # CRITICAL

    projected = wear_pressure + max(fl_slope, fr_slope) * 4  # spl ≈ 1 sample/lap
    if wear_pressure > t.wear_warning and grip < 0.72:
        return 2  # WARNING
    if projected > t.wear_critical * 0.97 and max(fl_slope, fr_slope) > 0.0:
        return 2  # WARNING

    # FR degrading asymmetrically
    if fr_slope > fl_slope + 0.0015 and wear_pressure > 0.38:
        return 1  # WATCH
    if axle_imbalance > 0.12 and wear_pressure * 0.85 > 0.25:
        return 1  # WATCH
    if rear_slope > 0.0022 and wear_pressure * 0.9 > 0.35:
        return 1  # WATCH

    return 0  # INFO


def generate_synthetic(n: int = 800, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate rule-distilled synthetic training data for cold-start."""
    rng = np.random.default_rng(seed)
    X_rows: list[list[float]] = []
    y_rows: list[int] = []

    while len(X_rows) < n:
        wp      = float(rng.uniform(0.25, 0.98))
        grip    = float(rng.uniform(0.40, 0.98))
        fl_sl   = float(rng.uniform(0.0, 0.025))
        fr_sl   = float(rng.uniform(0.0, 0.025))
        rr_sl   = float(rng.uniform(0.0, 0.018))
        axle    = float(rng.uniform(0.0, 0.28))
        laps_r  = float(rng.uniform(0.0, 45.0))
        stint_f = float(rng.uniform(0.0, 1.3))
        race_ph = float(rng.uniform(0.0, 1.0))
        label   = _synthetic_label(wp, grip, fl_sl, fr_sl, rr_sl, axle)
        X_rows.append([wp, grip, fl_sl, fr_sl, rr_sl, axle, laps_r, stint_f, race_ph])
        y_rows.append(label)

    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.int32)


# ── Real data from DB ──────────────────────────────────────────────────────

def _load_labeled_from_db() -> tuple[np.ndarray, np.ndarray]:
    """Query tire_strategy findings that have outcome labels from the flywheel."""
    try:
        import json as _json
        from sqlalchemy import select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except Exception:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)

    X_rows: list[list[float]] = []
    y_rows: list[int] = []

    try:
        with db_session() as session:
            stmt = (
                select(FeedbackRecord, InsightRecord)
                .outerjoin(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
            )
            rows = session.execute(stmt).all()
    except Exception as exc:
        logger.warning("tire_classifier: DB query failed: %s", exc)
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)

    for fb, ins in rows:
        if ins is None:
            continue
        if fb.correct is not None:
            is_correct = bool(fb.correct)
        elif fb.rating is not None:
            is_correct = int(fb.rating) >= 4
        else:
            continue

        try:
            findings = _json.loads(ins.findings_json or "[]")
        except Exception:
            continue

        tire = next((f for f in findings if f.get("agent") == "tire_strategy"), None)
        if tire is None:
            continue

        feats = tire.get("features", {})
        pred_label = _LABEL_INV.get(tire.get("risk", ins.risk), 0)
        # False prediction → downgrade one level (too alarming)
        true_label = pred_label if is_correct else max(0, pred_label - 1)

        X_rows.append([
            float(feats.get("wear_pressure", 0.0)),
            float(feats.get("grip_estimate", 0.75)),
            float(feats.get("fl_wear_slope", 0.0)),
            float(feats.get("fr_wear_slope", 0.0)),
            float(feats.get("rear_wear_slope", 0.0)),
            float(feats.get("axle_imbalance_fl_rl", 0.0)),
            float(feats.get("laps_remaining", 20.0)),
            float(feats.get("stint_fraction", 0.5)),
            float(feats.get("race_phase", 0.5)),
        ])
        y_rows.append(true_label)

    if not X_rows:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)

    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.int32)


# ── Training entry point ───────────────────────────────────────────────────

def train_from_labels(
    output_path: Path = _CLASSIFIER_PATH,
    real_oversample: int = 5,
    synthetic_n: int = 800,
) -> dict:
    """Train TireClassifier from synthetic cold-start + real flywheel labels.

    Real examples are repeated `real_oversample` times so a modest number of
    confirmed outcomes can already override the synthetic prior, giving a
    graceful cold-start → data-driven transition.
    """
    X_synth, y_synth = generate_synthetic(n=synthetic_n)
    X_real, y_real = _load_labeled_from_db()
    n_real = len(y_real)

    if n_real >= 10:
        X = np.vstack([X_synth, np.repeat(X_real, real_oversample, axis=0)])
        y = np.concatenate([y_synth, np.repeat(y_real, real_oversample)])
    else:
        X, y = X_synth, y_synth
        if n_real > 0:
            logger.info(
                "tire_classifier: %d real examples (min 10 to blend) — "
                "training on synthetic only this cycle", n_real,
            )

    unique, counts = np.unique(y, return_counts=True)
    class_dist = {_LABEL_MAP[int(k)]: int(v) for k, v in zip(unique, counts)}

    clf = TireClassifier().fit(X, y, n_real=n_real)
    from f1di.agents.classifier_utils import save_with_snapshot, record_history
    snap = save_with_snapshot(clf, output_path)
    record_history(clf, agent="tire", versioned_path=snap["versioned_path"], blocked=snap["blocked"], history_path=output_path.parent / "model_history.json")

    return {
        "n_synthetic": len(y_synth),
        "n_real": n_real,
        "n_total": len(y),
        "accuracy": round(clf.accuracy, 4),
        "classes": clf.classes_,
        "class_distribution": class_dist,
        "output_path": str(output_path),
        "snapshot_blocked": snap["blocked"],
        "versioned_path": snap["versioned_path"],
    }
