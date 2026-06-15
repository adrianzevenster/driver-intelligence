"""Logistic regression classifier for safety car / VSC risk.

Four classes: INFO (0), WATCH (1), WARNING (2), CRITICAL (3).
CRITICAL = SC or VSC actively deployed / imminent.
WARNING  = high probability of SC within next 2-3 laps.
WATCH    = elevated risk; monitor closely.
INFO     = normal racing conditions.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger("f1di.agents.safety_car_classifier")

_CLASSIFIER_PATH = Path("data/calibration/safety_car_classifier.pkl")

FEATURE_NAMES: list[str] = [
    "mean_speed_kph",
    "speed_delta_kph",
    "rain_intensity",
    "grip_estimate",
    "lockup_count",
    "throttle_smoothness",
    "race_phase",
    "brake_temp_front_max",
]

_LABEL_MAP: dict[int, str] = {0: "INFO", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"}
_LABEL_INV: dict[str, int] = {v: k for k, v in _LABEL_MAP.items()}

MODEL_VERSION = "lr-v1"
MODEL_TYPE = "LogisticRegression"


def _multiclass_brier(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    n, n_c = len(y), len(classes)
    cls_idx = {int(c): i for i, c in enumerate(classes)}
    Y_oh = np.zeros((n, n_c), dtype=np.float64)
    for i, yi in enumerate(y):
        Y_oh[i, cls_idx[int(yi)]] = 1.0
    return float(np.mean(np.sum((proba - Y_oh) ** 2, axis=1)))


def features_to_array(features) -> np.ndarray:
    return np.array([
        features.mean_speed_kph,
        features.speed_delta_kph,
        features.rain_intensity,
        features.grip_estimate,
        float(features.lockup_count),
        features.throttle_smoothness,
        features.race_phase,
        features.brake_temp_front_max,
    ], dtype=np.float64)


class SafetyCarClassifier:
    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        self.classes_: list[str] = []
        self.n_train: int = 0
        self.n_real: int = 0
        self.accuracy: float = 0.0
        self.brier_score: float = 1.0
        self.model_version: str = MODEL_VERSION
        self.model_type: str = MODEL_TYPE

    def fit(self, X: np.ndarray, y: np.ndarray, n_real: int = 0) -> "SafetyCarClassifier":
        from sklearn.metrics import accuracy_score
        X_s = self._scaler.fit_transform(X)
        self._model.fit(X_s, y)
        self.classes_ = [_LABEL_MAP[int(c)] for c in self._model.classes_]
        self.n_train = int(len(y))
        self.n_real = n_real
        proba = self._model.predict_proba(X_s)
        self.accuracy = float(accuracy_score(y, self._model.predict(X_s)))
        self.brier_score = float(_multiclass_brier(proba, y, self._model.classes_))
        logger.info(
            "SafetyCarClassifier fitted: n=%d n_real=%d acc=%.3f brier=%.4f",
            self.n_train, self.n_real, self.accuracy, self.brier_score,
        )
        return self

    def ood_score(self, features) -> float:
        x = features_to_array(features)
        z = np.abs((x - self._scaler.mean_) / np.maximum(self._scaler.scale_, 1e-8))
        return float(z.max())

    def predict(self, features) -> tuple[str, float, np.ndarray]:
        x = features_to_array(features).reshape(1, -1)
        x_s = self._scaler.transform(x)
        proba = self._model.predict_proba(x_s)[0]
        idx = int(np.argmax(proba))
        return _LABEL_MAP[idx], float(proba[idx]), proba

    def save(self, path: Path = _CLASSIFIER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path = _CLASSIFIER_PATH) -> "SafetyCarClassifier":
        return pickle.loads(path.read_bytes())


def _synthetic_label(
    mean_speed: float,
    speed_delta: float,
    rain: float,
    grip: float,
    lockups: int,
    brake_temp: float,
) -> int:
    # SC/VSC deployed — massive speed reduction is the clearest signal
    if mean_speed < 80.0:
        return 3  # CRITICAL
    if rain > 0.7 and grip < 0.55:
        return 3  # CRITICAL — extremely wet, likely red flag / SC

    # High probability of SC within next few laps
    if mean_speed < 160.0 or speed_delta < -60.0:
        return 2  # WARNING
    if rain > 0.5 and grip < 0.65:
        return 2  # WARNING

    # Elevated but manageable risk
    if rain > 0.35 or grip < 0.72:
        return 1  # WATCH
    if lockups >= 2 and brake_temp > 500.0:
        return 1  # WATCH — reactive emergency braking
    if speed_delta < -35.0 and rain > 0.2:
        return 1  # WATCH

    return 0  # INFO


def generate_synthetic(n: int = 800, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    while len(X) < n:
        speed  = float(rng.uniform(50.0, 320.0))
        delta  = float(rng.uniform(-100.0, 40.0))
        rain   = float(rng.uniform(0.0, 1.0))
        grip   = float(rng.uniform(0.40, 1.0))
        lockup = int(rng.integers(0, 5))
        smooth = float(rng.uniform(0.3, 1.0))
        phase  = float(rng.uniform(0.0, 1.0))
        brake  = float(rng.uniform(100.0, 700.0))
        X.append([speed, delta, rain, grip, float(lockup), smooth, phase, brake])
        y.append(_synthetic_label(speed, delta, rain, grip, lockup, brake))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def _load_labeled_from_db() -> tuple[np.ndarray, np.ndarray]:
    try:
        import json as _json
        from sqlalchemy import select
        from f1di.storage.database import db_session
        from f1di.storage.models import FeedbackRecord, InsightRecord
    except Exception:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)

    X, y = [], []
    try:
        with db_session() as session:
            rows = session.execute(
                select(FeedbackRecord, InsightRecord)
                .outerjoin(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
            ).all()
    except Exception as exc:
        logger.warning("safety_car_classifier DB query failed: %s", exc)
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
        sc = next((f for f in findings if f.get("agent") == "safety_car"), None)
        if sc is None:
            continue
        feats = sc.get("features", {})
        pred_label = _LABEL_INV.get(sc.get("risk", ins.risk), 0)
        true_label = pred_label if is_correct else max(0, pred_label - 1)
        X.append([
            float(feats.get("mean_speed_kph", 220.0)),
            float(feats.get("speed_delta_kph", 0.0)),
            float(feats.get("rain_intensity", 0.0)),
            float(feats.get("grip_estimate", 0.9)),
            float(feats.get("lockup_count", 0)),
            float(feats.get("throttle_smoothness", 0.8)),
            float(feats.get("race_phase", 0.5)),
            float(feats.get("brake_temp_front_max", 300.0)),
        ])
        y.append(true_label)

    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def train_from_labels(
    output_path: Path = _CLASSIFIER_PATH,
    real_oversample: int = 5,
    synthetic_n: int = 800,
) -> dict:
    X_s, y_s = generate_synthetic(n=synthetic_n)
    X_r, y_r = _load_labeled_from_db()
    n_real = len(y_r)

    if n_real >= 10:
        X = np.vstack([X_s, np.repeat(X_r, real_oversample, axis=0)])
        y = np.concatenate([y_s, np.repeat(y_r, real_oversample)])
    else:
        X, y = X_s, y_s

    unique, counts = np.unique(y, return_counts=True)
    clf = SafetyCarClassifier().fit(X, y, n_real=n_real)
    from f1di.agents.classifier_utils import save_with_snapshot, record_history
    snap = save_with_snapshot(clf, output_path)
    record_history(
        clf, agent="safety_car",
        versioned_path=snap["versioned_path"],
        blocked=snap["blocked"],
        history_path=output_path.parent / "model_history.json",
    )
    return {
        "n_synthetic": len(y_s), "n_real": n_real, "n_total": len(y),
        "accuracy": round(clf.accuracy, 4), "classes": clf.classes_,
        "class_distribution": {_LABEL_MAP[int(k)]: int(v) for k, v in zip(unique, counts)},
        "output_path": str(output_path),
        "snapshot_blocked": snap["blocked"],
        "versioned_path": snap["versioned_path"],
    }
