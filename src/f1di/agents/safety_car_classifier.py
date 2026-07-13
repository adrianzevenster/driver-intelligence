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

from f1di.agents.classifier_utils import _CALIBRATION_DIR, circuit_prec_for_track

logger = logging.getLogger("f1di.agents.safety_car_classifier")
_CLASSIFIER_PATH = _CALIBRATION_DIR / "safety_car_classifier.pkl"

FEATURE_NAMES: list[str] = [
    "mean_speed_kph",
    "speed_delta_kph",
    "rain_intensity",
    "grip_estimate",
    "lockup_count",
    "throttle_smoothness",
    "race_phase",
    "brake_temp_front_max",
    "circuit_avg_speed_kph",
    "circuit_type_enc",
    "race_laps_total",
    "circuit_precision_prior",
]

_LABEL_MAP: dict[int, str] = {0: "INFO", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"}
_LABEL_INV: dict[str, int] = {v: k for k, v in _LABEL_MAP.items()}

MODEL_VERSION = "hgb-v1"
MODEL_TYPE = "HistGradientBoosting"
DEFAULT_MODEL_TYPE = "hgbc"


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
        features.circuit_avg_speed_kph,
        features.circuit_type_enc,
        features.race_laps_total,
        features.circuit_precision_prior,
    ], dtype=np.float64)


class SafetyCarClassifier:
    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE) -> None:
        from f1di.agents.classifier_utils import build_model, _MODEL_DISPLAY, _MODEL_VERSION
        self._scaler, self._model = build_model(model_type, max_depth=4, agent="safety_car")
        self.classes_: list[str] = []
        self.n_train: int = 0
        self.n_real: int = 0
        self.accuracy: float = 0.0
        self.brier_score: float = 1.0
        self.cv_n_splits: int = 0
        self.cv_accuracy_std: float | None = None
        self.cv_brier_std: float | None = None
        self.cv_fold_accuracies: list[float] | None = None
        self.cv_fold_briers: list[float] | None = None
        self.real_sample_weight: float | None = None
        self.prior_cv_accuracy: float | None = None
        self.model_version: str = _MODEL_VERSION.get(model_type.lower(), model_type)
        self.model_type: str = _MODEL_DISPLAY.get(model_type.lower(), model_type)

    def fit(self, X: np.ndarray, y: np.ndarray, n_real: int = 0, sample_weight: np.ndarray | None = None) -> "SafetyCarClassifier":
        from sklearn.metrics import accuracy_score
        from f1di.agents.classifier_utils import fit_weighted_scaler
        fit_weighted_scaler(self._scaler, X, sample_weight)
        X_s = self._scaler.transform(X)
        self._model.fit(X_s, y, sample_weight=sample_weight)
        self.classes_ = [_LABEL_MAP[int(c)] for c in self._model.classes_]
        self.n_train = int(len(y))
        self.n_real = n_real

        from f1di.agents.classifier_utils import cross_val_eval
        cv = cross_val_eval(self._build_pipeline, X, y, _multiclass_brier, sample_weight=sample_weight, collect_predictions=True)
        if cv is not None:
            self.accuracy = cv["cv_accuracy"]
            self.brier_score = cv["cv_brier"]
            self.cv_n_splits = cv["n_splits"]
            self.cv_accuracy_std = cv["cv_accuracy_std"]
            self.cv_brier_std = cv["cv_brier_std"]
            self.cv_fold_accuracies = cv["fold_accuracies"]
            self.cv_fold_briers = cv["fold_briers"]
            from f1di.agents.classifier_utils import per_class_report
            self.cv_per_class = per_class_report(cv, _LABEL_MAP)
        else:
            proba = self._model.predict_proba(X_s)
            self.accuracy = float(accuracy_score(y, self._model.predict(X_s)))
            self.brier_score = float(_multiclass_brier(proba, y, self._model.classes_))
            self.cv_n_splits = 0
            self.cv_accuracy_std = None
            self.cv_brier_std = None
            self.cv_fold_accuracies = None
            self.cv_fold_briers = None
            self.cv_per_class = {}

        logger.info(
            "SafetyCarClassifier fitted: n=%d n_real=%d cv_acc=%.3f cv_brier=%.4f n_splits=%d",
            self.n_train, self.n_real, self.accuracy, self.brier_score, self.cv_n_splits,
        )
        return self

    @staticmethod
    def _build_pipeline(model_type: str = DEFAULT_MODEL_TYPE):
        from f1di.agents.classifier_utils import build_model
        return build_model(model_type, max_depth=4, agent="safety_car")

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
    throttle_smooth: float,
    race_phase: float,
) -> int:
    # CRITICAL: SC/VSC actively deployed — massive speed reduction or extreme conditions
    if mean_speed < 80.0:
        return 3
    if rain > 0.75 and grip < 0.50:
        return 3  # Red-flag / SC in extreme rain
    if mean_speed < 120.0 and speed_delta < -50.0:
        return 3  # Multi-car incident, cars bunching up behind SC

    # WARNING: High SC probability, non-linear danger combinations
    if mean_speed < 155.0 or speed_delta < -65.0:
        return 2
    if rain > 0.55 and grip < 0.62:
        return 2
    if lockups >= 3 and brake_temp > 580.0 and rain > 0.3:
        return 2  # Multiple drivers locking up in wet = incident imminent
    if speed_delta < -45.0 and throttle_smooth < 0.45:
        return 2  # Erratic braking across the field = incident
    if race_phase < 0.05 and (rain > 0.3 or lockups >= 2):
        return 2  # Formation/lap-1 incidents have elevated SC risk

    # WATCH: Elevated but manageable; multiple contributing factors
    if rain > 0.35 or grip < 0.68:
        return 1
    if lockups >= 2 and brake_temp > 480.0:
        return 1  # Reactive emergency braking
    if speed_delta < -35.0 and rain > 0.18:
        return 1
    if throttle_smooth < 0.40 and lockups >= 1:
        return 1  # Driver fighting the car
    if race_phase < 0.08 and mean_speed > 200.0:
        return 1  # First few laps — elevated incident risk even without obvious signals

    return 0  # INFO


def generate_synthetic(n: int = 1200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []

    # Structured scenario batches to ensure class coverage
    n_each = max(1, n // 8)

    def _make_row(speed, delta, rain, grip, lockup, smooth, phase, brake, c_speed, c_type, laps):
        return [speed, delta, rain, float(lockup), smooth, phase, brake, c_speed, c_type, float(laps)]

    scenarios = [
        # SC deployed: very low speed
        lambda: (
            float(rng.uniform(40.0, 90.0)),   float(rng.uniform(-90.0, -20.0)),
            float(rng.uniform(0.0, 0.9)),     float(rng.uniform(0.40, 0.85)),
            int(rng.integers(0, 4)),           float(rng.uniform(0.3, 0.7)),
            float(rng.uniform(0.0, 1.0)),     float(rng.uniform(150.0, 650.0)),
        ),
        # WARNING: heavy rain + low grip
        lambda: (
            float(rng.uniform(100.0, 200.0)), float(rng.uniform(-70.0, 0.0)),
            float(rng.uniform(0.55, 1.0)),    float(rng.uniform(0.40, 0.62)),
            int(rng.integers(0, 5)),           float(rng.uniform(0.3, 0.7)),
            float(rng.uniform(0.0, 1.0)),     float(rng.uniform(200.0, 650.0)),
        ),
        # WARNING: multi-lockup braking incident
        lambda: (
            float(rng.uniform(130.0, 220.0)), float(rng.uniform(-80.0, -30.0)),
            float(rng.uniform(0.25, 0.6)),    float(rng.uniform(0.45, 0.72)),
            int(rng.integers(3, 6)),           float(rng.uniform(0.3, 0.5)),
            float(rng.uniform(0.0, 1.0)),     float(rng.uniform(500.0, 700.0)),
        ),
        # WATCH: moderate rain
        lambda: (
            float(rng.uniform(180.0, 270.0)), float(rng.uniform(-40.0, 10.0)),
            float(rng.uniform(0.30, 0.55)),   float(rng.uniform(0.60, 0.80)),
            int(rng.integers(0, 3)),           float(rng.uniform(0.4, 0.8)),
            float(rng.uniform(0.0, 1.0)),     float(rng.uniform(200.0, 500.0)),
        ),
        # INFO: normal dry racing
        lambda: (
            float(rng.uniform(200.0, 320.0)), float(rng.uniform(-15.0, 25.0)),
            float(rng.uniform(0.0, 0.15)),    float(rng.uniform(0.75, 1.0)),
            int(rng.integers(0, 2)),           float(rng.uniform(0.65, 1.0)),
            float(rng.uniform(0.1, 0.9)),     float(rng.uniform(150.0, 450.0)),
        ),
    ]

    _circuit_speeds = [140.0, 175.0, 190.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 235.0, 250.0]
    _circuit_types  = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    _circuit_precs  = [0.088, 0.094, 0.119, 0.121, 0.152, 0.346, 0.413, 0.424, 0.440, 0.616]

    for sc_fn in scenarios:
        for _ in range(n_each):
            speed, delta, rain, grip, lockup, smooth, phase, brake = sc_fn()
            c_speed = float(rng.choice(_circuit_speeds))
            c_type  = float(rng.choice(_circuit_types))
            laps    = float(rng.integers(50, 79))
            c_prec  = float(rng.choice(_circuit_precs))
            X.append([speed, delta, rain, grip, float(lockup), smooth, phase, brake, c_speed, c_type, laps, c_prec])
            y.append(_synthetic_label(speed, delta, rain, grip, lockup, smooth, phase, brake))

    # Fill remaining with uniform random
    while len(X) < n:
        speed  = float(rng.uniform(50.0, 320.0))
        delta  = float(rng.uniform(-100.0, 40.0))
        rain   = float(rng.uniform(0.0, 1.0))
        grip   = float(rng.uniform(0.40, 1.0))
        lockup = int(rng.integers(0, 6))
        smooth = float(rng.uniform(0.3, 1.0))
        phase  = float(rng.uniform(0.0, 1.0))
        brake  = float(rng.uniform(100.0, 700.0))
        c_speed = float(rng.choice(_circuit_speeds))
        c_type  = float(rng.choice(_circuit_types))
        laps    = float(rng.integers(50, 79))
        c_prec  = float(rng.choice(_circuit_precs))
        X.append([speed, delta, rain, grip, float(lockup), smooth, phase, brake, c_speed, c_type, laps, c_prec])
        y.append(_synthetic_label(speed, delta, rain, grip, lockup, smooth, phase, brake))

    return np.array(X[:n], dtype=np.float64), np.array(y[:n], dtype=np.int32)


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
                .where(InsightRecord.findings_json.contains('"agent": "safety_car"'))
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
        true_label = pred_label if is_correct else 0
        X.append([
            float(feats.get("mean_speed_kph", 220.0)),
            float(feats.get("speed_delta_kph", 0.0)),
            float(feats.get("rain_intensity", 0.0)),
            float(feats.get("grip_estimate", 0.9)),
            float(feats.get("lockup_count", 0)),
            float(feats.get("throttle_smoothness", 0.8)),
            float(feats.get("race_phase", 0.5)),
            float(feats.get("brake_temp_front_max", 300.0)),
            float(feats.get("circuit_avg_speed_kph", 210.0)),
            float(feats.get("circuit_type_enc", 1.0)),
            float(feats.get("race_laps_total", 57.0)),
            circuit_prec_for_track(ins.track_id or ""),
        ])
        y.append(true_label)

    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def train_from_labels(
    output_path: Path = _CLASSIFIER_PATH,
    real_oversample: int = 20,
    synthetic_n: int = 800,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict:
    X_r, y_r = _load_labeled_from_db()
    n_real = len(y_r)
    X_s, y_s = generate_synthetic(n=max(synthetic_n, n_real * 10))

    from f1di.agents.classifier_utils import blend_with_transfer
    blend = blend_with_transfer(
        lambda: SafetyCarClassifier._build_pipeline(model_type), X_s, y_s, X_r, y_r, n_real,
        _multiclass_brier, weight_cap=real_oversample,
    )
    X, y, sample_weight = blend["X"], blend["y"], blend["sample_weight"]

    from f1di.agents.classifier_utils import class_balance_weights
    if n_real < 10:
        sample_weight = class_balance_weights(y, sample_weight)

    unique, counts = np.unique(y, return_counts=True)
    clf = SafetyCarClassifier(model_type=model_type).fit(X, y, n_real=n_real, sample_weight=sample_weight)
    clf.real_sample_weight = blend["real_weight"]
    clf.prior_cv_accuracy = blend["prior_cv"]["cv_accuracy"] if blend["prior_cv"] else None

    from f1di.agents.classifier_utils import save_with_snapshot, record_history
    snap = save_with_snapshot(clf, output_path)
    record_history(
        clf, agent="safety_car",
        versioned_path=snap["versioned_path"],
        blocked=snap["blocked"],
        history_path=output_path.parent / "model_history.json",
        threshold=snap.get("threshold"),
    )
    return {
        "n_synthetic": len(y_s), "n_real": n_real, "n_total": len(y),
        "accuracy": round(clf.accuracy, 4), "classes": clf.classes_,
        "class_distribution": {_LABEL_MAP[int(k)]: int(v) for k, v in zip(unique, counts)},
        "per_class": clf.cv_per_class,
        "output_path": str(output_path),
        "snapshot_blocked": snap["blocked"],
        "versioned_path": snap["versioned_path"],
        "real_sample_weight": round(clf.real_sample_weight, 4) if clf.real_sample_weight is not None else None,
        "prior_accuracy": round(clf.prior_cv_accuracy, 4) if clf.prior_cv_accuracy is not None else None,
        "transfer_lift": round(clf.accuracy - clf.prior_cv_accuracy, 4) if clf.prior_cv_accuracy is not None else None,
    }
