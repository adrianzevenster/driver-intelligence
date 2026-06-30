"""Logistic regression classifier for fuel strategy risk level.

Three classes: INFO (0), WATCH (1), WARNING (2).
WARNING = active fuel-save mode required to reach the end.
WATCH   = consumption rate elevated; monitor and consider lift-and-coast.
INFO    = fuel load on plan; no intervention needed.

Features are derived entirely from existing RaceFeatures so no schema changes
are required beyond the throttle_mean and ers_net_deploy_kw fields added to
the extractor.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from f1di.agents.classifier_utils import _CALIBRATION_DIR, circuit_prec_for_track

logger = logging.getLogger("f1di.agents.fuel_classifier")
_CLASSIFIER_PATH = _CALIBRATION_DIR / "fuel_classifier.pkl"

FEATURE_NAMES: list[str] = [
    "throttle_mean",
    "ers_net_deploy_kw",
    "battery_soc",
    "laps_remaining",
    "race_phase",
    "stint_fraction",
    "throttle_smoothness",
    "circuit_avg_speed_kph",
    "circuit_type_enc",
    "race_laps_total",
    "circuit_precision_prior",
]

_LABEL_MAP: dict[int, str] = {0: "INFO", 1: "WATCH", 2: "WARNING"}
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
        features.throttle_mean,
        features.ers_net_deploy_kw,
        features.battery_soc,
        features.laps_remaining,
        features.race_phase,
        features.stint_fraction,
        features.throttle_smoothness,
        features.circuit_avg_speed_kph,
        features.circuit_type_enc,
        features.race_laps_total,
        features.circuit_precision_prior,
    ], dtype=np.float64)


class FuelClassifier:
    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE) -> None:
        from f1di.agents.classifier_utils import build_model, _MODEL_DISPLAY, _MODEL_VERSION
        self._scaler, self._model = build_model(model_type, max_depth=4, agent="fuel")
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

    def fit(self, X: np.ndarray, y: np.ndarray, n_real: int = 0, sample_weight: np.ndarray | None = None) -> "FuelClassifier":
        from sklearn.metrics import accuracy_score
        X_s = self._scaler.fit_transform(X)
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
            "FuelClassifier fitted: n=%d n_real=%d cv_acc=%.3f cv_brier=%.4f n_splits=%d",
            self.n_train, self.n_real, self.accuracy, self.brier_score, self.cv_n_splits,
        )
        return self

    @staticmethod
    def _build_pipeline(model_type: str = DEFAULT_MODEL_TYPE):
        from f1di.agents.classifier_utils import build_model
        return build_model(model_type, max_depth=4, agent="fuel")

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
    def load(path: Path = _CLASSIFIER_PATH) -> "FuelClassifier":
        return pickle.loads(path.read_bytes())


def _synthetic_label(
    throttle_mean: float,
    ers_net_kw: float,
    battery_soc: float,
    laps_remaining: float,
    race_phase: float,
    smoothness: float,
    circuit_avg_speed: float,
    stint_fraction: float,
) -> int:
    # Composite fuel pressure: high throttle − ERS offset − SOC buffer.
    # Higher = burning faster than the ERS + battery can compensate.
    fuel_pressure = (throttle_mean / 100.0) - (ers_net_kw / 500.0) - (battery_soc * 0.15)

    # High-speed circuits burn more fuel — amplify pressure proportionally
    speed_factor = 1.0 + (circuit_avg_speed - 210.0) / 400.0  # +/- 10% for 40 kph spread
    effective_pressure = fuel_pressure * speed_factor

    # Long stint with high consumption + ERS deficit
    ers_deficit = max(0.0, -ers_net_kw / 100.0)  # negative ERS = driver is harvesting, not deploying
    if effective_pressure > 0.62 and laps_remaining > 12 and smoothness < 0.62:
        return 2  # WARNING — fuel save required
    if effective_pressure > 0.52 and laps_remaining > 10:
        return 2  # WARNING
    if effective_pressure > 0.48 and laps_remaining > 8 and ers_deficit > 0.2:
        return 2  # WARNING — ERS deficit compounds fuel issue
    if effective_pressure > 0.38 and laps_remaining > 6:
        return 1  # WATCH
    if race_phase < 0.22 and throttle_mean > 82 and battery_soc < 0.60:
        return 1  # WATCH — opening laps, full fuel load, already draining battery
    if stint_fraction > 0.85 and effective_pressure > 0.32:
        return 1  # WATCH — deep into stint, fuel model approaching margin
    return 0  # INFO


def generate_synthetic(n: int = 900, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    while len(X) < n:
        throttle  = float(rng.uniform(55.0, 95.0))
        ers_net   = float(rng.uniform(-20.0, 120.0))
        soc       = float(rng.uniform(0.15, 0.95))
        laps_rem  = float(rng.uniform(0.0, 50.0))
        phase     = float(rng.uniform(0.0, 1.0))
        stint_fr  = float(rng.uniform(0.0, 1.3))
        smoothness = float(rng.uniform(0.3, 1.0))
        circuit_speed = float(rng.choice([140.0, 175.0, 190.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 235.0, 250.0]))
        circuit_type  = float(rng.choice([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        race_laps     = float(rng.integers(50, 79))
        circuit_prec  = float(rng.choice([0.088, 0.094, 0.119, 0.121, 0.152, 0.346, 0.413, 0.424, 0.440, 0.616]))
        X.append([throttle, ers_net, soc, laps_rem, phase, stint_fr, smoothness, circuit_speed, circuit_type, race_laps, circuit_prec])
        y.append(_synthetic_label(throttle, ers_net, soc, laps_rem, phase, smoothness, circuit_speed, stint_fr))
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
        logger.warning("fuel_classifier DB query failed: %s", exc)
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
        fuel = next((f for f in findings if f.get("agent") == "fuel"), None)
        if fuel is None:
            continue
        feats = fuel.get("features", {})
        pred_label = _LABEL_INV.get(fuel.get("risk", ins.risk), 0)
        true_label = pred_label if is_correct else max(0, pred_label - 1)
        X.append([
            float(feats.get("throttle_mean", 72.0)),
            float(feats.get("ers_net_deploy_kw", 40.0)),
            float(feats.get("battery_soc", 0.6)),
            float(feats.get("laps_remaining", 20.0)),
            float(feats.get("race_phase", 0.5)),
            float(feats.get("stint_fraction", 0.5)),
            float(feats.get("throttle_smoothness", 0.8)),
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
    synthetic_n: int = 600,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict:
    X_s, y_s = generate_synthetic(n=synthetic_n)
    X_r, y_r = _load_labeled_from_db()
    n_real = len(y_r)

    from f1di.agents.classifier_utils import blend_with_transfer
    blend = blend_with_transfer(
        lambda: FuelClassifier._build_pipeline(model_type), X_s, y_s, X_r, y_r, n_real,
        _multiclass_brier, weight_cap=real_oversample,
    )
    X, y, sample_weight = blend["X"], blend["y"], blend["sample_weight"]

    from f1di.agents.classifier_utils import class_balance_weights
    if n_real < 10:
        sample_weight = class_balance_weights(y, sample_weight)

    unique, counts = np.unique(y, return_counts=True)
    clf = FuelClassifier(model_type=model_type).fit(X, y, n_real=n_real, sample_weight=sample_weight)
    clf.real_sample_weight = blend["real_weight"]
    clf.prior_cv_accuracy = blend["prior_cv"]["cv_accuracy"] if blend["prior_cv"] else None

    from f1di.agents.classifier_utils import save_with_snapshot, record_history, per_class_report, cross_val_eval
    snap = save_with_snapshot(clf, output_path)
    record_history(
        clf, agent="fuel",
        versioned_path=snap["versioned_path"],
        blocked=snap["blocked"],
        history_path=output_path.parent / "model_history.json",
        threshold=snap.get("threshold"),
    )
    _cv = cross_val_eval(clf._build_pipeline, X, y, _multiclass_brier, sample_weight=sample_weight, collect_predictions=True)
    return {
        "n_synthetic": len(y_s), "n_real": n_real, "n_total": len(y),
        "accuracy": round(clf.accuracy, 4), "classes": clf.classes_,
        "class_distribution": {_LABEL_MAP[int(k)]: int(v) for k, v in zip(unique, counts)},
        "per_class": per_class_report(_cv, _LABEL_MAP),
        "output_path": str(output_path),
        "snapshot_blocked": snap["blocked"],
        "versioned_path": snap["versioned_path"],
        "real_sample_weight": round(clf.real_sample_weight, 4) if clf.real_sample_weight is not None else None,
        "prior_accuracy": round(clf.prior_cv_accuracy, 4) if clf.prior_cv_accuracy is not None else None,
        "transfer_lift": round(clf.accuracy - clf.prior_cv_accuracy, 4) if clf.prior_cv_accuracy is not None else None,
    }
