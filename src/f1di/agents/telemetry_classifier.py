"""Logistic regression classifier for telemetry risk level.

Four classes: INFO (0), WATCH (1), WARNING (2), CRITICAL (3).
Same cold-start pattern as tire_classifier: synthetic rule-distilled data
blended with real flywheel labels once ≥10 labeled examples exist.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from f1di.agents.classifier_utils import _CALIBRATION_DIR, circuit_prec_for_track

logger = logging.getLogger("f1di.agents.telemetry_classifier")
_CLASSIFIER_PATH = _CALIBRATION_DIR / "telemetry_classifier.pkl"

FEATURE_NAMES: list[str] = [
    "brake_temp_front_max",
    "lockup_count",
    "brake_fade_risk",
    "fl_degradation_pressure",
    "fl_wear_slope",
    "fr_wear_slope",
    "crosswind_proxy",
    "race_phase",
    "laps_remaining",
    "circuit_avg_speed_kph",
    "circuit_type_enc",
    "race_laps_total",
    "circuit_precision_prior",
]

_LABEL_MAP: dict[int, str] = {0: "INFO", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"}
_LABEL_INV: dict[str, int] = {v: k for k, v in _LABEL_MAP.items()}

MODEL_VERSION = "lr-v1"
MODEL_TYPE = "LogisticRegression"
DEFAULT_MODEL_TYPE = "logistic"


def _multiclass_brier(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    n, n_c = len(y), len(classes)
    cls_idx = {int(c): i for i, c in enumerate(classes)}
    Y_oh = np.zeros((n, n_c), dtype=np.float64)
    for i, yi in enumerate(y):
        Y_oh[i, cls_idx[int(yi)]] = 1.0
    return float(np.mean(np.sum((proba - Y_oh) ** 2, axis=1)))


def features_to_array(features) -> np.ndarray:
    return np.array([
        features.brake_temp_front_max,
        float(features.lockup_count),
        features.brake_fade_risk,
        features.fl_degradation_pressure,
        features.fl_wear_slope,
        features.fr_wear_slope,
        features.crosswind_proxy,
        features.race_phase,
        features.laps_remaining,
        features.circuit_avg_speed_kph,
        features.circuit_type_enc,
        features.race_laps_total,
        features.circuit_precision_prior,
    ], dtype=np.float64)


class TelemetryClassifier:
    """Classifier for telemetry risk (INFO/WATCH/WARNING/CRITICAL)."""

    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE) -> None:
        from f1di.agents.classifier_utils import build_model, _MODEL_DISPLAY, _MODEL_VERSION
        self._scaler, self._model = build_model(model_type, max_depth=4, agent="telemetry")
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

    def fit(self, X: np.ndarray, y: np.ndarray, n_real: int = 0, sample_weight: np.ndarray | None = None) -> "TelemetryClassifier":
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
            "TelemetryClassifier fitted: n=%d n_real=%d cv_acc=%.3f cv_brier=%.4f n_splits=%d classes=%s",
            self.n_train, self.n_real, self.accuracy, self.brier_score, self.cv_n_splits, self.classes_,
        )
        return self

    @staticmethod
    def _build_pipeline(model_type: str = DEFAULT_MODEL_TYPE):
        from f1di.agents.classifier_utils import build_model
        return build_model(model_type, max_depth=4, agent="telemetry")

    def ood_score(self, features) -> float:
        x = features_to_array(features)
        z = np.abs((x - self._scaler.mean_) / np.maximum(self._scaler.scale_, 1e-8))
        return float(z.max())

    def predict(self, features) -> tuple[str, float, np.ndarray]:
        """Return (risk_label, confidence, proba_array)."""
        x = features_to_array(features).reshape(1, -1)
        x_s = self._scaler.transform(x)
        proba = self._model.predict_proba(x_s)[0]
        idx = int(np.argmax(proba))
        return _LABEL_MAP[idx], float(proba[idx]), proba

    def save(self, path: Path = _CLASSIFIER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path = _CLASSIFIER_PATH) -> "TelemetryClassifier":
        return pickle.loads(path.read_bytes())


# ── Synthetic data generation (cold-start) ─────────────────────────────────

def _synthetic_label(
    brake_temp: float,
    lockups: float,
    brake_fade: float,
    fl_dp: float,
    fl_slope: float,
    crosswind: float,
) -> int:
    from f1di.agents.thresholds import CircuitThresholds
    t = CircuitThresholds()
    if lockups >= 5:
        return 3  # CRITICAL
    if brake_temp > t.brake_temp_critical_c or lockups >= 3:
        return 2  # WARNING — matches agent safety floor
    if fl_dp > t.fl_degradation_pressure_critical or fl_slope > 0.008:
        return 2  # WARNING
    if brake_fade > 12.0 or crosswind > t.crosswind_watch * 0.85:
        return 1  # WATCH
    return 0  # INFO


def generate_synthetic(n: int = 800, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    while len(X) < n:
        # Stratify 1-in-5 samples as "brake_temp-only WARNING" so LR learns this path
        if len(X) % 5 == 0:
            brake_temp = float(rng.uniform(920.0, 1100.0))
            lockups    = float(rng.integers(0, 3))  # low lockup, brake_temp is the trigger
        else:
            brake_temp = float(rng.uniform(200.0, 1100.0))
            lockups    = float(rng.integers(0, 7))
        brake_fade = float(rng.uniform(0.0, 20.0))
        fl_dp      = float(rng.uniform(0.20, 0.95))
        fl_slope   = float(rng.uniform(0.0, 0.020))
        fr_slope   = float(rng.uniform(0.0, 0.020))
        crosswind  = float(rng.uniform(0.0, 30.0))
        phase      = float(rng.uniform(0.0, 1.0))
        laps_r     = float(rng.uniform(0.0, 45.0))
        circuit_speed = float(rng.choice([140.0, 175.0, 190.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 235.0, 250.0]))
        circuit_type  = float(rng.choice([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        race_laps     = float(rng.integers(50, 79))
        circuit_prec  = float(rng.choice([0.088, 0.094, 0.119, 0.121, 0.152, 0.346, 0.413, 0.424, 0.440, 0.616]))
        label = _synthetic_label(brake_temp, lockups, brake_fade, fl_dp, fl_slope, crosswind)
        X.append([brake_temp, lockups, brake_fade, fl_dp, fl_slope, fr_slope, crosswind, phase, laps_r, circuit_speed, circuit_type, race_laps, circuit_prec])
        y.append(label)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


# ── Real data from DB ──────────────────────────────────────────────────────

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
        logger.warning("telemetry_classifier DB query failed: %s", exc)
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
        tel = next((f for f in findings if f.get("agent") == "telemetry"), None)
        if tel is None:
            continue
        feats = tel.get("features", {})
        pred_label = _LABEL_INV.get(tel.get("risk", ins.risk), 0)
        true_label = pred_label if is_correct else max(0, pred_label - 1)
        X.append([
            float(feats.get("brake_temp_front_max", 400.0)),
            float(feats.get("lockup_count", 0.0)),
            float(feats.get("brake_fade_risk", 2.0)),
            float(feats.get("fl_degradation_pressure", 0.35)),
            float(feats.get("fl_wear_slope", 0.001)),
            float(feats.get("fr_wear_slope", 0.001)),
            float(feats.get("crosswind_proxy", 5.0)),
            float(feats.get("race_phase", 0.5)),
            float(feats.get("laps_remaining", 20.0)),
            float(feats.get("circuit_avg_speed_kph", 210.0)),
            float(feats.get("circuit_type_enc", 1.0)),
            float(feats.get("race_laps_total", 57.0)),
            circuit_prec_for_track(ins.track_id or ""),
        ])
        y.append(true_label)

    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


# ── Training entry point ───────────────────────────────────────────────────

def train_from_labels(
    output_path: Path = _CLASSIFIER_PATH,
    real_oversample: int = 20,
    synthetic_n: int = 800,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict:
    X_s, y_s = generate_synthetic(n=synthetic_n)
    X_r, y_r = _load_labeled_from_db()
    n_real = len(y_r)

    from f1di.agents.classifier_utils import blend_with_transfer
    blend = blend_with_transfer(
        lambda: TelemetryClassifier._build_pipeline(model_type), X_s, y_s, X_r, y_r, n_real,
        _multiclass_brier, weight_cap=real_oversample,
    )
    X, y, sample_weight = blend["X"], blend["y"], blend["sample_weight"]
    if 0 < n_real < 10:
        logger.info(
            "telemetry_classifier: %d real examples (min 10 to blend) — "
            "training on synthetic only this cycle", n_real,
        )

    from f1di.agents.classifier_utils import class_balance_weights
    if n_real < 10:
        sample_weight = class_balance_weights(y, sample_weight)

    unique, counts = np.unique(y, return_counts=True)
    clf = TelemetryClassifier(model_type=model_type).fit(X, y, n_real=n_real, sample_weight=sample_weight)
    clf.real_sample_weight = blend["real_weight"]
    clf.prior_cv_accuracy = blend["prior_cv"]["cv_accuracy"] if blend["prior_cv"] else None

    from f1di.agents.classifier_utils import save_with_snapshot, record_history, per_class_report, cross_val_eval
    snap = save_with_snapshot(clf, output_path)
    record_history(clf, agent="telemetry", versioned_path=snap["versioned_path"], blocked=snap["blocked"], history_path=output_path.parent / "model_history.json", threshold=snap.get("threshold"))
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


_INCREMENTAL_PATH = _CALIBRATION_DIR / "telemetry_incremental.pkl"


def partial_fit_from_labels(output_path: Path = _INCREMENTAL_PATH) -> dict:
    """Incrementally update an SGDClassifier with new real labels (warm-start)."""
    import pickle as _pickle
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    X_real, y_real = _load_labeled_from_db()
    if len(y_real) < 4:
        return {"skipped": True, "reason": "< 4 real labels"}

    all_classes = np.array(sorted(_LABEL_MAP.keys()), dtype=np.int32)

    clf, scaler = None, None
    if output_path.exists():
        try:
            with open(output_path, "rb") as fh:
                clf, scaler = _pickle.load(fh)
        except Exception:
            clf, scaler = None, None

    if clf is None:
        scaler = StandardScaler()
        clf = SGDClassifier(loss="log_loss", random_state=42, max_iter=1)
        X_syn, y_syn = generate_synthetic(n=400, seed=0)
        X_all = np.vstack([X_syn, X_real])
        y_all = np.concatenate([y_syn, y_real])
        clf.partial_fit(scaler.fit_transform(X_all), y_all, classes=all_classes)
    else:
        clf.partial_fit(scaler.transform(X_real), y_real)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        _pickle.dump((clf, scaler), fh)

    acc = float(accuracy_score(y_real, clf.predict(scaler.transform(X_real))))
    return {"n_real": len(y_real), "accuracy": acc, "incremental": True}
