"""Fusion meta-learner: predicts P(insight correct) from all 6 agent outputs.

Replaces the single isotonic-calibrated confidence with a richer estimate that
captures agent interaction effects (e.g. single-agent WARNING with others on
INFO is less reliable than multi-agent agreement).

Only activates in inference when n_real >= 20 to avoid degrading calibration
with a synthetic-only model.  Below that threshold the isotonic calibrator is
used unchanged.

Architecture:
    Input (16 features):
        {agent}_risk × 6       RISK_WEIGHT values for all agents
        {agent}_conf × 6       per-agent confidence
        n_high_risk            count of agents at WARNING or CRITICAL
        risk_agreement         1 - normalised std of risk weights (0=disagree, 1=agree)
        iso_confidence         isotonic calibrator output
        max_risk_weight        highest single-agent risk weight
    Target: 1 = insight was correct, 0 = incorrect (from FeedbackRecord)
    Model: HistGradientBoostingClassifier binary — captures non-linear interactions
           between agent agreement patterns that a linear model misses.
           StandardScaler retained for OOD detection (ood_score uses mean_/scale_).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger("f1di.inference.meta_learner")

from f1di.agents.classifier_utils import _CALIBRATION_DIR
_META_PATH = _CALIBRATION_DIR / "meta_learner.pkl"

# All 6 inference agents — order is part of the feature contract
_AGENT_ORDER = ["telemetry", "tire_strategy", "weather", "battery", "safety_car", "fuel"]

FEATURE_NAMES: list[str] = [
    "telemetry_risk", "tire_risk", "weather_risk", "battery_risk", "safety_car_risk", "fuel_risk",
    "telemetry_conf", "tire_conf", "weather_conf", "battery_conf", "safety_car_conf", "fuel_conf",
    "n_high_risk",
    "risk_agreement",
    "iso_confidence",
    "max_risk_weight",
]

# Maximum possible std across 6 risk weights using actual RISK_WEIGHT scale
# [0.25, 0.45, 0.7, 0.9] — max std at equal split between 0.9 and 0.25: (0.9-0.25)/2 = 0.325
_MAX_RW_STD = 0.325


def _binary_brier(proba: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> float:
    p_correct_idx = int(np.where(classes == 1)[0][0])
    return float(np.mean((proba[:, p_correct_idx] - y_true.astype(np.float64)) ** 2))


def findings_to_array(findings: list, iso_confidence: float) -> np.ndarray:
    """Pack all 6-agent findings + derived features into the meta-learner feature vector."""
    from f1di.confidence.calibration import RISK_WEIGHT
    from f1di.domain.schemas import RiskLevel

    _HIGH_RISK = {RiskLevel.WARNING, RiskLevel.CRITICAL}
    by_agent = {f.agent if hasattr(f, "agent") else f.get("agent"): f for f in findings}

    risk_weights: list[float] = []
    confs: list[float] = []
    n_high = 0
    _info_rw = float(RISK_WEIGHT.get(RiskLevel.INFO, 0.25))
    for agent in _AGENT_ORDER:
        f = by_agent.get(agent)
        if f is None:
            rw, conf = _info_rw, 0.5
        elif hasattr(f, "risk"):
            rw = float(RISK_WEIGHT.get(f.risk, _info_rw))
            conf = float(f.confidence)
            if f.risk in _HIGH_RISK:
                n_high += 1
        else:
            rl = RiskLevel[f.get("risk", "INFO")]
            rw = float(RISK_WEIGHT.get(rl, _info_rw))
            conf = float(f.get("confidence", 0.5))
            if rl in _HIGH_RISK:
                n_high += 1
        risk_weights.append(rw)
        confs.append(conf)

    rw_arr = np.array(risk_weights)
    agreement = max(0.0, 1.0 - float(np.std(rw_arr)) / _MAX_RW_STD)
    max_rw = float(rw_arr.max())

    return np.array(
        risk_weights + confs + [float(n_high), agreement, iso_confidence, max_rw],
        dtype=np.float64,
    )


MODEL_VERSION = "hgb-v1"
MODEL_TYPE = "HistGradientBoosting"
DEFAULT_MODEL_TYPE = "hgbc"


class MetaLearner:
    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE) -> None:
        from f1di.agents.classifier_utils import build_model, _MODEL_DISPLAY, _MODEL_VERSION
        self._scaler, self._model = build_model(model_type, max_depth=3, agent="meta")
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

    def fit(self, X: np.ndarray, y: np.ndarray, n_real: int = 0, sample_weight: np.ndarray | None = None) -> "MetaLearner":
        from sklearn.metrics import accuracy_score
        X_s = self._scaler.fit_transform(X)
        self._model.fit(X_s, y, sample_weight=sample_weight)
        self.n_train = int(len(y))
        self.n_real = n_real

        from f1di.agents.classifier_utils import cross_val_eval
        from sklearn.metrics import balanced_accuracy_score
        cv = cross_val_eval(
            self._build_pipeline, X, y, _binary_brier,
            sample_weight=sample_weight, scoring_fn=balanced_accuracy_score,
        )
        if cv is not None:
            self.accuracy = cv["cv_accuracy"]
            self.brier_score = cv["cv_brier"]
            self.cv_n_splits = cv["n_splits"]
            self.cv_accuracy_std = cv["cv_accuracy_std"]
            self.cv_brier_std = cv["cv_brier_std"]
            self.cv_fold_accuracies = cv["fold_accuracies"]
            self.cv_fold_briers = cv["fold_briers"]
        else:
            proba = self._model.predict_proba(X_s)  # (N, 2): [P(incorrect), P(correct)]
            self.accuracy = float(accuracy_score(y, self._model.predict(X_s)))
            self.brier_score = float(np.mean((proba[:, 1] - y.astype(np.float64)) ** 2))
            self.cv_n_splits = 0
            self.cv_accuracy_std = None
            self.cv_brier_std = None
            self.cv_fold_accuracies = None
            self.cv_fold_briers = None

        logger.info(
            "MetaLearner fitted: n=%d n_real=%d cv_acc=%.3f cv_brier=%.4f n_splits=%d",
            self.n_train, self.n_real, self.accuracy, self.brier_score, self.cv_n_splits,
        )
        return self

    @staticmethod
    def _build_pipeline(model_type: str = DEFAULT_MODEL_TYPE):
        from f1di.agents.classifier_utils import build_model
        return build_model(model_type, max_depth=3, agent="meta")

    def ood_score(self, findings: list, iso_confidence: float) -> float:
        """Max absolute Z-score of meta-learner features vs training distribution."""
        x = findings_to_array(findings, iso_confidence)
        z = np.abs((x - self._scaler.mean_) / np.maximum(self._scaler.scale_, 1e-8))
        return float(z.max())

    def predict_confidence(self, findings: list, iso_confidence: float) -> float:
        """Return P(correct) — replaces iso_confidence when n_real >= 20."""
        x = findings_to_array(findings, iso_confidence).reshape(1, -1)
        x_s = self._scaler.transform(x)
        # index 1 = P(correct=1)
        return float(self._model.predict_proba(x_s)[0][1])

    def get_feature_importances(self) -> dict[str, float]:
        """Return normalized feature importance for each input dimension.

        Uses coef_ for LogisticRegression, feature_importances_ for tree models that
        expose it (RF, GBT), and permutation importance for HGBT (which lacks a native
        gini/gain summary).  Values normalize to sum=1.0.
        """
        raw = None
        try:
            raw = np.abs(self._model.coef_[0])
        except AttributeError:
            pass

        if raw is None:
            try:
                raw = np.array(self._model.feature_importances_)
            except AttributeError:
                pass

        if raw is None:
            try:
                from sklearn.inspection import permutation_importance
                X_syn, y_syn = generate_synthetic(n=200, seed=99)
                X_s = self._scaler.transform(X_syn)
                r = permutation_importance(
                    self._model, X_s, y_syn, n_repeats=5, random_state=0, n_jobs=1,
                )
                raw = np.maximum(r.importances_mean, 0.0)
            except Exception:
                return {}

        total = float(raw.sum()) or 1.0
        names = FEATURE_NAMES[: len(raw)]
        return {name: round(float(v) / total, 4) for name, v in zip(names, raw)}

    def save(self, path: Path = _META_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path = _META_PATH) -> "MetaLearner":
        return pickle.loads(path.read_bytes())


# ── Synthetic cold-start data ──────────────────────────────────────────────

def _synthetic_label(
    risk_weights: list[float],
    confs: list[float],
    rng,
) -> int:
    """Synthetic ground truth: 1 = insight was correct, 0 = incorrect.

    Labels are based on n_high (agent consensus count) and firing agent
    confidence. iso_conf is intentionally excluded so the HGBC must learn
    agent interaction effects directly rather than echoing the isotonic
    calibrator.

    Decision rule:
      n_high >= 4 : almost always correct (5% noise)
      n_high == 3 : reliable when firing agents are confident (mean >= 0.62)
      n_high == 2 : reliable only when both are clearly confident (mean >= 0.70)
      n_high == 1 : solo false-alarm risk; correct only with high confidence (>= 0.79)
      n_high == 0 : INFO prediction — reliable by default (consensus silence)
    """
    from f1di.domain.schemas import RiskLevel
    from f1di.confidence.calibration import RISK_WEIGHT
    _HIGH = float(RISK_WEIGHT[RiskLevel.WARNING])

    rw_arr = np.array(risk_weights)
    n_high = int((rw_arr >= _HIGH).sum())

    if n_high >= 4:
        return 0 if rng.random() < 0.05 else 1

    if n_high == 3:
        firing_confs = [c for c, rw in zip(confs, risk_weights) if rw >= _HIGH]
        return 1 if float(np.mean(firing_confs)) >= 0.62 else 0

    if n_high == 2:
        firing_confs = [c for c, rw in zip(confs, risk_weights) if rw >= _HIGH]
        return 1 if float(np.mean(firing_confs)) >= 0.70 else 0

    if n_high == 1:
        firing_conf = max((c for c, rw in zip(confs, risk_weights) if rw >= _HIGH), default=0.5)
        return 1 if firing_conf >= 0.79 else 0

    # n_high == 0: agents agree nothing is wrong → correct by default
    return 1


def generate_synthetic(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    from f1di.confidence.calibration import RISK_WEIGHT
    from f1di.domain.schemas import RiskLevel
    rng = np.random.default_rng(seed)

    _rw_vals = [float(RISK_WEIGHT[r]) for r in RiskLevel]
    _high_thresh = float(RISK_WEIGHT[RiskLevel.WARNING])

    X, y = [], []
    # Structured scenario batches ensure coverage of adversarial cases
    n_each = max(1, n // 7)

    scenarios = [
        # Strong consensus (4-6 agents agree) — iso can be anything
        {"n_high_range": (4, 6), "iso_range": (0.20, 0.95)},
        # 3-agent consensus with LOW iso (tests iso can be overridden)
        {"n_high_range": (3, 3), "iso_range": (0.10, 0.50)},
        # 3-agent consensus with HIGH iso (agreement + iso → strong)
        {"n_high_range": (3, 3), "iso_range": (0.55, 0.95)},
        # Solo agent with HIGH agent confidence, LOW iso (specialist fires alone)
        {"n_high_range": (1, 1), "iso_range": (0.20, 0.55)},
        # No agents firing but HIGH iso (iso over-confident, agents disagree)
        {"n_high_range": (0, 0), "iso_range": (0.60, 0.95)},
        # All quiet, iso also quiet (straightforward no-risk)
        {"n_high_range": (0, 0), "iso_range": (0.05, 0.40)},
        # Mixed / random remainder
        {"n_high_range": (0, 6), "iso_range": (0.10, 0.95)},
    ]

    for sc in scenarios:
        lo_h, hi_h = sc["n_high_range"]
        lo_i, hi_i = sc["iso_range"]
        for _ in range(n_each):
            n_h_target = int(rng.integers(lo_h, hi_h + 1))
            indices = list(range(6))
            rng.shuffle(indices)
            rws = [float(_rw_vals[0])] * 6
            for i in range(min(n_h_target, 6)):
                rws[indices[i]] = float(rng.choice([_rw_vals[2], _rw_vals[3]]))
            for i in range(min(n_h_target, 6), 6):
                rws[indices[i]] = float(rng.choice([_rw_vals[0], _rw_vals[1]]))
            confs = [float(rng.uniform(0.45, 0.92)) for _ in range(6)]
            iso = float(rng.uniform(lo_i, hi_i))
            rw_arr = np.array(rws)
            n_high = int((rw_arr >= _high_thresh).sum())
            agreement = max(0.0, 1.0 - float(np.std(rw_arr)) / _MAX_RW_STD)
            max_rw = float(rw_arr.max())
            X.append(rws + confs + [float(n_high), agreement, iso, max_rw])
            y.append(_synthetic_label(rws, confs, rng))

    # Fill remainder with random samples
    while len(X) < n:
        rws   = [float(rng.choice(_rw_vals)) for _ in _AGENT_ORDER]
        confs = [float(rng.uniform(0.45, 0.92)) for _ in _AGENT_ORDER]
        iso   = float(rng.uniform(0.20, 0.95))
        rw_arr    = np.array(rws)
        n_high    = int((rw_arr >= _high_thresh).sum())
        agreement = max(0.0, 1.0 - float(np.std(rw_arr)) / _MAX_RW_STD)
        max_rw    = float(rw_arr.max())
        X.append(rws + confs + [float(n_high), agreement, iso, max_rw])
        y.append(_synthetic_label(rws, confs, rng))

    return np.array(X[:n], dtype=np.float64), np.array(y[:n], dtype=np.int32)


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
                .where(InsightRecord.shadow.is_(False))
            ).all()
    except Exception as exc:
        logger.warning("meta_learner DB query failed: %s", exc)
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)

    for fb, ins in rows:
        if ins is None:
            continue
        if fb.correct is not None:
            label = 1 if bool(fb.correct) else 0
        elif fb.rating is not None:
            label = 1 if int(fb.rating) >= 4 else 0
        else:
            continue

        try:
            findings_raw = _json.loads(ins.findings_json or "[]")
        except Exception:
            continue

        iso_conf = float(ins.confidence)

        from f1di.confidence.calibration import RISK_WEIGHT
        from f1di.domain.schemas import RiskLevel
        _HIGH_RISK = {RiskLevel.WARNING, RiskLevel.CRITICAL}
        _info_rw = float(RISK_WEIGHT.get(RiskLevel.INFO, 0.25))
        by_agent = {f.get("agent"): f for f in findings_raw}
        risk_weights, confs = [], []
        n_high = 0
        for agent in _AGENT_ORDER:
            f = by_agent.get(agent)
            if f is None:
                risk_weights.append(_info_rw)
                confs.append(0.5)
            else:
                try:
                    rl = RiskLevel[f.get("risk", "INFO")]
                    rw = float(RISK_WEIGHT[rl])
                    if rl in _HIGH_RISK:
                        n_high += 1
                except (KeyError, ValueError):
                    rw = 0.0
                risk_weights.append(rw)
                confs.append(float(f.get("confidence", 0.5)))

        rw_arr = np.array(risk_weights)
        agreement = max(0.0, 1.0 - float(np.std(rw_arr)) / _MAX_RW_STD)
        max_rw = float(rw_arr.max())
        X.append(risk_weights + confs + [float(n_high), agreement, iso_conf, max_rw])
        y.append(label)

    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


# ── Training entry point ───────────────────────────────────────────────────

def train_from_labels(
    output_path: Path = _META_PATH,
    real_oversample: int = 5,
    synthetic_n: int = 800,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict:
    """Train MetaLearner combining synthetic + real flywheel labels."""
    X_s, y_s = generate_synthetic(n=synthetic_n)
    X_r, y_r = _load_labeled_from_db()
    n_real = len(y_r)

    from f1di.agents.classifier_utils import blend_with_transfer
    blend = blend_with_transfer(
        lambda: MetaLearner._build_pipeline(model_type),
        X_s, y_s, X_r, y_r, n_real,
        _binary_brier, weight_cap=real_oversample,
    )
    X, y, sample_weight = blend["X"], blend["y"], blend["sample_weight"]

    # Upweight the minority (correct=1) class to counter the ~4:1 imbalance
    # produced by safety-car incidents being attributed to all 20 drivers.
    unique_full, counts_full = np.unique(y, return_counts=True)
    if len(counts_full) == 2 and counts_full[0] > 0 and counts_full[1] > 0:
        ratio = counts_full[0] / counts_full[1]  # incorrect / correct
        class_sw = np.where(y == 1, ratio, 1.0)
        sw_base = sample_weight if sample_weight is not None else np.ones(len(y))
        sample_weight = sw_base * class_sw

    unique, counts = np.unique(y, return_counts=True)
    meta = MetaLearner(model_type=model_type).fit(X, y, n_real=n_real, sample_weight=sample_weight)
    meta.real_sample_weight = blend["real_weight"]
    meta.prior_cv_accuracy = blend["prior_cv"]["cv_accuracy"] if blend["prior_cv"] else None

    from f1di.agents.classifier_utils import save_with_snapshot
    # First time real data is added the accuracy will naturally drop vs a
    # synthetic-only model (synthetic is overfit to its own distribution).
    # Allow up to 5 pp regression when the live model has zero real labels.
    import pickle as _pkl
    _prev_n_real = 0
    if output_path.exists():
        try:
            _prev_n_real = int(_pkl.loads(output_path.read_bytes()).n_real)
        except Exception:
            pass
    _delta = 0.05 if _prev_n_real == 0 else 0.02
    snap = save_with_snapshot(meta, output_path, min_accuracy_delta=_delta)
    return {
        "n_synthetic": len(y_s), "n_real": n_real, "n_total": len(y),
        "accuracy": round(meta.accuracy, 4),
        "active_in_inference": n_real >= 100,
        "class_distribution": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "output_path": str(output_path),
        "snapshot_blocked": snap["blocked"],
        "versioned_path": snap["versioned_path"],
        "real_sample_weight": round(meta.real_sample_weight, 4) if meta.real_sample_weight is not None else None,
        "prior_accuracy": round(meta.prior_cv_accuracy, 4) if meta.prior_cv_accuracy is not None else None,
        "transfer_lift": round(meta.accuracy - meta.prior_cv_accuracy, 4) if meta.prior_cv_accuracy is not None else None,
    }
