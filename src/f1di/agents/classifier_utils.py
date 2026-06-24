"""Shared utilities for agent classifiers.

Provides save_with_snapshot() — mirrors the isotonic calibrator's versioning
pattern: always writes a timestamped copy; only promotes to the live path when
accuracy hasn't regressed, so a bad retrain can't silently clobber a good model.

Provides record_history() — appends a structured entry to model_history.json
so every classifier fit is traceable alongside calibrator retrains.

Provides multiclass_brier() and cross_val_eval() — every classifier used to
report accuracy/Brier by scoring the model on the exact rows it was just
fitted on (including oversampled real-label duplicates), which is always
optimistic and can't actually detect a regression. cross_val_eval() reports
held-out, k-fold metrics instead so the regression guard in save_with_snapshot
is judging generalization, not memorization.

Provides blend_with_transfer() — every classifier used to "blend" real
flywheel labels into the synthetic prior by literally duplicating each real
row a fixed number of times (np.repeat(X_real, 5, axis=0)) once n_real
crossed a hard threshold. That makes the real data's influence jump
discontinuously at the threshold and never grow afterward — 11 real examples
and 500 real examples got the same relative weight. blend_with_transfer()
replaces the repeat with a continuous sample_weight that ramps from "same
weight as one synthetic row" at the entry floor up to a cap as n_real grows,
and reports the synthetic-only "prior" CV score alongside so the caller can
measure the actual lift from transfer learning rather than assume it helped.
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("f1di.agents.classifier_utils")

_HISTORY_PATH = Path("data/calibration/model_history.json")
_BEST_PARAMS_DIR = Path("data/calibration")

MODEL_TYPES = ["logistic", "hgbc"]
_MODEL_DISPLAY = {"logistic": "LogisticRegression", "hgbc": "HistGradientBoosting"}
_MODEL_VERSION = {"logistic": "lr-v1", "hgbc": "hgb-v1"}

_HGBC_DEFAULTS: dict = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "min_samples_leaf": 15,
    "l2_regularization": 0.1,
    "random_state": 42,
}


def load_best_params(agent: str) -> dict:
    """Return saved Optuna best-params for *agent*, or {} if none found."""
    path = _BEST_PARAMS_DIR / f"{agent}_best_params.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("params", {})
    except Exception:
        return {}


def save_best_params(
    agent: str,
    params: dict,
    best_score: float,
    baseline_score: float,
    n_trials: int,
) -> None:
    """Persist Optuna results to data/calibration/{agent}_best_params.json."""
    try:
        _BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
        (_BEST_PARAMS_DIR / f"{agent}_best_params.json").write_text(
            json.dumps(
                {
                    "agent": agent,
                    "params": params,
                    "cv_accuracy": round(best_score, 4),
                    "baseline_cv_accuracy": round(baseline_score, 4),
                    "improvement_pp": round((best_score - baseline_score) * 100, 2),
                    "n_trials": n_trials,
                    "tuned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
            )
        )
    except OSError as exc:
        logger.warning(
            "save_best_params skipped — cannot write calibration dir (permission issue?): %s", exc
        )


def build_model(model_type: str = "logistic", max_depth: int = 4, agent: str | None = None):
    """Return (StandardScaler, fitted-ready sklearn model) for `model_type`.

    StandardScaler is always returned even for HGBC — callers that use
    ood_score() rely on scaler.mean_/scale_ to detect out-of-distribution
    inputs, independent of whether the model itself needs scaling.

    If *agent* is given, any saved Optuna best-params for that agent are
    merged on top of the HGBC defaults so a post-tune retrain automatically
    picks up the improved hyperparameters without any manual wiring.
    """
    from sklearn.preprocessing import StandardScaler
    mt = model_type.lower()
    if mt in ("logistic", "lr"):
        from sklearn.linear_model import LogisticRegression
        return StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    elif mt in ("hgbc", "hgb", "histgradientboosting"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = {**_HGBC_DEFAULTS, "max_depth": max_depth}
        if agent:
            params.update(load_best_params(agent))
        return StandardScaler(), HistGradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose from {MODEL_TYPES}")


def class_balance_weights(y: np.ndarray, sample_weight: np.ndarray | None) -> np.ndarray:
    """Multiply existing sample_weight by class inverse frequency.

    Corrects for label imbalance (e.g. 75% INFO) so minority classes like
    WARNING/CRITICAL get equal gradient influence. Applied on top of the
    blend_with_transfer weights so real-vs-synthetic and class-balance
    corrections are both active simultaneously.
    """
    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)
    inv_freq = np.ones(len(y), dtype=np.float64)
    for cls, cnt in zip(unique, counts):
        inv_freq[y == cls] = len(y) / (n_classes * cnt)
    if sample_weight is None:
        return inv_freq
    return sample_weight * inv_freq


def multiclass_brier(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Mean squared error between predicted probability vectors and one-hot targets."""
    n, n_c = len(y), len(classes)
    cls_idx = {int(c): i for i, c in enumerate(classes)}
    Y_oh = np.zeros((n, n_c), dtype=np.float64)
    for i, yi in enumerate(y):
        Y_oh[i, cls_idx[int(yi)]] = 1.0
    return float(np.mean(np.sum((proba - Y_oh) ** 2, axis=1)))


def cross_val_eval(
    build_pipeline,
    X: np.ndarray,
    y: np.ndarray,
    brier_fn=multiclass_brier,
    n_splits: int = 5,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
    scoring_fn=None,
    collect_predictions: bool = False,
) -> dict | None:
    """Honest accuracy/Brier via stratified k-fold CV.

    Args:
        build_pipeline: zero-arg callable returning a fresh (scaler, model) pair
            with the same hyperparameters as the production model — called once
            per fold so each fold trains on an unfitted copy.
        X, y: the full training set (synthetic + weighted real rows) — the
            same data passed to .fit().
        brier_fn: (proba, y_true, classes) -> float.
        sample_weight: optional per-row weights, sliced per fold and passed to
            the fold model's .fit() — used by blend_with_transfer() below.

    Returns:
        {"cv_accuracy", "cv_brier", "cv_accuracy_std", "cv_brier_std",
        "fold_accuracies", "fold_briers", "n_splits"}, or None if there isn't
        enough data per class to run CV honestly (caller should fall back to a
        plain train-set score in that case, since it's the best available
        signal with very little data).

        fold_accuracies/fold_briers (one value per fold) are kept around so
        callers — and the model_history.json / Model Lab UI that read them —
        can show the spread, not just the mean. A 0.91 mean over folds of
        [0.95, 0.94, 0.93, 0.85, 0.88] is a much less reliable number than the
        same mean over [0.91, 0.91, 0.92, 0.90, 0.91], and the regression
        guard below uses that spread to size its threshold.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold

    _score_fn = scoring_fn if scoring_fn is not None else accuracy_score

    classes_present, counts = np.unique(y, return_counts=True)
    if len(classes_present) < 2 or counts.min() < 2:
        return None
    n_splits_eff = max(2, min(n_splits, int(counts.min())))

    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    accs, briers = [], []
    all_y_true: list = []
    all_y_pred: list = []
    for train_idx, test_idx in skf.split(X, y):
        scaler, model = build_pipeline()
        X_tr = scaler.fit_transform(X[train_idx])
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None
        model.fit(X_tr, y[train_idx], sample_weight=sw_tr)
        X_te = scaler.transform(X[test_idx])
        proba = model.predict_proba(X_te)
        y_pred = model.predict(X_te)
        accs.append(float(_score_fn(y[test_idx], y_pred)))
        briers.append(float(brier_fn(proba, y[test_idx], model.classes_)))
        if collect_predictions:
            all_y_true.extend(y[test_idx].tolist())
            all_y_pred.extend(y_pred.tolist())

    result = {
        "cv_accuracy": float(np.mean(accs)),
        "cv_brier": float(np.mean(briers)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_brier_std": float(np.std(briers)),
        "fold_accuracies": accs,
        "fold_briers": briers,
        "n_splits": n_splits_eff,
    }
    if collect_predictions:
        result["cv_y_true"] = all_y_true
        result["cv_y_pred"] = all_y_pred
    return result


def per_class_report(cv: dict | None, label_map: dict[int, str]) -> dict:
    """Compute per-class precision/recall/f1 from CV fold predictions.

    Returns a dict keyed by class name, each value {"precision", "recall", "f1", "support"}.
    Empty dict if cv is None or predictions weren't collected.
    """
    if cv is None or "cv_y_true" not in cv:
        return {}
    from sklearn.metrics import classification_report
    y_true = np.array(cv["cv_y_true"])
    y_pred = np.array(cv["cv_y_pred"])
    labels = sorted(label_map.keys())
    names = [label_map[k] for k in labels]
    try:
        rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                    output_dict=True, zero_division=0)
        return {cls: {"precision": round(rep[cls]["precision"], 3),
                      "recall": round(rep[cls]["recall"], 3),
                      "f1": round(rep[cls]["f1-score"], 3),
                      "support": rep[cls]["support"]}
                for cls in names if cls in rep}
    except Exception:
        return {}


# Below REAL_WEIGHT_FLOOR real examples, real data isn't used at all — too
# noisy to trust over the synthetic prior. Above it, each real row's weight
# ramps linearly from 1.0 (same as one synthetic row) up to REAL_WEIGHT_CAP
# by the time n_real reaches REAL_WEIGHT_SATURATION, instead of jumping
# straight to a fixed multiplier regardless of whether n_real is 10 or 500.
REAL_WEIGHT_FLOOR = 10
REAL_WEIGHT_SATURATION = 500


def real_sample_weight(
    n_real: int,
    cap: float,
    floor: int = REAL_WEIGHT_FLOOR,
    saturation: int = REAL_WEIGHT_SATURATION,
) -> float:
    """Per-row weight for real examples, continuous in n_real (see module docstring)."""
    if n_real < floor:
        return 0.0
    growth = min(1.0, (n_real - floor) / max(1, saturation - floor))
    return 1.0 + (cap - 1.0) * growth


def blend_with_transfer(
    build_pipeline,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    X_real: np.ndarray,
    y_real: np.ndarray,
    n_real: int,
    brier_fn=multiclass_brier,
    weight_cap: float = 5.0,
    n_splits: int = 5,
) -> dict:
    """Blend a synthetic prior with real flywheel labels via continuous sample
    weighting (see real_sample_weight) and report the synthetic-only "prior"
    CV score for comparison.

    The caller still does its own cross_val_eval on the returned (X, y,
    sample_weight) to get the blended model's CV score — the gap between that
    and prior_cv here is the actual, measurable lift from transfer learning
    onto real race outcomes, rather than an assumption that blending helped.

    Returns:
        {"X", "y", "sample_weight", "real_weight", "prior_cv"} where
        sample_weight is None when n_real is below the floor (no blending —
        X/y are just the synthetic prior) and prior_cv is the output of
        cross_val_eval on the synthetic-only data (or None if that itself
        couldn't be cross-validated).
    """
    prior_cv = cross_val_eval(build_pipeline, X_synth, y_synth, brier_fn, n_splits=n_splits)

    weight = real_sample_weight(n_real, cap=weight_cap)
    if weight <= 0.0:
        return {"X": X_synth, "y": y_synth, "sample_weight": None, "real_weight": 0.0, "prior_cv": prior_cv}

    X = np.vstack([X_synth, X_real])
    y = np.concatenate([y_synth, y_real])
    sample_weight = np.concatenate([np.ones(len(y_synth)), np.full(len(y_real), weight)])
    return {"X": X, "y": y, "sample_weight": sample_weight, "real_weight": float(weight), "prior_cv": prior_cv}


def record_history(
    clf, agent: str, versioned_path: str, blocked: bool,
    history_path: Path = _HISTORY_PATH, threshold: float | None = None,
) -> None:
    """Append one classifier fit entry to model_history.json."""
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("record_history skipped — cannot create calibration dir: %s", exc)
        return
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
        "cv_n_splits": getattr(clf, "cv_n_splits", 0),
        "cv_accuracy_std": round(s, 4) if (s := getattr(clf, "cv_accuracy_std", None)) is not None else None,
        "cv_brier_std": round(s, 4) if (s := getattr(clf, "cv_brier_std", None)) is not None else None,
        "cv_fold_accuracies": [round(v, 4) for v in fa] if (fa := getattr(clf, "cv_fold_accuracies", None)) else None,
        "cv_fold_briers": [round(v, 4) for v in fb] if (fb := getattr(clf, "cv_fold_briers", None)) else None,
        "real_sample_weight": round(w, 4) if (w := getattr(clf, "real_sample_weight", None)) is not None else None,
        "prior_cv_accuracy": round(p, 4) if (p := getattr(clf, "prior_cv_accuracy", None)) is not None else None,
        "transfer_lift": round(clf.accuracy - p, 4) if (p := getattr(clf, "prior_cv_accuracy", None)) is not None else None,
        "versioned_path": versioned_path,
        "blocked": blocked,
        "block_threshold": round(threshold, 4) if threshold is not None else None,
    })
    try:
        history_path.write_text(json.dumps(entries, indent=2))
        logger.info("model_history updated: agent=%s version=%s acc=%.4f", agent, getattr(clf, "model_version", "?"), clf.accuracy)
    except OSError as exc:
        logger.warning("record_history write failed — permission issue on volume: %s", exc)


def save_with_snapshot(
    clf,
    live_path: Path,
    min_accuracy_delta: float = 0.02,
    z_score: float = 1.64,
) -> dict:
    """Save *clf* with a versioned snapshot and an accuracy regression guard.

    accuracy is now a k-fold CV mean (see cross_val_eval), which is noisier than
    the train-set score this guard used to compare. A flat min_accuracy_delta
    can't tell a real regression from fold noise: with ~5 folds, a swing of a
    couple points between runs is normal even when nothing changed. When both
    the new and previous model have a recorded cv_accuracy_std, the effective
    threshold is widened to the larger of min_accuracy_delta and z_score
    standard errors of the difference of the two means — i.e. block only when
    the drop is bigger than what fold noise alone would plausibly produce.
    Falls back to the flat min_accuracy_delta when either model lacks std
    (e.g. too little data to fold, or a pkl saved before this guard existed).

    Args:
        clf: Any classifier with an `.accuracy` and `.n_real` attribute.
        live_path: Path of the canonical live pkl (e.g. `data/calibration/tire_classifier.pkl`).
        min_accuracy_delta: Floor for the block threshold — always block at
            least this much regression, even if fold noise would excuse it.
        z_score: Multiplier on the standard error of the difference used to
            widen the threshold when fold std is available. 1.64 ≈ one-sided
            90% confidence that an observed drop is real, not noise.

    Returns:
        Dict with keys: blocked, versioned_path, accuracy, prev_accuracy, threshold.
    """
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stem = live_path.stem
    versioned_path = live_path.parent / f"{stem}_{ts}.pkl"

    try:
        live_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "save_with_snapshot skipped — cannot create calibration dir (permission issue?): %s", exc
        )
        return {
            "blocked": False,
            "versioned_path": str(versioned_path),
            "accuracy": round(clf.accuracy, 4),
            "prev_accuracy": None,
            "threshold": round(min_accuracy_delta, 4),
        }

    prev_accuracy: float | None = None
    prev_std: float | None = None
    prev_n_splits: int = 0
    if live_path.exists():
        try:
            prev = pickle.loads(live_path.read_bytes())
            prev_accuracy = float(prev.accuracy)
            prev_std = getattr(prev, "cv_accuracy_std", None)
            prev_n_splits = getattr(prev, "cv_n_splits", 0) or 0
        except Exception:
            pass

    # Always write the versioned copy for audit.
    try:
        versioned_path.write_bytes(pickle.dumps(clf))
    except OSError as exc:
        logger.warning("save_with_snapshot write failed — permission issue on volume: %s", exc)
        return {
            "blocked": False,
            "versioned_path": str(versioned_path),
            "accuracy": round(clf.accuracy, 4),
            "prev_accuracy": round(prev_accuracy, 4) if prev_accuracy is not None else None,
            "threshold": round(min_accuracy_delta, 4),
        }

    new_std = getattr(clf, "cv_accuracy_std", None)
    new_n_splits = getattr(clf, "cv_n_splits", 0) or 0

    threshold = min_accuracy_delta
    if prev_std is not None and new_std is not None and prev_n_splits > 1 and new_n_splits > 1:
        se_diff = float(np.sqrt(new_std ** 2 / new_n_splits + prev_std ** 2 / prev_n_splits))
        threshold = max(min_accuracy_delta, z_score * se_diff)

    blocked = (
        prev_accuracy is not None
        and clf.accuracy < prev_accuracy - threshold
    )

    if not blocked:
        try:
            shutil.copy2(versioned_path, live_path)
            logger.info(
                "%s saved: acc=%.4f n_real=%d versioned=%s",
                stem, clf.accuracy, clf.n_real, versioned_path.name,
            )
        except OSError as exc:
            logger.warning(
                "%s copy to live path failed — permission issue on volume: %s", stem, exc
            )
    else:
        logger.warning(
            "%s retrain BLOCKED — new acc %.4f regressed from %.4f (delta=%.4f > threshold=%.4f); "
            "versioned copy saved, live model unchanged.",
            stem, clf.accuracy, prev_accuracy,
            prev_accuracy - clf.accuracy, threshold,
        )

    return {
        "blocked": blocked,
        "versioned_path": str(versioned_path),
        "accuracy": round(clf.accuracy, 4),
        "prev_accuracy": round(prev_accuracy, 4) if prev_accuracy is not None else None,
        "threshold": round(threshold, 4),
    }
