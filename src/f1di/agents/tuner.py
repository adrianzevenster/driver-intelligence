"""Optuna-based hyperparameter tuner for HGBC classifier agents.

Searches max_iter, max_depth, learning_rate, min_samples_leaf, and
l2_regularization using TPE. Saves the best params to
data/calibration/{agent}_best_params.json; subsequent retrains pick them
up automatically via build_model(agent=...).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("f1di.agents.tuner")

_AGENT_MODULES = {
    "tire":       "f1di.agents.tire_classifier",
    "battery":    "f1di.agents.battery_classifier",
    "weather":    "f1di.agents.weather_classifier",
    "telemetry":  "f1di.agents.telemetry_classifier",
    "safety_car": "f1di.agents.safety_car_classifier",
    "fuel":       "f1di.agents.fuel_classifier",
    "meta":       "f1di.inference.meta_learner",
}

_SYNTH_N = {
    "tire": 1200, "battery": 900, "weather": 900,
    "telemetry": 1200, "safety_car": 1200, "fuel": 900, "meta": 1200,
}

_SEARCH_SPACE = {
    "max_iter":          ("int",   100, 500),
    "max_depth":         ("int",   2,   8),
    "learning_rate":     ("float", 0.01, 0.3,  True),   # log scale
    "min_samples_leaf":  ("int",   5,   80),
    "l2_regularization": ("float", 1e-4, 1.0,  True),   # log scale
}


def _suggest(trial, name: str):
    kind, *args = _SEARCH_SPACE[name]
    if kind == "int":
        return trial.suggest_int(name, *args)
    log = args[2] if len(args) > 2 else False
    return trial.suggest_float(name, args[0], args[1], log=log)


def tune_agent(agent: str, n_trials: int = 30, seed: int = 42) -> dict:
    """Run Optuna search for one agent. Returns result dict and saves best params.

    Uses the agent's own generate_synthetic function as the evaluation dataset
    so the search space is grounded in realistic data distributions. CV is
    5-fold stratified (same as the production retrain) so the metric is directly
    comparable to what you see in the Model Lab.
    """
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("pip install optuna to use the tuner") from exc

    if agent not in _AGENT_MODULES:
        raise ValueError(f"Unknown agent {agent!r}. Choose from {sorted(_AGENT_MODULES)}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    import importlib
    mod = importlib.import_module(_AGENT_MODULES[agent])
    generate_synthetic = mod.generate_synthetic

    from f1di.agents.classifier_utils import (
        cross_val_eval,
        multiclass_brier,
        save_best_params,
        _HGBC_DEFAULTS,
    )

    X, y = generate_synthetic(n=_SYNTH_N[agent], seed=seed)

    # Baseline: default HGBC params (no best-params file influence — raw defaults)
    def _baseline_build():
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import HistGradientBoostingClassifier
        return StandardScaler(), HistGradientBoostingClassifier(**_HGBC_DEFAULTS)

    baseline_cv = cross_val_eval(_baseline_build, X, y, multiclass_brier, n_splits=5)
    baseline = baseline_cv["cv_accuracy"] if baseline_cv else 0.0
    logger.info("tuner[%s] baseline cv_acc=%.4f", agent, baseline)

    trial_log: list[dict] = []

    def objective(trial):
        params = {name: _suggest(trial, name) for name in _SEARCH_SPACE}

        def _build():
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import HistGradientBoostingClassifier
            return StandardScaler(), HistGradientBoostingClassifier(random_state=42, **params)

        cv = cross_val_eval(_build, X, y, multiclass_brier, n_splits=5)
        score = cv["cv_accuracy"] if cv else 0.0
        trial_log.append({"trial": trial.number, "score": round(score, 4), "params": params})
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score  = study.best_value

    save_best_params(agent, best_params, best_score, baseline, n_trials)
    logger.info(
        "tuner[%s] done — best cv_acc=%.4f baseline=%.4f improvement=+%.2fpp n_trials=%d",
        agent, best_score, baseline, (best_score - baseline) * 100, n_trials,
    )

    return {
        "agent":                agent,
        "best_params":          best_params,
        "best_cv_accuracy":     round(best_score, 4),
        "baseline_cv_accuracy": round(baseline, 4),
        "improvement_pp":       round((best_score - baseline) * 100, 2),
        "n_trials":             n_trials,
        "n_complete":           len(study.trials),
        "trial_scores":         [t["score"] for t in trial_log],
    }
