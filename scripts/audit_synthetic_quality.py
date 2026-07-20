#!/usr/bin/env python
"""Synthetic label quality audit.

Answers: are the synthetic training labels pulling classifiers in the right direction,
or are real flywheel labels contradicting them?

For each agent:
  1. Load real labels from the DB.
  2. If n_real < 8, skip (too few for a meaningful split).
  3. Stratified 80/20 split of real labels.
  4. Train two models on the train fold:
       - synthetic-only (no real labels)
       - blended (synthetic + 5x oversampled train-fold real labels)
  5. Evaluate both on the held-out real-label test fold.
  6. Print comparison. Flag agents where blending hurts more than 3pp.

Usage:
    uv run python scripts/audit_synthetic_quality.py
    uv run python scripts/audit_synthetic_quality.py --min-real 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

_AGENTS = {
    "tire":      "f1di.agents.tire_classifier",
    "battery":   "f1di.agents.battery_classifier",
    "weather":   "f1di.agents.weather_classifier",
    "telemetry": "f1di.agents.telemetry_classifier",
}


def _stratified_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.20, seed: int = 0):
    """Stratified split preserving class proportions."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return (
        X[train_idx], y[train_idx],
        X[test_idx],  y[test_idx],
    )


def _eval(clf_cls, scaler_cls, X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score
    from f1di.agents.battery_classifier import _multiclass_brier
    scaler = scaler_cls()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    clf = clf_cls(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_tr_s, y_train)
    preds = clf.predict(X_te_s)
    proba = clf.predict_proba(X_te_s)
    acc   = float(accuracy_score(y_test, preds))
    brier = float(_multiclass_brier(proba, y_test, clf.classes_))
    return acc, brier


def audit_agent(agent: str, mod_name: str, min_real: int) -> dict | None:
    import importlib
    mod = importlib.import_module(mod_name)

    load_fn = getattr(mod, "_load_labeled_from_db")
    synth_fn = getattr(mod, "generate_synthetic")

    X_real, y_real = load_fn()
    n_real = len(y_real)
    if n_real < min_real:
        return {"agent": agent, "skipped": True, "n_real": n_real, "reason": f"< {min_real} real labels"}

    X_tr, y_tr, X_te, y_te = _stratified_split(X_real, y_real)

    # Synthetic-only baseline
    X_synth, y_synth = synth_fn(n=600, seed=42)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    acc_synth, brier_synth = _eval(LogisticRegression, StandardScaler, X_synth, y_synth, X_te, y_te)

    # Blended: synthetic + 5x oversampled train-fold real
    oversample = 5
    X_blend = np.vstack([X_synth, np.repeat(X_tr, oversample, axis=0)])
    y_blend = np.concatenate([y_synth, np.repeat(y_tr, oversample)])
    acc_blend, brier_blend = _eval(LogisticRegression, StandardScaler, X_blend, y_blend, X_te, y_te)

    acc_delta   = acc_blend - acc_synth
    brier_delta = brier_blend - brier_synth  # negative = better

    return {
        "agent":       agent,
        "skipped":     False,
        "n_real":      n_real,
        "n_train":     len(y_tr),
        "n_test":      len(y_te),
        "acc_synth":   round(acc_synth, 4),
        "brier_synth": round(brier_synth, 4),
        "acc_blend":   round(acc_blend, 4),
        "brier_blend": round(brier_blend, 4),
        "acc_delta":   round(acc_delta, 4),
        "brier_delta": round(brier_delta, 4),
        # True when real labels are consistent with synthetic (blending helps or is neutral)
        "aligned":     acc_delta >= -0.03,
    }


def main(min_real: int) -> None:
    from f1di.storage.database import get_engine
    get_engine()

    print(f"\n{_BOLD}Synthetic label quality audit{_RESET}")
    print(f"Min real labels to audit: {min_real}\n")
    print(f"{'Agent':<12} {'n_real':>6} {'n_test':>6}  "
          f"{'acc synth':>9} {'acc blend':>9} {'Δacc':>7}  "
          f"{'brier synth':>11} {'brier blend':>11} {'Δbrier':>7}  {'aligned':>7}")
    print("─" * 110)

    warnings: list[str] = []
    any_run = False

    for agent, mod_name in _AGENTS.items():
        try:
            result = audit_agent(agent, mod_name, min_real)
        except Exception as exc:
            print(f"{agent:<12}  ERROR: {exc}")
            continue

        if result["skipped"]:
            color = _YELLOW
            print(f"{color}{agent:<12}{_RESET}  skipped — {result['reason']}")
            continue

        any_run = True
        delta_acc   = result["acc_delta"]
        delta_brier = result["brier_delta"]
        aligned     = result["aligned"]
        color = _GREEN if aligned else _RED

        delta_acc_str   = f"{delta_acc:+.3f}"
        delta_brier_str = f"{delta_brier:+.3f}"
        flag = "✓" if aligned else "⚠ MISALIGNED"

        print(
            f"{color}{agent:<12}{_RESET} "
            f"{result['n_real']:>6} {result['n_test']:>6}  "
            f"{result['acc_synth']:>9.3f} {result['acc_blend']:>9.3f} {delta_acc_str:>7}  "
            f"{result['brier_synth']:>11.4f} {result['brier_blend']:>11.4f} {delta_brier_str:>7}  "
            f"{color}{flag}{_RESET}"
        )

        if not aligned:
            warnings.append(
                f"  {agent}: blending hurts accuracy by {abs(delta_acc)*100:.1f}pp — "
                "real labels may conflict with synthetic generators. "
                "Consider auditing the synthetic label function."
            )

    print()
    if not any_run:
        print(f"{_YELLOW}No agents had enough real labels (min={min_real}).{_RESET}")
        print("Run a race weekend labelling pass first: POST /v1/outcomes/label")
    elif warnings:
        print(f"{_RED}⚠ Misaligned agents (real labels contradict synthetic):{_RESET}")
        for w in warnings:
            print(w)
        print("\nInterpretation: a large negative Δacc means the synthetic labels are")
        print("teaching the model patterns that don't match real race outcomes.")
        print("Action: review the synthetic generator for that agent, or reduce")
        print("synthetic_n so real labels have more relative weight.")
    else:
        print(f"{_GREEN}✓ All audited agents show blending helps or is neutral.{_RESET}")
        print("Real flywheel labels are consistent with synthetic training data.")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic label quality audit")
    parser.add_argument("--min-real", type=int, default=8,
                        help="Minimum real labels required to run audit for an agent (default: 8)")
    args = parser.parse_args()
    main(args.min_real)
