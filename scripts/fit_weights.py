#!/usr/bin/env python
"""Fit logistic regression weights for compute_raw_score() and compare to current heuristics.

Usage:
    uv run python scripts/fit_weights.py
    uv run python scripts/fit_weights.py --n-races 60 --seed 7

Current hardcoded weights in calibration.py:
  max_risk=0.30, model_confidence=0.25, mean_risk=0.20,
  evidence_strength=0.15, agent_agreement=0.10

The script generates synthetic scenarios, extracts the five intermediate calibration
features from compute_raw_score(), fits a logistic regression against ground-truth
labels, and prints a comparison table.  No production state is modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_FEATURE_ORDER = ["risk_max", "model_confidence", "risk_mean", "evidence_strength", "agent_agreement"]
_CURRENT_WEIGHTS = {
    "risk_max": 0.30,
    "model_confidence": 0.25,
    "risk_mean": 0.20,
    "evidence_strength": 0.15,
    "agent_agreement": 0.10,
}


def _normalize_coefs(coefs: list[float]) -> list[float]:
    total = sum(abs(c) for c in coefs)
    return [c / total if total > 1e-9 else 0.0 for c in coefs]


def main(n_races: int, seed: int) -> None:
    print(f"Generating synthetic calibration dataset (n_races={n_races}, seed={seed}) ...")
    from f1di.confidence.fitting import generate_feature_dataset

    features_list, y = generate_feature_dataset(n_races=n_races, seed=seed)
    n = len(features_list)
    if n < 10:
        print(f"Too few samples ({n}) — increase --n-races")
        sys.exit(1)
    print(f"  {n} windows generated")

    X = [[f.get(feat, 0.0) for feat in _FEATURE_ORDER] for f in features_list]

    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        lr.fit(X, [round(yi) for yi in y])
        raw_coefs = [float(c) for c in lr.coef_[0]]
    except ImportError:
        print("scikit-learn not installed — falling back to correlation-based ranking")
        raw_coefs = []
        for i, feat in enumerate(_FEATURE_ORDER):
            col = [row[i] for row in X]
            n_col = len(col)
            mean_x = sum(col) / n_col
            mean_y = sum(y) / n_col
            cov = sum((col[j] - mean_x) * (y[j] - mean_y) for j in range(n_col))
            var_x = sum((v - mean_x) ** 2 for v in col)
            raw_coefs.append(cov / (var_x ** 0.5 + 1e-9))

    fitted = _normalize_coefs(raw_coefs)

    print()
    print(f"{'Feature':<22} {'Current':>10} {'Fitted (LR)':>12}  {'Delta':>8}")
    print("-" * 56)
    for feat, fit_w in zip(_FEATURE_ORDER, fitted):
        cur_w = _CURRENT_WEIGHTS[feat]
        delta = fit_w - cur_w
        flag = " <-- review" if abs(delta) > 0.08 else ""
        print(f"  {feat:<20} {cur_w:>10.4f} {fit_w:>12.4f}  {delta:>+8.4f}{flag}")

    print()
    max_delta = max(abs(fw - _CURRENT_WEIGHTS[f]) for f, fw in zip(_FEATURE_ORDER, fitted))
    if max_delta > 0.08:
        print(
            "RECOMMENDATION: At least one weight differs by >8 pp from the fitted value.\n"
            "Consider updating the weights in src/f1di/confidence/calibration.py:compute_raw_score()."
        )
    else:
        print("RECOMMENDATION: Current weights are within 8 pp of fitted values — no change needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate compute_raw_score() weights via logistic regression")
    parser.add_argument("--n-races", type=int, default=30, help="Synthetic scenario count (default 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args.n_races, args.seed)
