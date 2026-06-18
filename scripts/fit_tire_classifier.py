#!/usr/bin/env python
"""Train the tire strategy logistic regression classifier.

Combines synthetic rule-distilled data (cold start) with any flywheel-labeled
tire_strategy insights from the database.  After the first race weekend of
labeled data, real outcomes begin overriding the synthetic prior.

Usage:
    uv run python scripts/fit_tire_classifier.py
    uv run python scripts/fit_tire_classifier.py --synthetic-n 1200 --oversample 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


def main(synthetic_n: int, oversample: int, output: str | None) -> None:
    from f1di.agents.tire_classifier import train_from_labels, _CLASSIFIER_PATH

    out_path = Path(output) if output else _CLASSIFIER_PATH
    print(f"\nFitting TireClassifier → {out_path}")
    print(f"  synthetic_n={synthetic_n}  real_oversample={oversample}")
    print()

    report = train_from_labels(
        output_path=out_path,
        synthetic_n=synthetic_n,
        real_oversample=oversample,
    )

    n_real = report["n_real"]
    n_synth = report["n_synthetic"]
    acc = report["accuracy"]
    dist = report.get("class_distribution", {})

    print("Results")
    print("  Synthetic examples :", n_synth)
    print(
        "  Real examples      :",
        f"{n_real}  {_YELLOW}(below blend threshold of 10 — synthetic only){_RESET}"
        if n_real < 10 else
        f"{n_real}  (blended with {oversample}× oversample)",
    )
    print(f"  Total training size: {report['n_total']}")
    print(f"  Training accuracy  : {acc:.4f}")
    print()
    print("Class distribution (training labels):")
    for cls in ("INFO", "WATCH", "WARNING", "CRITICAL"):
        n = dist.get(cls, 0)
        bar = "█" * (n // 20)
        print(f"  {cls:<10} {n:>5}  {bar}")

    per = report.get("per_class", {})
    if per:
        print("Per-class (CV held-out):")
        print(f"  {'Class':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
        print(f"  {'─'*42}")
        for cls in ("INFO", "WATCH", "WARNING", "CRITICAL"):
            m = per.get(cls)
            if m:
                print(f"  {cls:<10} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6}")
        print()

    color = _GREEN if acc >= 0.85 else _YELLOW
    print(f"{color}Classifier saved to {out_path}{_RESET}")
    if n_real == 0:
        print(
            f"{_YELLOW}No flywheel labels yet — model reproduces the hand-written rules.\n"
            f"Run again after the first race weekend to blend real outcomes.{_RESET}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tire strategy classifier")
    parser.add_argument("--synthetic-n", type=int, default=800)
    parser.add_argument("--oversample", type=int, default=5,
                        help="How many times to repeat real examples in training mix")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args.synthetic_n, args.oversample, args.output)
