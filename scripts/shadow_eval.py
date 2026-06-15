#!/usr/bin/env python
"""Print shadow-vs-production comparison and promotion recommendation.

Usage:
    uv run python scripts/shadow_eval.py
    uv run python scripts/shadow_eval.py --version weights-v2 --min-n 30

Exit code 0 if the challenger is recommended for promotion, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_CHALLENGER_VERSION = "weights-v2"


def main(challenger_version: str, min_n: int) -> None:
    from f1di.storage.database import db_session, init_db
    from f1di.storage.repository import shadow_compare, shadow_evaluate

    init_db()
    with db_session() as session:
        comparison = shadow_compare(session, challenger_version)
        evaluation = shadow_evaluate(session, challenger_version, min_n=min_n)

    prod = comparison.get("production", {})
    shadow = comparison.get("shadow", {})

    print(f"Challenger: {challenger_version}")
    print()
    print(f"{'':30} {'Production':>14} {'Shadow (v2)':>14}")
    print("-" * 60)
    print(f"  {'N insights':28} {prod.get('n', 0):>14} {shadow.get('n', 0):>14}")
    print(f"  {'Avg confidence':28} {prod.get('avg_confidence', 0):>14.4f} {shadow.get('avg_confidence', 0):>14.4f}")
    print(f"  {'Avg uncertainty':28} {prod.get('avg_uncertainty', 0):>14.4f} {shadow.get('avg_uncertainty', 0):>14.4f}")
    print()

    prod_dist = prod.get("risk_distribution", {})
    shadow_dist = shadow.get("risk_distribution", {})
    risks = sorted(set(prod_dist) | set(shadow_dist))
    if risks:
        print("Risk distribution:")
        for risk in risks:
            print(f"  {risk:<12} prod={prod_dist.get(risk, 0):>5}  shadow={shadow_dist.get(risk, 0):>5}")
        print()

    rec = evaluation.get("recommendation", "insufficient_data")
    promote = evaluation.get("promote", False)

    if evaluation.get("recommendation") == "insufficient_data":
        n_shadow = evaluation.get("n_shadow", 0)
        print(
            f"INSUFFICIENT DATA — {n_shadow} shadow insights collected, need ≥ {min_n}.\n"
            "Keep F1DI_SHADOW_CHALLENGER_ENABLED=true and let it accumulate more traffic."
        )
        sys.exit(1)

    print(f"  U-stat:         {evaluation.get('u_statistic', 'n/a')}")
    print(f"  p-value:        {evaluation.get('p_value', 'n/a')}")
    print(f"  Rank-biserial:  {evaluation.get('rank_biserial_correlation', 'n/a')}")
    print(f"  Shadow escal.:  {evaluation.get('shadow_escalation_rate', 'n/a')}")
    print(f"  Prod escal.:    {evaluation.get('prod_escalation_rate', 'n/a')}")
    print()
    print(f"Recommendation: {rec.upper()}")
    print()

    if promote:
        print(
            "PROMOTE: Update the weights in src/f1di/confidence/calibration.py by replacing\n"
            "compute_raw_score() with compute_raw_score_v2(), then set\n"
            "F1DI_SHADOW_CHALLENGER_ENABLED=false."
        )
        sys.exit(0)
    else:
        print("HOLD: Do not promote yet — see recommendation above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shadow challenger promotion evaluation")
    parser.add_argument("--version", default=_CHALLENGER_VERSION, help="challenger_version tag")
    parser.add_argument("--min-n", type=int, default=30, help="Minimum shadow insights required")
    args = parser.parse_args()
    main(args.version, args.min_n)
