#!/usr/bin/env python
"""Empirically fit optimal policy confidence thresholds from labeled insights.

Reads all insights with outcome labels or human feedback from the DB and computes,
for each confidence level, the precision of WARNING/CRITICAL predictions.  Outputs
the confidence threshold at which precision crosses the target, and prints
ready-to-paste F1DI_ env var recommendations.

Does NOT modify any config — purely diagnostic.

Usage:
    uv run python scripts/fit_policy_thresholds.py
    uv run python scripts/fit_policy_thresholds.py --target-driver 0.80 --target-engineer 0.60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_DEFAULT_TARGET_DRIVER = 0.75
_DEFAULT_TARGET_ENGINEER = 0.60
_N_BINS = 10


def _precision_at_threshold(
    pairs: list[tuple[float, bool]], threshold: float
) -> tuple[float, int]:
    """Return (precision, n) for all pairs with confidence >= threshold."""
    subset = [(c, ok) for c, ok in pairs if c >= threshold]
    if not subset:
        return 0.0, 0
    n_correct = sum(1 for _, ok in subset if ok)
    return n_correct / len(subset), len(subset)


def main(target_driver: float, target_engineer: float) -> None:
    from f1di.storage.database import db_session, init_db
    from f1di.storage.models import FeedbackRecord, InsightRecord
    from sqlalchemy import select

    init_db()

    pairs_by_audience: dict[str, list[tuple[float, bool]]] = {}

    with db_session() as session:
        stmt = (
            select(FeedbackRecord, InsightRecord)
            .outerjoin(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
            .where(InsightRecord.risk.in_(["WARNING", "CRITICAL"]))
        )
        for fb, ins in session.execute(stmt).all():
            if ins is None:
                continue
            if fb.correct is not None:
                is_correct = fb.correct
            elif fb.rating is not None:
                is_correct = fb.rating >= 4
            else:
                continue
            audience = ins.audience or "DRIVER"
            pairs_by_audience.setdefault(audience, []).append((ins.confidence, is_correct))

    all_pairs = [p for ps in pairs_by_audience.values() for p in ps]
    if not all_pairs:
        print("No labeled insights found in the DB.")
        print("Run the flywheel (F1DI_INGESTION_AUTO_ENABLED=true) to generate outcome labels.")
        sys.exit(1)

    print(f"Labeled WARNING/CRITICAL insights: {len(all_pairs)}")
    print(f"  by audience: { {k: len(v) for k, v in pairs_by_audience.items()} }")
    print()

    # ── Precision-vs-confidence table ─────────────────────────────────────
    thresholds = [i / _N_BINS for i in range(_N_BINS + 1)]
    print(f"{'Threshold':>12} {'Precision':>10} {'N':>6}")
    print("-" * 32)
    for thr in thresholds:
        prec, n = _precision_at_threshold(all_pairs, thr)
        bar = "█" * int(prec * 20)
        print(f"  >= {thr:.2f}   {prec:>8.1%}  {n:>5}  {bar}")
    print()

    # ── Find optimal thresholds ────────────────────────────────────────────
    def _find_threshold(pairs: list[tuple[float, bool]], target: float) -> tuple[float, float, int]:
        """Return (threshold, precision, n) where precision first crosses target."""
        best_thr, best_prec, best_n = 0.0, 0.0, len(pairs)
        for thr in thresholds:
            prec, n = _precision_at_threshold(pairs, thr)
            if prec >= target and n >= 5:
                best_thr, best_prec, best_n = thr, prec, n
                break
        return best_thr, best_prec, best_n

    driver_pairs = pairs_by_audience.get("DRIVER", all_pairs)
    engineer_pairs = pairs_by_audience.get("ENGINEER", all_pairs)

    d_thr, d_prec, d_n = _find_threshold(driver_pairs, target_driver)
    e_thr, e_prec, e_n = _find_threshold(engineer_pairs, target_engineer)

    from f1di.config.settings import settings
    print(f"Current thresholds:  driver={settings.confidence_min_driver}  engineer={settings.confidence_min_engineer}")
    print()
    print(f"Driver   (target {target_driver:.0%}):  threshold={d_thr:.2f}  achieved={d_prec:.1%}  n={d_n}")
    print(f"Engineer (target {target_engineer:.0%}):  threshold={e_thr:.2f}  achieved={e_prec:.1%}  n={e_n}")
    print()

    if abs(d_thr - settings.confidence_min_driver) < 0.05 and abs(e_thr - settings.confidence_min_engineer) < 0.05:
        print("Current thresholds are within 5 pp of empirically fitted values — no change recommended.")
    else:
        print("Recommended .env updates:")
        if abs(d_thr - settings.confidence_min_driver) >= 0.05:
            print(f"  F1DI_CONFIDENCE_MIN_DRIVER={d_thr:.2f}   # was {settings.confidence_min_driver}")
        if abs(e_thr - settings.confidence_min_engineer) >= 0.05:
            print(f"  F1DI_CONFIDENCE_MIN_ENGINEER={e_thr:.2f}  # was {settings.confidence_min_engineer}")
        print()
        print("Note: accumulate ≥50 labeled insights per audience before trusting these values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empirically fit policy confidence thresholds")
    parser.add_argument("--target-driver", type=float, default=_DEFAULT_TARGET_DRIVER,
                        help="Target precision for driver-facing insights (default 0.75)")
    parser.add_argument("--target-engineer", type=float, default=_DEFAULT_TARGET_ENGINEER,
                        help="Target precision for engineer insights (default 0.60)")
    args = parser.parse_args()
    main(args.target_driver, args.target_engineer)
