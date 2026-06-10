#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _feedback_pairs() -> list[tuple[float, float]]:
    """Return (calibrated_confidence, label) pairs from human feedback records."""
    from sqlalchemy import select
    from f1di.storage.database import db_session, init_db
    from f1di.storage.models import FeedbackRecord, InsightRecord

    init_db()
    pairs: list[tuple[float, float]] = []
    with db_session() as session:
        stmt = (
            select(FeedbackRecord, InsightRecord)
            .join(InsightRecord, FeedbackRecord.insight_id == InsightRecord.insight_id)
        )
        for fb, ins in session.execute(stmt).all():
            if fb.correct is not None:
                label = 1.0 if fb.correct else 0.0
            elif fb.rating is not None:
                label = (fb.rating - 1) / 4.0
            else:
                continue
            pairs.append((ins.confidence, label))
    return pairs


def main(min_feedback: int, output: Path, quality_output: Path) -> None:
    from f1di.confidence.calibration import ConfidenceCalibrator
    from f1di.confidence.fitting import calibration_ece, calibration_brier, fit_and_save

    feedback_pairs = _feedback_pairs()
    print(f"Feedback records: {len(feedback_pairs)}")

    if len(feedback_pairs) < min_feedback:
        print(f"Need ≥ {min_feedback} feedback records to retrain. Exiting without changes.")
        return

    # Fit base calibrator on synthetic data
    fit_and_save(output_path=None, n_races=30, seed=42)

    # Collect synthetic (X, y) to augment with feedback
    from f1di.confidence.fitting import _build_scenarios
    from f1di.features.extractor import extract_features
    from f1di.confidence.calibration import compute_raw_score
    from f1di.confidence.fitting import _AGENTS, _ground_truth_label

    scenarios = _build_scenarios(per_type=30)
    X_syn, y_syn = [], []
    for sc in scenarios:
        try:
            from f1di.simulator.generator import SyntheticRaceSimulator
            from f1di.rag.store import HybridMemoryRetriever, load_markdown_knowledge
            knowledge_path = Path("data/knowledge")
            retriever = HybridMemoryRetriever()
            if knowledge_path.exists():
                load_markdown_knowledge(retriever, knowledge_path)
            sim = SyntheticRaceSimulator(sc["profile"], retriever)
            window = sim.run(n_laps=sc["laps"], incident_plan=sc.get("incidents"))
            features = extract_features(window)
            findings = [a.analyze(window, []) for a in _AGENTS]
            raw, _ = compute_raw_score(findings)
            label = _ground_truth_label(window, features)
            X_syn.append(raw)
            y_syn.append(label)
        except Exception:
            pass

    # Combine: synthetic base + real feedback (weighted 3× for real signal)
    X_fb = [p[0] for p in feedback_pairs]
    y_fb = [p[1] for p in feedback_pairs]
    X = X_syn + X_fb * 3
    y = y_syn + y_fb * 3

    calibrator = ConfidenceCalibrator.fit(X, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(output)
    print(f"Calibrator saved → {output}")

    ece = calibration_ece(calibrator, n_races=15, seed=999)
    brier = calibration_brier(calibrator, n_races=15, seed=999)
    quality = {
        "ece": ece,
        "brier_score": brier,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_dataset": {
            "generator": "synthetic+feedback",
            "n_synthetic": len(X_syn),
            "n_feedback": len(feedback_pairs),
            "feedback_weight": 3,
        },
    }
    quality_output.parent.mkdir(parents=True, exist_ok=True)
    quality_output.write_text(json.dumps(quality, indent=2))
    print(f"ECE  → {ece:.4f}  ({'PASS' if ece <= 0.15 else 'FAIL'})")
    print(f"Brier→ {brier:.4f}  ({'PASS' if brier <= 0.15 else 'FAIL'})")
    print(f"Quality → {quality_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain calibrator from human feedback")
    parser.add_argument("--min-feedback", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("data/calibration/isotonic.pkl"))
    parser.add_argument("--quality", type=Path, default=Path("data/calibration/quality.json"))
    args = parser.parse_args()
    main(args.min_feedback, args.output, args.quality)
