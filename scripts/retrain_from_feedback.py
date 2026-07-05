#!/usr/bin/env python
"""Trigger a calibrator + classifier retrain from accumulated feedback.

This is a thin CLI wrapper around the production retrain path in
f1di.confidence.online and f1di.agents.auto_retrain. Prefer the API
endpoints (/retrain/calibrator, /retrain/classifiers) in normal operation;
use this script for manual or CI-triggered runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main(skip_classifiers: bool, min_feedback: int) -> None:
    from f1di.storage.database import init_db
    from f1di.storage.models import FeedbackRecord
    from f1di.storage.database import db_session
    from sqlalchemy import func, select

    init_db()

    with db_session() as session:
        n_feedback = session.scalar(select(func.count()).select_from(FeedbackRecord)) or 0

    print(f"Feedback records in DB: {n_feedback}")
    if n_feedback < min_feedback:
        print(f"Need >= {min_feedback} feedback records to retrain. Exiting without changes.")
        sys.exit(0)

    print("Retraining calibrator …")
    from f1di.confidence.online import retrain
    result = retrain()
    print(json.dumps(result, indent=2))

    if not skip_classifiers:
        print("\nRetraining classifiers …")
        from f1di.agents.auto_retrain import maybe_retrain_all
        maybe_retrain_all(threshold=0)
        print("Classifier retrain complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain calibrator (and optionally classifiers) from feedback")
    parser.add_argument("--min-feedback", type=int, default=20, help="Minimum feedback rows required")
    parser.add_argument("--skip-classifiers", action="store_true", help="Only retrain calibrator")
    args = parser.parse_args()
    main(skip_classifiers=args.skip_classifiers, min_feedback=args.min_feedback)
