#!/usr/bin/env python
"""Train TelemetryClassifier from synthetic + flywheel-labeled data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1di.agents.telemetry_classifier import train_from_labels

if __name__ == "__main__":
    r = train_from_labels()
    blocked = "  [BLOCKED]" if r.get("snapshot_blocked") else ""
    print(f"\nTelemetryClassifier — n_real={r['n_real']}  n_total={r['n_total']}  acc={r['accuracy']:.4f}{blocked}")
    print(f"\nClass distribution:")
    for cls in ("INFO", "WATCH", "WARNING", "CRITICAL"):
        n = r["class_distribution"].get(cls, 0)
        print(f"  {cls:<10} {n:>5}  {'█' * (n // 20)}")

    per = r.get("per_class", {})
    if per:
        print(f"\nPer-class (CV held-out):  {'Class':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
        print(f"  {'─'*42}")
        for cls in ("INFO", "WATCH", "WARNING", "CRITICAL"):
            m = per.get(cls)
            if m:
                print(f"  {cls:<10} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6}")

    print(f"\nSaved → {r['output_path']}")
    if r.get("versioned_path"):
        print(f"Snapshot → {r['versioned_path']}")
