#!/usr/bin/env python
"""Train the fuel strategy classifier."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1di.agents.fuel_classifier import train_from_labels

report = train_from_labels()
blocked = "  [BLOCKED]" if report.get("snapshot_blocked") else ""
print(f"\nFuelClassifier — n_real={report['n_real']}  n_total={report['n_total']}  acc={report['accuracy']:.4f}{blocked}")
print(f"\nClass distribution:")
for cls in ("INFO", "WATCH", "WARNING"):
    n = report["class_distribution"].get(cls, 0)
    print(f"  {cls:<10} {n:>5}  {'█' * (n // 15)}")

per = report.get("per_class", {})
if per:
    print(f"\nPer-class (CV held-out):  {'Class':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print(f"  {'─'*42}")
    for cls in ("INFO", "WATCH", "WARNING"):
        m = per.get(cls)
        if m:
            print(f"  {cls:<10} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6}")

print(f"\nSaved → {report['output_path']}")
