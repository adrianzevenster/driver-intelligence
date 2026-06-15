#!/usr/bin/env python
"""Train TelemetryClassifier from synthetic + flywheel-labeled data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1di.agents.telemetry_classifier import train_from_labels

if __name__ == "__main__":
    r = train_from_labels()
    blocked = "  [BLOCKED — regression guard]" if r.get("snapshot_blocked") else ""
    print(f"TelemetryClassifier: n_real={r['n_real']}  n_total={r['n_total']}  acc={r['accuracy']:.4f}{blocked}")
    print(f"  classes:      {r['classes']}")
    print(f"  distribution: {r['class_distribution']}")
    print(f"  saved:        {r['output_path']}")
    if r.get("versioned_path"):
        print(f"  snapshot:     {r['versioned_path']}")
