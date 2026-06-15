#!/usr/bin/env python
"""Train the weather strategy logistic regression classifier."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1di.agents.weather_classifier import train_from_labels

report = train_from_labels()
print(f"\nWeatherClassifier — n_real={report['n_real']}  n_total={report['n_total']}  acc={report['accuracy']:.4f}")
for cls in ("INFO", "WATCH", "WARNING"):
    n = report["class_distribution"].get(cls, 0)
    print(f"  {cls:<10} {n:>5}  {'█' * (n // 15)}")
print(f"\nSaved → {report['output_path']}")
