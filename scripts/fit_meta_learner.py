#!/usr/bin/env python
"""Train the fusion meta-learner (P(insight correct) from 4-agent outputs)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1di.inference.meta_learner import train_from_labels

report = train_from_labels()
active = report["active_in_inference"]
print(f"\nMetaLearner — n_real={report['n_real']}  n_total={report['n_total']}  acc={report['accuracy']:.4f}")
print(f"Active in inference: {'YES (n_real >= 20)' if active else 'NO  (need >= 20 real labels)'}")
dist = report["class_distribution"]
print(f"  correct={dist.get('1', 0)}  incorrect={dist.get('0', 0)}")
print(f"\nSaved → {report['output_path']}")
