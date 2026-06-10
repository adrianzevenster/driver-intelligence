#!/usr/bin/env python
from __future__ import annotations

import json
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1di.confidence.fitting import calibration_brier, calibration_ece, fit_and_save

if __name__ == "__main__":
    output = Path("data/calibration/isotonic.pkl")
    calibrator = fit_and_save(output_path=output, n_races=30, seed=42)

    ece = calibration_ece(calibrator, n_races=15, seed=999)
    brier = calibration_brier(calibrator, n_races=15, seed=999)
    fixture_path = Path("data/fixtures/real_replay_eval.json")
    fixture_cases = json.loads(fixture_path.read_text(encoding="utf-8")) if fixture_path.exists() else []
    fixture_bytes = fixture_path.read_bytes() if fixture_path.exists() else b""
    class_counts = Counter(str(c.get("class", "unknown")) for c in fixture_cases)
    positive_count = sum(1 for c in fixture_cases if "expected_min_risk" in c)
    hard_negative_count = sum(1 for c in fixture_cases if "expected_max_risk" in c)
    quality = {
        "ece": ece,
        "brier_score": brier,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_dataset": {
            "generator": "synthetic",
            "n_races": 30,
            "fit_seed": 42,
            "ece_seed": 999,
        },
        "replay_fixture": {
            "path": str(fixture_path),
            "sha256": hashlib.sha256(fixture_bytes).hexdigest() if fixture_bytes else None,
            "case_count": len(fixture_cases),
            "positive_count": positive_count,
            "hard_negative_count": hard_negative_count,
            "class_counts": dict(sorted(class_counts.items())),
        },
    }
    quality_path = Path("data/calibration/quality.json")
    quality_path.write_text(json.dumps(quality, indent=2))

    print(f"Calibrator saved  → {output}")
    print(f"Calibration ECE   → {ece:.4f}  ({'PASS' if ece <= 0.15 else 'FAIL — exceeds 0.15 threshold'})")
    print(f"Calibration Brier → {brier:.4f}  ({'PASS' if brier <= 0.15 else 'FAIL — exceeds 0.15 threshold'})")
    print(f"Quality metadata  → {quality_path}")
