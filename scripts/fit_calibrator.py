#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1di.confidence.fitting import calibration_ece, fit_and_save

if __name__ == "__main__":
    output = Path("data/calibration/isotonic.pkl")
    calibrator = fit_and_save(output_path=output, n_races=30, seed=42)

    ece = calibration_ece(calibrator, n_races=15, seed=999)
    quality = {"ece": ece, "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    quality_path = Path("data/calibration/quality.json")
    quality_path.write_text(json.dumps(quality, indent=2))

    print(f"Calibrator saved  → {output}")
    print(f"Calibration ECE   → {ece:.4f}  ({'PASS' if ece <= 0.15 else 'FAIL — exceeds 0.15 threshold'})")
    print(f"Quality metadata  → {quality_path}")
