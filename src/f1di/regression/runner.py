from __future__ import annotations

import json
from pathlib import Path

from f1di.inference.fusion import InferenceOrchestrator
from f1di.regression.gates import evaluate_gates
from f1di.storage.replay import read_windows


def run_replay(input_jsonl: Path, output_report: Path) -> dict:
    orchestrator = InferenceOrchestrator()
    insights = [orchestrator.analyze(window) for window in read_windows(input_jsonl)]
    report = evaluate_gates(insights)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    failed = [name for name, value in report.items() if name.startswith("pass_") and value is not True]
    if failed:
        raise SystemExit(f"Regression gates failed: {', '.join(failed)}")
    return report
