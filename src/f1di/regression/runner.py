from __future__ import annotations

import json
import logging
from pathlib import Path

from f1di.evaluation.llm_judge import evaluate_recommendation
from f1di.inference.fusion import InferenceOrchestrator
from f1di.regression.gates import evaluate_gates
from f1di.storage.replay import read_windows

logger = logging.getLogger("f1di.regression.runner")


def run_replay(
    input_jsonl: Path,
    output_report: Path,
    orchestrator: InferenceOrchestrator | None = None,
) -> dict:
    if orchestrator is None:
        orchestrator = InferenceOrchestrator()
    insights = [orchestrator.analyze(window) for window in read_windows(input_jsonl)]

    judge_scores = []
    for insight in insights:
        score = evaluate_recommendation(
            insight.recommendation,
            risk=insight.risk.value,
            audience=insight.audience.value,
        )
        if score is not None:
            judge_scores.append(score)
    if judge_scores:
        logger.info(
            "judge_scores n=%d mean=%.3f",
            len(judge_scores),
            sum(s.mean for s in judge_scores) / len(judge_scores),
        )

    report = evaluate_gates(insights, judge_scores=judge_scores or None)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    # None means the gate was skipped (optional) — only False is a hard failure.
    failed = [name for name, value in report.items() if name.startswith("pass_") and value is False]
    if failed:
        raise SystemExit(f"Regression gates failed: {', '.join(failed)}")
    return report
