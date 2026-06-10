#!/usr/bin/env python
"""Run the LLM judge against all replay fixture cases and save a quality report.

Usage:
    uv run python scripts/run_llm_judge_eval.py
    uv run python scripts/run_llm_judge_eval.py --fixture data/fixtures/real_replay_eval.json
    uv run python scripts/run_llm_judge_eval.py --fail-threshold 0.60

Exit code 1 if the mean judge score falls below --fail-threshold (default 0.65).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_DEFAULT_FIXTURE = Path("data/fixtures/real_replay_eval.json")
_DEFAULT_OUTPUT = Path("data/evaluation/llm_judge_report.json")
_FAIL_THRESHOLD = 0.65


def main(fixture_path: Path, output_path: Path, fail_threshold: float) -> None:
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}")
        sys.exit(1)

    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(cases)} cases from {fixture_path}")

    from f1di.inference.fusion import InferenceOrchestrator
    from f1di.rag.store import HybridMemoryRetriever
    from f1di.regression.real_replay import window_from_case
    from f1di.evaluation.llm_judge import evaluate_recommendation

    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    results = []
    failed_llm = 0

    for i, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{i}")
        audience = case.get("audience", "DRIVER")
        try:
            window = window_from_case(case)
            insight = orchestrator.analyze(window)
        except Exception as exc:
            print(f"  [{i}/{len(cases)}] {case_id}: inference error — {exc}")
            results.append({"case_id": case_id, "error": str(exc)})
            continue

        score = evaluate_recommendation(
            insight.recommendation,
            risk=insight.risk.value,
            audience=audience,
        )
        if score is None:
            failed_llm += 1
            print(f"  [{i}/{len(cases)}] {case_id}: LLM judge unavailable, skipping")
            results.append({"case_id": case_id, "llm_judge_skipped": True,
                            "recommendation": insight.recommendation, "risk": insight.risk.value})
            continue

        result = {
            "case_id": case_id,
            "risk": insight.risk.value,
            "audience": audience,
            "recommendation": insight.recommendation,
            "judge_scores": score.to_dict(),
            "mean_score": round(score.mean, 4),
        }
        results.append(result)
        status = "PASS" if score.mean >= fail_threshold else "FAIL"
        print(
            f"  [{i}/{len(cases)}] {case_id} [{insight.risk.value}] "
            f"mean={score.mean:.2f} ({status}) — {score.rationale}"
        )

    scored = [r for r in results if "mean_score" in r]
    if not scored:
        print("\nNo cases scored — LLM backend may be unavailable.")
        aggregate = {}
    else:
        overall_mean = sum(r["mean_score"] for r in scored) / len(scored)
        dim_means = {}
        for dim in ("safety", "actionability", "register", "calibration"):
            dim_means[dim] = round(
                sum(r["judge_scores"][dim] for r in scored) / len(scored), 4
            )
        aggregate = {
            "n_scored": len(scored),
            "n_skipped": failed_llm,
            "overall_mean": round(overall_mean, 4),
            "by_dimension": dim_means,
            "pass": overall_mean >= fail_threshold,
            "fail_threshold": fail_threshold,
        }
        print(f"\nOverall mean: {overall_mean:.4f}  "
              f"({'PASS' if overall_mean >= fail_threshold else 'FAIL — below threshold'})")
        for dim, val in dim_means.items():
            print(f"  {dim:>14}: {val:.4f}")

    report = {
        "fixture": str(fixture_path),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aggregate": aggregate,
        "cases": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved → {output_path}")

    if scored and not aggregate.get("pass", True):
        print(f"FAIL: overall mean {aggregate['overall_mean']:.4f} < threshold {fail_threshold}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM judge evaluation over replay fixture")
    parser.add_argument("--fixture", type=Path, default=_DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--fail-threshold", type=float, default=_FAIL_THRESHOLD)
    args = parser.parse_args()
    main(args.fixture, args.output, args.fail_threshold)
