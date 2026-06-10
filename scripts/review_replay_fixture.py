#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from f1di.inference.fusion import InferenceOrchestrator
from f1di.rag.store import HybridMemoryRetriever
from f1di.regression.real_replay import evaluate_cases, load_cases


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact review table for replay fixture labels.")
    parser.add_argument("--fixture", default="data/fixtures/real_replay_eval.json")
    parser.add_argument("--failed-only", action="store_true")
    args = parser.parse_args()

    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    with patch("f1di.llm.advisor.generate_recommendation", return_value=""):
        report = evaluate_cases(load_cases(Path(args.fixture)), orchestrator)

    rows = [r for r in report["cases"] if r["pass"] or not args.failed_only]
    if args.failed_only:
        rows = [r for r in report["cases"] if not r["pass"]]

    print(
        "case_id,class,source_type,expected,observed,policy,agents,evidence_sources,pass,rationale"
    )
    for row in rows:
        source_type = row.get("source", {}).get("type", "")
        expected = row.get("expected_min_risk") or f"<={row.get('expected_max_risk')}"
        agents = "|".join(row["active_agents"])
        evidence = "|".join(row["evidence_sources"])
        rationale = _clip(row.get("label", {}).get("rationale", "").replace(",", ";"), 120)
        print(
            f"{row['case_id']},{row['class']},{source_type},{expected},"
            f"{row['observed_risk']},{row['observed_policy']},{agents},"
            f"{evidence},{row['pass']},{rationale}"
        )

    print(
        f"summary,cases={len(report['cases'])},recall={report['case_recall']},"
        f"false_positive_rate={report['false_positive_rate']},"
        f"agent_activation={report['agent_activation_rate']},"
        f"source_retrieval={report['source_retrieval_rate']},"
        f"policy_correctness={report['policy_correctness']}"
    )


if __name__ == "__main__":
    main()

