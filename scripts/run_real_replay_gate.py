#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from f1di.inference.fusion import InferenceOrchestrator
from f1di.rag.store import HybridMemoryRetriever
from f1di.regression.real_replay import run_real_replay_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="data/fixtures/real_replay_eval.json")
    parser.add_argument("--output", default="data/scenarios/real_replay_report.json")
    args = parser.parse_args()

    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    with patch("f1di.llm.advisor.generate_recommendation", return_value=""):
        report = run_real_replay_gate(Path(args.fixture), Path(args.output), orchestrator=orchestrator)

    summary = {
        "cases": len(report["cases"]),
        "positive_cases": report["positive_cases"],
        "nominal_cases": report["nominal_cases"],
        "case_recall": report["case_recall"],
        "false_positive_rate": report["false_positive_rate"],
        "agent_activation_rate": report["agent_activation_rate"],
        "source_retrieval_rate": report["source_retrieval_rate"],
        "policy_correctness": report["policy_correctness"],
        "passes": {
            name: value
            for name, value in report.items()
            if name.startswith("pass_")
        },
        "report": args.output,
    }
    print(summary)


if __name__ == "__main__":
    main()
