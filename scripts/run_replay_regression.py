from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from f1di.inference.fusion import InferenceOrchestrator
from f1di.rag.store import HybridMemoryRetriever
from f1di.regression.runner import run_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/scenarios/synthetic_race.jsonl")
    parser.add_argument("--output", default="data/scenarios/regression_report.json")
    args = parser.parse_args()

    # Rules-only path: BM25 retriever + patched LLM so the latency gate
    # measures the inference engine itself, not network/model-load overhead.
    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    with patch("f1di.llm.advisor.generate_recommendation", return_value=""):
        print(run_replay(Path(args.input), Path(args.output), orchestrator=orchestrator))


if __name__ == "__main__":
    main()
