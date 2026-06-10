from __future__ import annotations

from pathlib import Path

from f1di.inference.fusion import InferenceOrchestrator
from f1di.rag.store import HybridMemoryRetriever
from f1di.regression.real_replay import evaluate_cases, load_cases


def test_labeled_real_replay_fixture_gate_passes():
    cases = load_cases(Path("data/fixtures/real_replay_eval.json"))
    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())

    report = evaluate_cases(cases, orchestrator)

    assert report["pass_case_recall"], report["cases"]
    assert report["pass_nominal_false_positive"], report["cases"]
    assert report["pass_agent_activation"], report["cases"]
    assert report["pass_evidence"], report["cases"]
    assert report["pass_expected_sources"], report["cases"]
    assert report["pass_policy_correctness"], report["cases"]
    assert report["false_positive_rate"] == 0.0
    assert report["agent_activation_rate"] == 1.0
    assert report["source_retrieval_rate"] == 1.0
    assert report["policy_correctness"] == 1.0
    assert report["by_class"]["tire_cliff"]["recall"] == 1.0
    assert report["by_class"]["weather_crossover"]["recall"] == 1.0
    assert report["by_class"]["ers_depletion"]["recall"] == 1.0
    assert report["by_class"]["brake_lockup"]["recall"] == 1.0
    assert report["by_class"]["tire_cliff"]["positive_cases"] >= 2
    assert report["by_class"]["weather_crossover"]["positive_cases"] >= 2
    assert report["by_class"]["ers_depletion"]["positive_cases"] >= 2
    assert report["by_class"]["brake_lockup"]["positive_cases"] >= 2
    assert all("calibration_debug" in case for case in report["cases"])
