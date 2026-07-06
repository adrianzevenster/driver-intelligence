from pathlib import Path

import pytest

from f1di.agents.tire import TireStrategyAgent
from f1di.api.main import app_version, health, ready
from f1di.confidence.calibration import ConfidenceCalibrator
from f1di.confidence.fitting import calibration_ece, generate_calibration_dataset
from f1di.domain.schemas import InsightAudience
from f1di.features.extractor import extract_features
from f1di.inference.fusion import InferenceOrchestrator
from f1di.rag.store import HybridMemoryRetriever
from f1di.regression.gates import evaluate_gates
from f1di.simulator.generator import DriverProfile, IncidentPlan, SyntheticRaceSimulator


def test_synthetic_replay_passes_core_gates():
    """Rules-only path (no LLM) must stay under the 250ms p95 latency gate."""
    from unittest.mock import patch

    sim = SyntheticRaceSimulator(seed=123)
    samples = sim.generate_samples(
        session_id="regression-001",
        profile=DriverProfile(driver_id="DRV-REG", braking_aggression=1.2, tire_preservation=0.88),
        laps=8,
        incidents=[IncidentPlan(lap=5, kind="lockup", severity=1.0)],
    )
    windows = sim.rolling_windows(samples, size=10, step=12)
    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())

    # Patch out the LLM so the gate measures rules-engine latency only.
    with patch("f1di.llm.advisor.generate_recommendation", return_value="test"):
        insights = [orchestrator.analyze(w) for w in windows]

    report = evaluate_gates(insights)
    assert report["pass_grounding"]
    assert report["pass_latency"]
    assert any(i.risk.value in {"WARNING", "CRITICAL"} for i in insights)


def test_simulator_write_and_read_jsonl(tmp_path: Path):
    sim = SyntheticRaceSimulator(seed=1)
    windows = sim.rolling_windows(sim.generate_samples(session_id="io", laps=2), size=8, step=8)
    out = tmp_path / "race.jsonl"
    sim.write_jsonl(windows, out)
    assert out.exists()
    assert len(out.read_text().splitlines()) == len(windows)


def test_operational_endpoints_are_available():
    assert health() == {"status": "ok"}
    assert ready()["status"] == "ready"
    assert app_version()["name"] == "f1-driver-intelligence"


def test_calibrator_discriminates_critical_from_nominal():
    X, y = generate_calibration_dataset(n_races=6, seed=99)
    calibrator = ConfidenceCalibrator.fit(X, y)

    sim = SyntheticRaceSimulator(seed=42)
    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    orchestrator.calibrator = calibrator

    nominal_samples = sim.generate_samples(
        session_id="test-nominal",
        laps=2,
        profile=DriverProfile(driver_id="NOM"),
    )
    nominal_insight = orchestrator.analyze(sim.rolling_windows(nominal_samples, size=8, step=1)[0])

    critical_samples = sim.generate_samples(
        session_id="test-critical",
        laps=10,
        profile=DriverProfile(driver_id="CRIT", braking_aggression=1.3, tire_preservation=0.75),
        incidents=[IncidentPlan(lap=7, kind="lockup", severity=1.0)],
    )
    critical_insight = orchestrator.analyze(sim.rolling_windows(critical_samples, size=8, step=12)[-1])

    assert critical_insight.confidence > nominal_insight.confidence
    assert round(critical_insight.confidence, 6) >= 0.70


def test_tire_agent_projected_cliff_triggers_warning():
    sim = SyntheticRaceSimulator(seed=42)
    samples = sim.generate_samples(
        session_id="cliff-test",
        laps=6,
        profile=DriverProfile(driver_id="AGG", braking_aggression=1.3, tire_preservation=0.78),
    )
    windows = sim.rolling_windows(samples, size=12, step=4)
    agent = TireStrategyAgent()
    retriever = HybridMemoryRetriever()

    projected_warning_fired = any(
        agent.analyze(w, extract_features(w), retriever).risk.value == "WARNING"
        for w in windows
        if w.latest.tire_wear_fl < 0.66
    )
    assert projected_warning_fired


def test_warning_insight_shows_to_driver():
    X, y = generate_calibration_dataset(n_races=6, seed=77)
    calibrator = ConfidenceCalibrator.fit(X, y)

    sim = SyntheticRaceSimulator(seed=42)
    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    orchestrator.calibrator = calibrator

    samples = sim.generate_samples(
        session_id="policy-test",
        laps=10,
        profile=DriverProfile(driver_id="WARN", braking_aggression=1.3, tire_preservation=0.75),
        incidents=[IncidentPlan(lap=7, kind="lockup", severity=1.0)],
    )
    windows = sim.rolling_windows(samples, size=8, step=12)
    insights = [orchestrator.analyze(w, audience=InsightAudience.DRIVER) for w in windows]

    warning_or_critical = [i for i in insights if i.risk.value in {"WARNING", "CRITICAL"}]
    assert warning_or_critical, "Expected at least one WARNING/CRITICAL insight"
    assert any(i.policy == "SHOW" for i in warning_or_critical), (
        f"All WARNING/CRITICAL insights suppressed from driver — policies: "
        f"{[i.policy for i in warning_or_critical]}, "
        f"confidences: {[round(i.confidence, 3) for i in warning_or_critical]}"
    )


def test_calibration_ece_within_threshold():
    cal_path = Path("data/calibration/isotonic.pkl")
    if not cal_path.exists():
        pytest.skip("calibrator not fitted — run scripts/fit_calibrator.py first")
    cal = ConfidenceCalibrator.load(cal_path)
    ece = calibration_ece(cal, n_races=10, seed=999)
    assert ece <= 0.15, f"ECE {ece:.4f} exceeds 0.15 threshold — calibration quality has degraded"


def test_tire_agent_axle_imbalance_watch_triggers():
    sim = SyntheticRaceSimulator(seed=42)
    samples = sim.generate_samples(
        session_id="imbalance-test",
        laps=8,
        profile=DriverProfile(driver_id="MED"),
    )
    windows = sim.rolling_windows(samples, size=12, step=4)
    agent = TireStrategyAgent()
    retriever = HybridMemoryRetriever()

    watch_fired = any(
        agent.analyze(w, extract_features(w), retriever).risk.value == "WATCH"
        for w in windows
        if w.latest.tire_wear_fl < 0.66
    )
    assert watch_fired
