from __future__ import annotations

from f1di.inference.fusion import InferenceOrchestrator
from f1di.rag.store import HybridMemoryRetriever
from f1di.regression.real_replay import build_window

_ORCHESTRATOR = None


def _orchestrator() -> InferenceOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    return _ORCHESTRATOR


def _insight(profile: str, compound: str, lap: int, stint_lap: int, track_id: str, driver_id: str = "VER"):
    window = build_window(
        {
            "case_id": f"rec_quality_{profile}",
            "profile": profile,
            "compound": compound,
            "lap": lap,
            "stint_lap": stint_lap,
            "track_id": track_id,
            "driver_id": driver_id,
        }
    )
    return _orchestrator().analyze(window)


def test_critical_recommendation_contains_stability_language():
    insight = _insight("brake_lockup", "HARD", 37, 22, "singapore")
    rec = insight.recommendation.lower()
    assert any(kw in rec for kw in ("stability", "brake", "tire", "cliff", "intensity", "intervention", "prioriti", "pit", "deployed", "alert")), (
        f"CRITICAL recommendation missing safety/stability language: {insight.recommendation!r}"
    )


def test_tire_warning_recommendation_mentions_pit():
    insight = _insight("front_left_cliff", "MEDIUM", 28, 18, "silverstone")
    rec = insight.recommendation.lower()
    assert any(kw in rec for kw in ("pit", "box", "window", "protect", "tyre", "tire", "degradation")), (
        f"WARNING tire recommendation missing pit/strategy language: {insight.recommendation!r}"
    )


def test_nominal_recommendation_is_continuation():
    insight = _insight("nominal", "MEDIUM", 12, 8, "bahrain")
    rec = insight.recommendation.lower()
    assert any(kw in rec for kw in ("continue", "monitor", "current", "plan", "within")), (
        f"INFO recommendation unexpectedly alarmist: {insight.recommendation!r}"
    )


def test_weather_crossover_recommendation_mentions_conditions():
    insight = _insight("rain_crossover", "MEDIUM", 22, 14, "spa")
    rec = insight.recommendation.lower()
    assert any(kw in rec for kw in ("rain", "weather", "compound", "condition", "intermediate", "switch", "evolv")), (
        f"WARNING weather recommendation missing conditions language: {insight.recommendation!r}"
    )


def test_ers_depletion_recommendation_mentions_deployment():
    insight = _insight("ers_depletion", "SOFT", 19, 10, "spa")
    rec = insight.recommendation.lower()
    assert any(kw in rec for kw in ("ers", "deployment", "battery", "soc", "monitor", "depletion", "adjust")), (
        f"WARNING ERS recommendation missing deployment language: {insight.recommendation!r}"
    )


def test_recommendation_is_non_empty_string():
    for profile, compound, lap, stint_lap, track_id in [
        ("nominal", "MEDIUM", 10, 5, "monza"),
        ("critical_cliff_imminent", "SOFT", 55, 30, "silverstone"),
        ("multi_stress", "MEDIUM", 38, 20, "singapore"),
        ("cold_restart", "MEDIUM", 12, 1, "spa"),
    ]:
        insight = _insight(profile, compound, lap, stint_lap, track_id)
        assert isinstance(insight.recommendation, str)
        assert len(insight.recommendation) > 10, f"Recommendation too short for profile={profile}"
