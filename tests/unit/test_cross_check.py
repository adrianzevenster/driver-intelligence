"""Unit tests for tire._cross_check() — the classifier/projection agreement logic."""
from __future__ import annotations

import pytest

from f1di.agents.tire import _cross_check
from f1di.domain.schemas import AgentFinding, RiskLevel


def _finding(risk_str: str, conf: float, feats: dict | None = None) -> AgentFinding:
    return AgentFinding(
        agent="tire_strategy",
        risk=RiskLevel[risk_str],
        summary="test summary",
        confidence=conf,
        features=feats or {},
    )


def _cliff(eta: float | None, horizon: int = 15) -> dict:
    probs = {i: float(i) / horizon for i in range(1, horizon + 1)}
    return {"eta_laps": eta, "probability_by_lap": probs, "horizon_laps": horizon, "n_sims": 200}


class TestCrossCheck:
    def test_warning_eta_at_threshold_boosts_confidence(self):
        f = _finding("WARNING", conf=0.75)
        result = _cross_check(f, _cliff(eta=4.0))
        assert result.confidence == pytest.approx(0.80, abs=1e-9)
        assert result.features.get("clf_agrees_cliff") is True

    def test_critical_eta_within_4_boosts_confidence(self):
        f = _finding("CRITICAL", conf=0.80)
        result = _cross_check(f, _cliff(eta=2.0))
        assert result.confidence == pytest.approx(0.85, abs=1e-9)
        assert result.features.get("clf_agrees_cliff") is True

    def test_confidence_boost_capped_at_094(self):
        f = _finding("CRITICAL", conf=0.92)
        result = _cross_check(f, _cliff(eta=1.0))
        assert result.confidence == pytest.approx(0.94, abs=1e-9)

    def test_info_eta_within_3_upgrades_to_warning(self):
        f = _finding("INFO", conf=0.55)
        result = _cross_check(f, _cliff(eta=2.0))
        assert result.risk == RiskLevel.WARNING
        assert result.confidence == pytest.approx(0.68, abs=1e-9)
        assert "Monte Carlo" in result.summary
        assert result.features.get("clf_agrees_cliff") is False

    def test_watch_eta_at_threshold_upgrades_to_warning(self):
        f = _finding("WATCH", conf=0.62)
        result = _cross_check(f, _cliff(eta=3.0))
        assert result.risk == RiskLevel.WARNING

    def test_info_eta_exact_1_summary_uses_singular_lap(self):
        f = _finding("INFO", conf=0.55)
        result = _cross_check(f, _cliff(eta=1.0))
        assert "1 lap:" in result.summary

    def test_critical_no_eta_reduces_confidence(self):
        f = _finding("CRITICAL", conf=0.85)
        result = _cross_check(f, _cliff(eta=None))
        assert result.confidence == pytest.approx(0.80, abs=1e-9)
        assert result.features.get("clf_disagrees_cliff") is True

    def test_critical_no_eta_confidence_floored_at_048(self):
        f = _finding("CRITICAL", conf=0.50)
        result = _cross_check(f, _cliff(eta=None))
        assert result.confidence == pytest.approx(0.48, abs=1e-9)

    def test_no_adjustment_info_no_eta(self):
        # INFO + eta=None → no condition matches → finding returned unchanged
        f = _finding("INFO", conf=0.55)
        result = _cross_check(f, _cliff(eta=None))
        assert result is f

    def test_no_adjustment_warning_eta_beyond_threshold(self):
        # eta=5 > 4 → first branch misses; risk is WARNING not INFO/WATCH → second also misses
        f = _finding("WARNING", conf=0.75)
        result = _cross_check(f, _cliff(eta=5.0))
        assert result is f

    def test_existing_features_preserved_on_agree(self):
        f = _finding("CRITICAL", conf=0.75, feats={"wear_pressure": 0.82})
        result = _cross_check(f, _cliff(eta=3.0))
        assert result.features["wear_pressure"] == pytest.approx(0.82)
        assert result.features.get("clf_agrees_cliff") is True
