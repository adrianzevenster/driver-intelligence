from __future__ import annotations

import pytest


# ── Percentile helper ──────────────────────────────────────────────────────


def test_percentile_median():
    from f1di.agents.threshold_fitter import _percentile
    assert _percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_empty():
    from f1di.agents.threshold_fitter import _percentile
    assert _percentile([], 50) == 0.0


def test_percentile_single():
    from f1di.agents.threshold_fitter import _percentile
    assert _percentile([7.0], 99) == 7.0


# ── Bayesian shrinkage ─────────────────────────────────────────────────────


def test_shrink_full_weight():
    from f1di.agents.threshold_fitter import _shrink
    # n >= min_n → pure circuit estimate
    result = _shrink(0.80, 0.70, n=8, min_n=8)
    assert result == pytest.approx(0.80, abs=0.001)


def test_shrink_zero_weight():
    from f1di.agents.threshold_fitter import _shrink
    # n == 0 → pure prior
    result = _shrink(0.80, 0.70, n=0, min_n=8)
    assert result == pytest.approx(0.70, abs=0.001)


def test_shrink_partial_weight():
    from f1di.agents.threshold_fitter import _shrink
    # n == 4, min_n == 8 → 50/50 blend
    result = _shrink(0.80, 0.70, n=4, min_n=8)
    assert result == pytest.approx(0.75, abs=0.001)


# ── _wear_thresholds_from_stints ───────────────────────────────────────────


def test_wear_thresholds_typical_stints():
    from f1di.agents.threshold_fitter import _wear_thresholds_from_stints
    stints = [15.0] * 5 + [20.0] * 5 + [28.0] * 5 + [35.0] * 5
    crit, warn = _wear_thresholds_from_stints(stints, "silverstone")
    assert warn < crit, "warning threshold must be below critical"
    assert 0.40 <= warn <= 0.82
    assert 0.52 <= crit <= 0.90


def test_wear_thresholds_empty_stints_high_wear_circuit():
    from f1di.agents.threshold_fitter import _wear_thresholds_from_stints, _PRIOR
    crit, warn = _wear_thresholds_from_stints([], "bahrain")
    # High-wear circuit → should be lower than default prior
    assert crit < _PRIOR.wear_critical or crit <= _PRIOR.wear_critical + 0.001


def test_wear_thresholds_empty_stints_low_wear_circuit():
    from f1di.agents.threshold_fitter import _wear_thresholds_from_stints, _PRIOR
    crit, warn = _wear_thresholds_from_stints([], "monaco")
    # Low-wear circuit → should be higher than or equal to default prior
    assert crit > _PRIOR.wear_critical or crit >= _PRIOR.wear_critical - 0.001


def test_wear_thresholds_degenerate_p90_too_small():
    from f1di.agents.threshold_fitter import _wear_thresholds_from_stints, _PRIOR
    # All stints very short → p90 < 3, fall back to prior
    crit, warn = _wear_thresholds_from_stints([1.0] * 10, "silverstone")
    assert crit == _PRIOR.wear_critical
    assert warn == _PRIOR.wear_warning


def test_wear_thresholds_clamps_warn_above_lower_bound():
    from f1di.agents.threshold_fitter import _wear_thresholds_from_stints
    # Extremely early pits would push raw_warn below 0.40
    stints = [1.0, 2.0, 50.0] * 5
    crit, warn = _wear_thresholds_from_stints(stints, "monza")
    assert warn >= 0.40


def test_wear_thresholds_crit_always_above_warn():
    from f1di.agents.threshold_fitter import _wear_thresholds_from_stints
    import random
    rng = random.Random(42)
    for _ in range(50):
        stints = [rng.uniform(5, 40) for _ in range(20)]
        crit, warn = _wear_thresholds_from_stints(stints, "barcelona")
        assert crit >= warn + 0.05, f"crit={crit} warn={warn}"


# ── fit_from_fastf1 (mocked) ───────────────────────────────────────────────


def test_fit_from_fastf1_returns_empty_when_fastf1_missing(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "fastf1", None)
    from f1di.agents import threshold_fitter
    # force re-import to hit the ImportError branch
    import importlib
    importlib.reload(threshold_fitter)
    result = threshold_fitter.fit_from_fastf1(years=[2024], n_per_year=1)
    assert isinstance(result, dict)
    # When fastf1 is unavailable, returns empty dict
    assert len(result) == 0


# ── fit_and_save (no I/O, dry) ─────────────────────────────────────────────


def test_fit_and_save_no_fastf1_returns_report(tmp_path, monkeypatch):
    """fit_and_save gracefully handles fastf1-unavailable case."""
    from unittest.mock import patch

    fake_output = tmp_path / "thresholds.json"

    with patch("f1di.agents.threshold_fitter.fit_from_fastf1", return_value={}):
        from f1di.agents.threshold_fitter import fit_and_save
        report = fit_and_save(years=[2024], n_per_year=2, output_path=fake_output)

    assert "fitted" in report
    assert "skipped" in report
    assert "n_fitted" in report
    assert isinstance(report["fitted"], list)
    assert report["n_fitted"] == 0


def test_fit_and_save_writes_thresholds(tmp_path):
    """fit_and_save writes thresholds.json when data is available."""
    from unittest.mock import patch
    from f1di.agents.thresholds import CircuitThresholds

    fake_thresholds = {
        "silverstone": CircuitThresholds(wear_critical=0.75, wear_warning=0.58),
        "monaco": CircuitThresholds(wear_critical=0.82, wear_warning=0.65),
    }
    fake_output = tmp_path / "thresholds.json"

    with patch("f1di.agents.threshold_fitter.fit_from_fastf1", return_value=fake_thresholds):
        from f1di.agents.threshold_fitter import fit_and_save
        report = fit_and_save(years=[2024], n_per_year=2, output_path=fake_output)

    assert fake_output.exists()
    assert report["n_fitted"] == 2
    assert "silverstone" in report["fitted"]
    assert "monaco" in report["fitted"]


def test_fit_and_save_merge_preserves_existing(tmp_path):
    """merge=True keeps existing circuits not covered by new fit."""
    import json
    from f1di.agents.thresholds import CircuitThresholds

    existing = {
        "bahrain": {
            "wear_critical": 0.68, "wear_warning": 0.52,
            "brake_temp_critical_c": 850.0,
            "fl_degradation_pressure_critical": 0.62,
            "fl_degradation_pressure_warning": 0.44,
            "rain_warning": 0.30,
            "battery_soc_warning": 0.25,
            "crosswind_watch": 40.0,
        }
    }
    fake_output = tmp_path / "thresholds.json"
    fake_output.write_text(json.dumps(existing))

    new_thresholds = {
        "silverstone": CircuitThresholds(wear_critical=0.75, wear_warning=0.60),
    }
    from unittest.mock import patch
    with patch("f1di.agents.threshold_fitter.fit_from_fastf1", return_value=new_thresholds):
        from f1di.agents.threshold_fitter import fit_and_save
        fit_and_save(years=[2024], n_per_year=2, output_path=fake_output, merge=True)

    data = json.loads(fake_output.read_text())
    assert "bahrain" in data, "Existing bahrain entry must be preserved under merge=True"
    assert "silverstone" in data
