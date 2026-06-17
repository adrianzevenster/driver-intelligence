"""Unit tests for undercut.py private helpers: _break_even_quad, _mc_success_prob, _slope_eff_per_lap."""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from f1di.strategy.undercut import (
    _DEGRADATION_ACCEL,
    _break_even_quad,
    _mc_success_prob,
    _slope_eff_per_lap,
)


def _fake_features(fl_ema=0.0, fr_ema=0.0, fl_slope=0.0, fr_slope=0.0, sf=0.5):
    return types.SimpleNamespace(
        fl_wear_slope_ema=fl_ema,
        fr_wear_slope_ema=fr_ema,
        fl_wear_slope=fl_slope,
        fr_wear_slope=fr_slope,
        stint_fraction=sf,
    )


def _fake_window(n_samples: int = 5, first_lap: int = 10, last_lap: int = 14):
    samples = [types.SimpleNamespace(lap=first_lap + i) for i in range(n_samples)]
    return types.SimpleNamespace(samples=samples, latest=samples[-1])


def _cliff_dict(all_laps_prob: float = 0.0, horizon: int = 15) -> dict:
    """Build a minimal cliff dict where probability_by_lap is uniform."""
    probs = {i: all_laps_prob for i in range(1, horizon + 1)}
    return {"eta_laps": None, "probability_by_lap": probs, "horizon_laps": horizon, "n_sims": 200}


# ── _break_even_quad ──────────────────────────────────────────────────────────


class TestBreakEvenQuad:
    def test_linear_case_zero_slope(self):
        # slope_eff=0 → linear: N = pit_loss_s / (G * wear_rival)
        result = _break_even_quad(pit_loss_s=22.0, wear_rival=0.5, slope_eff=0.0, G=2.0)
        assert result == pytest.approx(22.0, rel=1e-6)

    def test_quadratic_case_reduces_break_even_vs_linear(self):
        # Adding positive slope_eff means later laps are more valuable →
        # break-even lap count is lower than the purely linear case.
        linear = _break_even_quad(pit_loss_s=22.0, wear_rival=0.5, slope_eff=0.0, G=2.0)
        quad = _break_even_quad(pit_loss_s=22.0, wear_rival=0.5, slope_eff=0.01, G=2.0)
        assert quad is not None
        assert quad < linear

    def test_zero_wear_zero_slope_returns_none(self):
        result = _break_even_quad(pit_loss_s=22.0, wear_rival=0.0, slope_eff=0.0, G=2.0)
        assert result is None

    def test_return_none_when_zero_gain_capacity(self):
        # G=0 means no pace gain per wear unit → can never break even
        result = _break_even_quad(pit_loss_s=22.0, wear_rival=0.5, slope_eff=0.0, G=0.0)
        assert result is None

    def test_positive_break_even_lap(self):
        result = _break_even_quad(pit_loss_s=10.0, wear_rival=0.4, slope_eff=0.005, G=2.0)
        assert result is not None
        assert result > 0

    def test_analytical_solution_holds(self):
        # Manually verify the quadratic formula for a known input.
        pit = 10.0
        w = 0.3
        se = 0.02
        G = 2.0
        a = G * se / 2.0
        b = G * w + G * se / 2.0
        c = -pit
        expected = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        result = _break_even_quad(pit_loss_s=pit, wear_rival=w, slope_eff=se, G=G)
        assert result == pytest.approx(expected, rel=1e-9)


# ── _mc_success_prob ──────────────────────────────────────────────────────────


class TestMcSuccessProb:
    def test_zero_gain_produces_zero_probability(self):
        # With wear_rival=0 and slope_eff=0, gain is always 0 which is never > threshold.
        cliff = _cliff_dict(all_laps_prob=1.0, horizon=10)
        result = _mc_success_prob(cliff, wear_rival=0.0, slope_eff_rival=0.0,
                                  pit_loss_s=5.0, gap_s=0.0, G=2.0, seed=0)
        assert result == pytest.approx(0.0)

    def test_guaranteed_gain_produces_probability_one(self):
        # CDF mass on lap 1 only: probs[1]=1.0, rest 0.0.
        # rival_n = 1 for every MC draw.
        # G * (1 * wear_rival + slope_eff * 1 * 2 / 2) = G * (wear_rival + slope_eff)
        # With G=100, wear_rival=1.0, pit_loss_s=0.01 → gain=100 >> threshold.
        probs = {1: 1.0, **{i: 1.0 for i in range(2, 16)}}
        cliff = {"eta_laps": 1.0, "probability_by_lap": probs, "horizon_laps": 15, "n_sims": 200}
        result = _mc_success_prob(cliff, wear_rival=1.0, slope_eff_rival=0.0,
                                  pit_loss_s=0.01, gap_s=0.0, G=100.0, seed=0)
        assert result == pytest.approx(1.0)

    def test_deterministic_with_seed(self):
        cliff = _cliff_dict(all_laps_prob=0.3, horizon=15)
        kwargs = dict(cliff_rival=cliff, wear_rival=0.5, slope_eff_rival=0.005,
                      pit_loss_s=22.0, gap_s=0.0, G=2.0)
        r1 = _mc_success_prob(**kwargs, seed=7)
        r2 = _mc_success_prob(**kwargs, seed=7)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        cliff = _cliff_dict(all_laps_prob=0.4, horizon=15)
        kwargs = dict(cliff_rival=cliff, wear_rival=0.5, slope_eff_rival=0.005,
                      pit_loss_s=22.0, gap_s=0.0, G=2.0)
        # Seeds 0 and 999 almost certainly give different draws over 2000 samples.
        r0 = _mc_success_prob(**kwargs, seed=0)
        r999 = _mc_success_prob(**kwargs, seed=999)
        # We can't guarantee they differ exactly, but result must be in [0, 1].
        assert 0.0 <= r0 <= 1.0
        assert 0.0 <= r999 <= 1.0

    def test_positive_gap_s_reduces_probability(self):
        # A gap means the threshold is higher; prob should not increase.
        cliff = _cliff_dict(all_laps_prob=0.5, horizon=15)
        kwargs = dict(cliff_rival=cliff, wear_rival=0.5, slope_eff_rival=0.01,
                      pit_loss_s=10.0, G=2.0, seed=42)
        no_gap = _mc_success_prob(**kwargs, gap_s=0.0)
        with_gap = _mc_success_prob(**kwargs, gap_s=5.0)
        assert no_gap >= with_gap

    def test_never_crosses_horizon_samples_give_large_n(self):
        # CDF all zeros → rival never crosses in horizon → rival_n = horizon for all draws.
        horizon = 10
        cliff = _cliff_dict(all_laps_prob=0.0, horizon=horizon)
        # gain = G * (horizon * wear) = 2.0 * (10 * 0.5) = 10 > pit_loss_s=5 → always succeed.
        result = _mc_success_prob(cliff, wear_rival=0.5, slope_eff_rival=0.0,
                                  pit_loss_s=5.0, gap_s=0.0, G=2.0, seed=0)
        assert result == pytest.approx(1.0)


# ── _slope_eff_per_lap ────────────────────────────────────────────────────────


class TestSlopeEffPerLap:
    def _call(self, fl_ema=0.008, fr_ema=0.0, sf=0.5) -> float:
        features = _fake_features(fl_ema=fl_ema, fr_ema=fr_ema, sf=sf)
        # 5 samples spanning laps 10–14 (4-lap span): spl = 5/4 = 1.25
        window = _fake_window(n_samples=5, first_lap=10, last_lap=14)
        return _slope_eff_per_lap(features, window)

    def test_zero_stint_fraction_no_acceleration(self):
        spl = 5 / 4  # 5 samples, 4-lap span
        raw = 0.008 * spl
        result = self._call(fl_ema=0.008, sf=0.0)
        assert result == pytest.approx(raw * 1.0, rel=1e-9)

    def test_full_stint_fraction_applies_max_acceleration(self):
        spl = 5 / 4
        raw = 0.008 * spl
        result = self._call(fl_ema=0.008, sf=1.0)
        assert result == pytest.approx(raw * (1.0 + _DEGRADATION_ACCEL), rel=1e-9)

    def test_fl_higher_than_fr_wins(self):
        # FL EMA > FR EMA → FL is binding slope.
        features = _fake_features(fl_ema=0.010, fr_ema=0.006, sf=0.0)
        window = _fake_window(n_samples=5, first_lap=10, last_lap=14)
        result = _slope_eff_per_lap(features, window)
        spl = 5 / 4
        assert result == pytest.approx(0.010 * spl, rel=1e-9)

    def test_fr_higher_than_fl_wins(self):
        features = _fake_features(fl_ema=0.004, fr_ema=0.012, sf=0.0)
        window = _fake_window(n_samples=5, first_lap=10, last_lap=14)
        result = _slope_eff_per_lap(features, window)
        spl = 5 / 4
        assert result == pytest.approx(0.012 * spl, rel=1e-9)

    def test_falls_back_to_plain_slope_when_ema_zero(self):
        # fl_ema=0 → `0.0 or features.fl_wear_slope` triggers the fallback.
        features = _fake_features(fl_ema=0.0, fr_ema=0.0, fl_slope=0.007, fr_slope=0.003, sf=0.0)
        window = _fake_window(n_samples=5, first_lap=10, last_lap=14)
        result = _slope_eff_per_lap(features, window)
        spl = 5 / 4
        assert result == pytest.approx(0.007 * spl, rel=1e-9)

    def test_single_sample_window_uses_spl_one(self):
        # If lap_span == 0 (single sample or same lap), spl defaults to 1.0.
        features = _fake_features(fl_ema=0.008, sf=0.5)
        window = _fake_window(n_samples=1, first_lap=10, last_lap=10)
        result = _slope_eff_per_lap(features, window)
        # spl=1.0; stint_fraction clipped to 0.5
        expected = 0.008 * 1.0 * (1.0 + _DEGRADATION_ACCEL * 0.5)
        assert result == pytest.approx(expected, rel=1e-9)
