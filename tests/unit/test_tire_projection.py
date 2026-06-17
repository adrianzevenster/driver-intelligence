from __future__ import annotations

from f1di.agents.tire_projection import project_cliff


class TestProjectCliff:
    def test_steep_slope_projects_near_term_crossing_with_high_confidence(self):
        result = project_cliff(
            fl_wear=0.70, fr_wear=0.65,
            fl_wear_slope=0.01, fr_wear_slope=0.008,
            samples_per_lap=4.0, wear_critical=0.78, laps_remaining=20,
            seed=0,
        )
        assert result["eta_laps"] is not None
        assert result["eta_laps"] < 10
        assert result["probability_by_lap"][result["horizon_laps"]] > 0.9

    def test_flat_slope_never_confidently_crosses(self):
        result = project_cliff(
            fl_wear=0.30, fr_wear=0.28,
            fl_wear_slope=0.0, fr_wear_slope=0.0,
            samples_per_lap=4.0, wear_critical=0.78, laps_remaining=20,
            seed=0,
        )
        assert result["eta_laps"] is None
        assert all(p < 0.5 for p in result["probability_by_lap"].values())

    def test_probability_by_lap_is_monotonically_nondecreasing(self):
        result = project_cliff(
            fl_wear=0.60, fr_wear=0.55,
            fl_wear_slope=0.004, fr_wear_slope=0.002,
            samples_per_lap=4.0, wear_critical=0.78, laps_remaining=20,
            seed=1,
        )
        probs = [result["probability_by_lap"][lap] for lap in sorted(result["probability_by_lap"])]
        assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:]))

    def test_binding_tire_is_whichever_wear_is_higher(self):
        # FR is higher and degrading fast; FL is low and flat. The projection
        # should track FR, not average the two or default to FL.
        result_fr_binding = project_cliff(
            fl_wear=0.20, fr_wear=0.74,
            fl_wear_slope=0.0, fr_wear_slope=0.01,
            samples_per_lap=4.0, wear_critical=0.78, laps_remaining=20,
            seed=0,
        )
        assert result_fr_binding["eta_laps"] is not None
        assert result_fr_binding["eta_laps"] < 10

    def test_horizon_capped_when_laps_remaining_is_large(self):
        result = project_cliff(
            fl_wear=0.30, fr_wear=0.28,
            fl_wear_slope=0.001, fr_wear_slope=0.001,
            samples_per_lap=4.0, wear_critical=0.78, laps_remaining=200,
            seed=0,
        )
        assert result["horizon_laps"] == 30

    def test_horizon_matches_laps_remaining_when_small(self):
        result = project_cliff(
            fl_wear=0.30, fr_wear=0.28,
            fl_wear_slope=0.001, fr_wear_slope=0.001,
            samples_per_lap=4.0, wear_critical=0.78, laps_remaining=5,
            seed=0,
        )
        assert result["horizon_laps"] == 5

    def test_deterministic_with_seed(self):
        kwargs = dict(
            fl_wear=0.55, fr_wear=0.50,
            fl_wear_slope=0.003, fr_wear_slope=0.002,
            samples_per_lap=4.0, wear_critical=0.78, laps_remaining=20,
        )
        a = project_cliff(**kwargs, seed=42)
        b = project_cliff(**kwargs, seed=42)
        assert a == b
