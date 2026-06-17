from __future__ import annotations

from unittest.mock import patch

from f1di.domain.schemas import TelemetryWindow


def _sample(lap: int, wear: float, wear_slope_lap: float, n_in_lap: int = 0) -> dict:
    """One telemetry sample; wear grows by wear_slope_lap per lap, distributed
    evenly across n_in_lap's position within the lap so the window's slope()
    calculation sees a realistic per-sample gradient.
    """
    w = wear + wear_slope_lap * n_in_lap / 4.0
    return {
        "session_id": "test", "driver_id": "VER", "track_id": "bahrain",
        "timestamp_ms": lap * 90_000 + n_in_lap * 1000, "lap": lap, "sector": 2, "distance_m": 1000.0,
        "speed_kph": 280.0, "acceleration_g": 0.5, "throttle_pct": 75.0, "brake_pressure_bar": 0.0,
        "steering_angle_deg": 5.0, "yaw_rate_deg_s": 3.0, "slip_angle_deg": 1.0,
        "wheel_speed_fl": 280.0, "wheel_speed_fr": 280.0, "wheel_speed_rl": 280.0, "wheel_speed_rr": 280.0,
        "compound": "MEDIUM", "stint_lap": lap,
        "tire_temp_fl_c": 90.0, "tire_temp_fr_c": 90.0, "tire_temp_rl_c": 88.0, "tire_temp_rr_c": 88.0,
        "tire_wear_fl": w, "tire_wear_fr": w * 0.95, "tire_wear_rl": w * 0.85, "tire_wear_rr": w * 0.83,
        "grip_estimate": max(0.5, 0.95 - w * 0.3), "lockup_event": False,
        "battery_soc": 0.85, "ers_deploy_kw": 120.0, "ers_regen_kw": 80.0, "pu_thermal_state": 0.65,
        "track_temp_c": 35.0, "ambient_temp_c": 22.0, "humidity_pct": 55.0,
        "wind_speed_kph": 10.0, "wind_direction_deg": 180.0, "rain_intensity": 0.0, "evolving_grip": 0.9,
        "brake_temp_fl_c": 300.0, "brake_temp_fr_c": 295.0, "brake_temp_rl_c": 250.0, "brake_temp_rr_c": 245.0,
    }


def _window(driver: str, lap: int, wear: float, wear_slope_lap: float) -> TelemetryWindow:
    samples = [_sample(lap - 4 + i, wear - wear_slope_lap * (4 - i), wear_slope_lap, n_in_lap=i) for i in range(5)]
    return TelemetryWindow.model_validate({
        "session_id": "test", "driver_id": driver, "track_id": "bahrain",
        "samples": [{**s, "driver_id": driver} for s in samples],
    })


class TestUndercutWindow:
    def test_worn_driver_vs_fresh_rival_low_success_probability(self):
        """Driver heavily worn, rival just pitted (very fresh) — the rival
        isn't going anywhere near their own cliff soon, so the undercut
        shouldn't look favorable.
        """
        from f1di.strategy.undercut import undercut_window

        driver_window = _window("VER", lap=20, wear=0.60, wear_slope_lap=0.01)
        rival_window = _window("HAM", lap=20, wear=0.05, wear_slope_lap=0.005)

        def fake_build_window(*, year, round_num, driver, lap_number=None, session_type="R"):
            return driver_window if driver == "VER" else rival_window

        with patch("f1di.knowledge.fastf1_session.build_window", fake_build_window):
            result = undercut_window(2024, 1, "VER", "HAM", 20)

        assert result["driver"] == "VER"
        assert result["rival"] == "HAM"
        assert result["rival_cliff_eta_laps"] is None  # rival nowhere near their cliff
        assert result["undercut_success_probability"] < 0.5

    def test_both_similarly_worn_gives_higher_success_probability(self):
        """Both drivers similarly worn and close to their own cliff — by the
        time the undercut would pay off, the rival's own tires are also
        likely to have forced a stop, so this should read more favorably
        than the fresh-rival case.
        """
        from f1di.strategy.undercut import undercut_window

        driver_window = _window("VER", lap=20, wear=0.65, wear_slope_lap=0.012)
        rival_window = _window("HAM", lap=20, wear=0.65, wear_slope_lap=0.012)

        def fake_build_window(*, year, round_num, driver, lap_number=None, session_type="R"):
            return driver_window if driver == "VER" else rival_window

        with patch("f1di.knowledge.fastf1_session.build_window", fake_build_window):
            fresh_rival = undercut_window(2024, 1, "VER", "HAM", 20)

        rival_window_fresh = _window("HAM", lap=20, wear=0.03, wear_slope_lap=0.003)

        def fake_build_window_2(*, year, round_num, driver, lap_number=None, session_type="R"):
            return driver_window if driver == "VER" else rival_window_fresh

        with patch("f1di.knowledge.fastf1_session.build_window", fake_build_window_2):
            very_fresh_rival = undercut_window(2024, 1, "VER", "HAM", 20)

        assert fresh_rival["undercut_success_probability"] >= very_fresh_rival["undercut_success_probability"]

    def test_zero_wear_driver_has_no_break_even(self):
        """A driver with essentially no wear gains nothing from a fresh tire,
        so there's no laps_to_break_even and the undercut can't succeed.
        """
        from f1di.strategy.undercut import undercut_window

        driver_window = _window("VER", lap=5, wear=0.0, wear_slope_lap=0.0)
        rival_window = _window("HAM", lap=5, wear=0.0, wear_slope_lap=0.0)

        def fake_build_window(*, year, round_num, driver, lap_number=None, session_type="R"):
            return driver_window if driver == "VER" else rival_window

        with patch("f1di.knowledge.fastf1_session.build_window", fake_build_window):
            result = undercut_window(2024, 1, "VER", "HAM", 5)

        assert result["laps_to_break_even"] is None
        assert result["undercut_success_probability"] == 0.0

    def test_response_includes_model_caveat(self):
        from f1di.strategy.undercut import undercut_window

        driver_window = _window("VER", lap=20, wear=0.5, wear_slope_lap=0.01)
        rival_window = _window("HAM", lap=20, wear=0.3, wear_slope_lap=0.005)

        def fake_build_window(*, year, round_num, driver, lap_number=None, session_type="R"):
            return driver_window if driver == "VER" else rival_window

        with patch("f1di.knowledge.fastf1_session.build_window", fake_build_window):
            result = undercut_window(2024, 1, "VER", "HAM", 20)

        assert "model_caveat" in result
        assert "heuristic" in result["model_caveat"].lower() or "Heuristic" in result["model_caveat"]
