"""Unit tests for battery and weather classifiers."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from f1di.agents.battery_classifier import (
    BatteryClassifier, generate_synthetic as bat_synth,
    train_from_labels as bat_train, FEATURE_NAMES as BAT_FEATURES,
)
from f1di.agents.weather_classifier import (
    WeatherClassifier, generate_synthetic as wx_synth,
    train_from_labels as wx_train, FEATURE_NAMES as WX_FEATURES,
)
from f1di.features.extractor import RaceFeatures


def _features(**kw) -> RaceFeatures:
    base = dict(
        lap=15, sector=2, mean_speed_kph=220.0, speed_delta_kph=0.0,
        fl_wear=0.40, fr_wear=0.38, rear_wear_mean=0.30,
        fl_wear_slope=0.001, fr_wear_slope=0.001, rear_wear_slope=0.001,
        axle_imbalance_fl_rl=0.06, brake_temp_front_max=380.0, brake_fade_risk=2.0,
        fl_degradation_pressure=0.33, battery_soc=0.55, battery_soc_slope=-0.004,
        rain_intensity=0.0, crosswind_proxy=4.0, grip_estimate=0.82,
        lockup_count=0, throttle_smoothness=0.88,
        laps_remaining=22.0, stint_fraction=0.55, race_phase=0.45,
    )
    base.update(kw)
    return RaceFeatures(**base)


# ── BatteryClassifier ──────────────────────────────────────────────────────

class TestBatteryClassifier:
    def test_synthetic_shape(self):
        X, y = bat_synth(n=100)
        assert X.shape == (100, len(BAT_FEATURES))
        assert y.shape == (100,)

    def test_synthetic_classes_present(self):
        X, y = bat_synth(n=400, seed=42)
        assert set(y.tolist()) == {0, 1, 2}

    def test_fit_accuracy(self):
        X, y = bat_synth(n=400)
        clf = BatteryClassifier().fit(X, y)
        assert clf.accuracy > 0.70

    def test_predict_warning(self):
        X, y = bat_synth(n=400)
        clf = BatteryClassifier().fit(X, y)
        f = _features(battery_soc=0.10, battery_soc_slope=-0.020, mean_speed_kph=240.0)
        risk, conf, proba = clf.predict(f)
        assert risk in ("WATCH", "WARNING")
        assert 0.0 < conf <= 1.0
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_predict_info(self):
        X, y = bat_synth(n=400)
        clf = BatteryClassifier().fit(X, y)
        f = _features(battery_soc=0.50, battery_soc_slope=-0.003, mean_speed_kph=230.0)
        risk, _, _ = clf.predict(f)
        assert risk in ("INFO", "WATCH")

    def test_save_load_roundtrip(self, tmp_path):
        X, y = bat_synth(n=200)
        clf = BatteryClassifier().fit(X, y)
        p = tmp_path / "bat.pkl"
        clf.save(p)
        loaded = BatteryClassifier.load(p)
        f = _features()
        r1, c1, _ = clf.predict(f)
        r2, c2, _ = loaded.predict(f)
        assert r1 == r2
        assert c1 == pytest.approx(c2)

    def test_train_no_db(self, tmp_path):
        out = tmp_path / "bat_clf.pkl"
        with patch("f1di.agents.battery_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(BAT_FEATURES))), np.empty(0, dtype=np.int32))):
            r = bat_train(output_path=out, synthetic_n=200)
        assert out.exists()
        assert r["n_real"] == 0
        assert r["accuracy"] > 0.70


# ── WeatherClassifier ──────────────────────────────────────────────────────

class TestWeatherClassifier:
    def test_synthetic_shape(self):
        X, y = wx_synth(n=100)
        assert X.shape == (100, len(WX_FEATURES))

    def test_synthetic_classes_present(self):
        X, y = wx_synth(n=400, seed=42)
        assert set(y.tolist()) == {0, 1, 2}

    def test_fit_accuracy(self):
        X, y = wx_synth(n=400)
        clf = WeatherClassifier().fit(X, y)
        assert clf.accuracy > 0.70

    def test_predict_warning_heavy_rain(self):
        X, y = wx_synth(n=400)
        clf = WeatherClassifier().fit(X, y)
        f = _features(rain_intensity=0.55, grip_estimate=0.58)
        risk, conf, _ = clf.predict(f)
        assert risk == "WARNING"

    def test_predict_watch_crosswind(self):
        X, y = wx_synth(n=400)
        clf = WeatherClassifier().fit(X, y)
        f = _features(rain_intensity=0.0, crosswind_proxy=20.0, brake_fade_risk=10.0)
        risk, _, _ = clf.predict(f)
        assert risk in ("WATCH", "WARNING")

    def test_predict_info_dry_calm(self):
        X, y = wx_synth(n=400)
        clf = WeatherClassifier().fit(X, y)
        f = _features(rain_intensity=0.0, crosswind_proxy=2.0, grip_estimate=0.92)
        risk, _, _ = clf.predict(f)
        assert risk == "INFO"

    def test_save_load_roundtrip(self, tmp_path):
        X, y = wx_synth(n=200)
        clf = WeatherClassifier().fit(X, y)
        p = tmp_path / "wx.pkl"
        clf.save(p)
        loaded = WeatherClassifier.load(p)
        f = _features()
        r1, c1, _ = clf.predict(f)
        r2, c2, _ = loaded.predict(f)
        assert r1 == r2
        assert c1 == pytest.approx(c2)

    def test_train_no_db(self, tmp_path):
        out = tmp_path / "wx_clf.pkl"
        with patch("f1di.agents.weather_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(WX_FEATURES))), np.empty(0, dtype=np.int32))):
            r = wx_train(output_path=out, synthetic_n=200)
        assert out.exists()
        assert r["n_real"] == 0
        assert r["accuracy"] > 0.70


# ── Agent integration — classifier path vs rule fallback ──────────────────

class TestAgentClassifierIntegration:
    def test_battery_agent_rule_fallback(self):
        from f1di.agents.battery import BatteryAgent
        from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow
        from unittest.mock import MagicMock

        sample = TelemetrySample(
            session_id="t", driver_id="VER", track_id="monza",
            timestamp_ms=0, lap=10, sector=1, distance_m=0.0, corner_id="T1",
            speed_kph=310.0, acceleration_g=0.0, throttle_pct=90.0, brake_pressure_bar=0.0,
            steering_angle_deg=1.0, yaw_rate_deg_s=0.5, slip_angle_deg=0.0,
            wheel_speed_fl=310.0, wheel_speed_fr=310.0, wheel_speed_rl=310.0, wheel_speed_rr=310.0,
            compound=Compound.MEDIUM, stint_lap=8,
            tire_temp_fl_c=85.0, tire_temp_fr_c=84.0, tire_temp_rl_c=80.0, tire_temp_rr_c=79.0,
            tire_wear_fl=0.30, tire_wear_fr=0.28, tire_wear_rl=0.22, tire_wear_rr=0.20,
            grip_estimate=0.88, battery_soc=0.10, ers_deploy_kw=80.0, ers_regen_kw=5.0,
            pu_thermal_state=0.5, track_temp_c=28.0, ambient_temp_c=18.0, humidity_pct=40.0,
            wind_speed_kph=4.0, wind_direction_deg=0.0, rain_intensity=0.0, evolving_grip=0.92,
            brake_temp_fl_c=280.0, brake_temp_fr_c=275.0, brake_temp_rl_c=240.0, brake_temp_rr_c=238.0,
            lockup_event=False,
        )
        window = TelemetryWindow(session_id="t", driver_id="VER", track_id="monza", samples=[sample])
        retriever = MagicMock()
        retriever.search.return_value = []
        agent = BatteryAgent()

        with patch("f1di.agents.battery._CLASSIFIER_PATH", Path("/nonexistent/bat.pkl")):
            import f1di.agents.battery as bat_mod
            bat_mod._clf_mtime = 0.0
            from f1di.features.extractor import extract_features
            result = agent.analyze(window, extract_features(window), retriever)

        assert result.risk.value in ("INFO", "WATCH", "WARNING")
        # Features are only populated for non-INFO findings; verify the contract
        if result.risk.value != "INFO":
            assert "battery_soc" in result.features

    def test_weather_agent_full_features_stored(self):
        from f1di.agents.weather import WeatherAgent
        from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow
        from unittest.mock import MagicMock

        sample = TelemetrySample(
            session_id="t", driver_id="LEC", track_id="spa",
            timestamp_ms=0, lap=18, sector=2, distance_m=0.0, corner_id="T1",
            speed_kph=240.0, acceleration_g=0.0, throttle_pct=75.0, brake_pressure_bar=0.0,
            steering_angle_deg=8.0, yaw_rate_deg_s=3.0, slip_angle_deg=0.2,
            wheel_speed_fl=240.0, wheel_speed_fr=240.0, wheel_speed_rl=240.0, wheel_speed_rr=240.0,
            compound=Compound.INTERMEDIATE, stint_lap=5,
            tire_temp_fl_c=70.0, tire_temp_fr_c=68.0, tire_temp_rl_c=65.0, tire_temp_rr_c=63.0,
            tire_wear_fl=0.15, tire_wear_fr=0.14, tire_wear_rl=0.12, tire_wear_rr=0.11,
            grip_estimate=0.55, battery_soc=0.65, ers_deploy_kw=40.0, ers_regen_kw=20.0,
            pu_thermal_state=0.3, track_temp_c=18.0, ambient_temp_c=14.0, humidity_pct=85.0,
            wind_speed_kph=22.0, wind_direction_deg=45.0, rain_intensity=0.42, evolving_grip=0.62,
            brake_temp_fl_c=180.0, brake_temp_fr_c=175.0, brake_temp_rl_c=150.0, brake_temp_rr_c=148.0,
            lockup_event=False,
        )
        window = TelemetryWindow(session_id="t", driver_id="LEC", track_id="spa", samples=[sample])
        retriever = MagicMock()
        retriever.search.return_value = []
        agent = WeatherAgent()

        with patch("f1di.agents.weather._CLASSIFIER_PATH", Path("/nonexistent/wx.pkl")):
            import f1di.agents.weather as wx_mod
            wx_mod._clf_mtime = 0.0
            from f1di.features.extractor import extract_features
            result = agent.analyze(window, extract_features(window), retriever)

        assert result.risk.value in ("INFO", "WATCH", "WARNING")
        for key in ("rain_intensity", "grip_estimate", "race_phase"):
            assert key in result.features
