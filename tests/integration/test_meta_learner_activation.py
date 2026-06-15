"""Integration test: meta-learner activation in InferenceOrchestrator.

Validates the n_real >= 20 gate in fusion.py:
- Below 20 real labels: predict_confidence must NOT be called
- At 20 real labels:  predict_confidence MUST be called and the confidence
  must equal round(0.6 * meta_conf + 0.4 * iso_conf, 4)
"""
from __future__ import annotations

from unittest.mock import patch

from f1di.domain.schemas import TelemetrySample, TelemetryWindow
from f1di.inference.fusion import InferenceOrchestrator
from f1di.inference.meta_learner import MetaLearner, generate_synthetic
from f1di.rag.store import HybridMemoryRetriever


def _sample_dict(lap: int = 10) -> dict:
    return {
        "session_id": "meta_test",
        "driver_id": "HAM",
        "track_id": "monza",
        "timestamp_ms": 1_000 * lap,
        "lap": lap,
        "sector": 1,
        "distance_m": 500.0,
        "speed_kph": 320.0,
        "acceleration_g": 0.3,
        "throttle_pct": 90.0,
        "brake_pressure_bar": 0.0,
        "steering_angle_deg": 2.0,
        "yaw_rate_deg_s": 1.5,
        "slip_angle_deg": 0.5,
        "wheel_speed_fl": 320.0,
        "wheel_speed_fr": 320.0,
        "wheel_speed_rl": 320.0,
        "wheel_speed_rr": 320.0,
        "compound": "SOFT",
        "stint_lap": lap,
        "tire_temp_fl_c": 95.0,
        "tire_temp_fr_c": 95.0,
        "tire_temp_rl_c": 90.0,
        "tire_temp_rr_c": 90.0,
        "tire_wear_fl": 0.20,
        "tire_wear_fr": 0.20,
        "tire_wear_rl": 0.18,
        "tire_wear_rr": 0.18,
        "grip_estimate": 0.88,
        "lockup_event": False,
        "battery_soc": 0.90,
        "ers_deploy_kw": 100.0,
        "ers_regen_kw": 70.0,
        "pu_thermal_state": 0.60,
        "track_temp_c": 38.0,
        "ambient_temp_c": 25.0,
        "humidity_pct": 45.0,
        "wind_speed_kph": 8.0,
        "wind_direction_deg": 90.0,
        "rain_intensity": 0.0,
        "evolving_grip": 0.92,
        "brake_temp_fl_c": 400.0,
        "brake_temp_fr_c": 410.0,
        "brake_temp_rl_c": 310.0,
        "brake_temp_rr_c": 315.0,
    }


def _window() -> TelemetryWindow:
    samples = [TelemetrySample(**_sample_dict(lap=i)) for i in range(1, 4)]
    return TelemetryWindow(
        session_id="meta_test",
        driver_id="HAM",
        track_id="monza",
        race_total_laps=53,
        samples=samples,
    )


def _make_meta(n_real: int) -> MetaLearner:
    X, y = generate_synthetic(n=400, seed=42)
    return MetaLearner().fit(X, y, n_real=n_real)


def _run_insight(meta_or_none):
    orc = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    with patch("f1di.inference.fusion._get_meta_learner", return_value=meta_or_none):
        return orc.analyze(_window())


class TestMetaLearnerActivation:
    def test_meta_learner_inactive_below_threshold(self):
        meta = _make_meta(n_real=19)
        assert meta.n_real == 19

        calls: list = []
        original_predict = meta.predict_confidence

        def tracked_predict(findings, iso_conf):
            calls.append(iso_conf)
            return original_predict(findings, iso_conf)

        meta.predict_confidence = tracked_predict
        _run_insight(meta)
        assert not calls, (
            f"predict_confidence must not be called when n_real={meta.n_real} < 20; "
            f"was called {len(calls)} times"
        )

    def test_meta_learner_active_at_threshold(self):
        meta = _make_meta(n_real=20)
        assert meta.n_real == 20

        captured: list = []

        def sentinel_predict(findings, iso_conf):
            captured.append(iso_conf)
            return 0.88  # deterministic sentinel value

        meta.predict_confidence = sentinel_predict
        insight = _run_insight(meta)

        assert captured, "predict_confidence must be called when n_real >= 20"
        iso_conf = captured[0]
        expected = round(0.6 * 0.88 + 0.4 * iso_conf, 4)
        assert insight.confidence == expected, (
            f"Expected blended confidence {expected} "
            f"(0.6 * 0.88 + 0.4 * {iso_conf:.4f}); got {insight.confidence}"
        )

    def test_meta_learner_confidence_in_valid_range(self):
        meta = _make_meta(n_real=20)
        insight = _run_insight(meta)
        assert 0.0 <= insight.confidence <= 1.0, (
            f"Meta-learner blended confidence {insight.confidence} is out of [0, 1]"
        )
