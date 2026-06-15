"""Integration test: full HTTP inference chain via FastAPI TestClient.

Verifies that POST /v1/insights:
- Returns a valid DriverInsight with all required fields
- Includes 4 findings with classifier schema fields (class_probabilities,
  ood_score, clf_source, ood_flagged) on every finding
- Populates clf_source and class_probabilities when classifiers are patched in
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from f1di.api.main import app


def _sample(lap: int = 10) -> dict:
    return {
        "session_id": "test_session",
        "driver_id": "VER",
        "track_id": "silverstone",
        "timestamp_ms": 1_000 * lap,
        "lap": lap,
        "sector": 2,
        "distance_m": 1_000.0,
        "speed_kph": 280.0,
        "acceleration_g": 0.5,
        "throttle_pct": 75.0,
        "brake_pressure_bar": 0.0,
        "steering_angle_deg": 5.0,
        "yaw_rate_deg_s": 3.0,
        "slip_angle_deg": 1.0,
        "wheel_speed_fl": 280.0,
        "wheel_speed_fr": 280.0,
        "wheel_speed_rl": 280.0,
        "wheel_speed_rr": 280.0,
        "compound": "MEDIUM",
        "stint_lap": lap,
        "tire_temp_fl_c": 90.0,
        "tire_temp_fr_c": 90.0,
        "tire_temp_rl_c": 88.0,
        "tire_temp_rr_c": 88.0,
        "tire_wear_fl": 0.30,
        "tire_wear_fr": 0.30,
        "tire_wear_rl": 0.25,
        "tire_wear_rr": 0.25,
        "grip_estimate": 0.82,
        "lockup_event": False,
        "battery_soc": 0.85,
        "ers_deploy_kw": 120.0,
        "ers_regen_kw": 80.0,
        "pu_thermal_state": 0.65,
        "track_temp_c": 35.0,
        "ambient_temp_c": 22.0,
        "humidity_pct": 55.0,
        "wind_speed_kph": 10.0,
        "wind_direction_deg": 180.0,
        "rain_intensity": 0.0,
        "evolving_grip": 0.90,
        "brake_temp_fl_c": 350.0,
        "brake_temp_fr_c": 360.0,
        "brake_temp_rl_c": 280.0,
        "brake_temp_rr_c": 290.0,
    }


def _window_payload(n_samples: int = 3) -> dict:
    return {
        "session_id": "test_session",
        "driver_id": "VER",
        "track_id": "silverstone",
        "race_total_laps": 52,
        "samples": [_sample(lap=i) for i in range(1, n_samples + 1)],
    }


class _FakeClassifier:
    """Picklable stand-in satisfying all four agent classifier interfaces."""

    classes_ = ["INFO", "WATCH", "WARNING", "CRITICAL"]
    n_real = 5
    accuracy = 0.88
    brier_score = 0.10

    def predict(self, *args, **kwargs):
        return "INFO", 0.72, [0.55, 0.25, 0.14, 0.06]

    def ood_score(self, *args, **kwargs):
        return 1.8  # below the 4.0 penalty threshold


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


class TestInferenceEndpoint:
    def test_inference_returns_valid_driver_insight(self, client):
        resp = client.post("/v1/insights", json=_window_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert "insight_id" in body
        assert "risk" in body
        assert "confidence" in body
        assert "findings" in body
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["risk"] in {"INFO", "WATCH", "WARNING", "CRITICAL"}

    def test_findings_have_classifier_fields(self, client):
        resp = client.post("/v1/insights", json=_window_payload())
        assert resp.status_code == 200
        findings = resp.json()["findings"]
        assert len(findings) == 4
        for f in findings:
            assert "class_probabilities" in f, f"missing class_probabilities in {f['agent']}"
            assert "ood_score" in f, f"missing ood_score in {f['agent']}"
            assert "clf_source" in f, f"missing clf_source in {f['agent']}"
            assert "ood_flagged" in f, f"missing ood_flagged in {f['agent']}"

    def test_findings_use_classifiers_when_pkls_exist(self, client):
        fake = _FakeClassifier()
        with (
            patch("f1di.agents.tire._get_classifier", return_value=fake),
            patch("f1di.agents.battery._get_classifier", return_value=fake),
            patch("f1di.agents.weather._get_classifier", return_value=fake),
            patch("f1di.agents.telemetry._get_classifier", return_value=fake),
        ):
            resp = client.post("/v1/insights", json=_window_payload())

        assert resp.status_code == 200
        findings = resp.json()["findings"]
        classifier_findings = [f for f in findings if f.get("clf_source") == "classifier"]
        assert len(classifier_findings) == 4, (
            f"Expected all 4 findings to use classifier; got {len(classifier_findings)}"
        )
        for f in classifier_findings:
            assert f["class_probabilities"], f"Expected non-empty class_probabilities for {f['agent']}"
            assert set(f["class_probabilities"]) == {"INFO", "WATCH", "WARNING", "CRITICAL"}
            assert isinstance(f["ood_score"], float)
            assert f["ood_flagged"] is False  # ood=1.8 is below the 4.0 penalty threshold
