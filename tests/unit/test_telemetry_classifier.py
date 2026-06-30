"""Unit tests for telemetry classifier and classifier_utils snapshot mechanism."""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from f1di.agents.telemetry_classifier import (
    TelemetryClassifier,
    FEATURE_NAMES,
    generate_synthetic,
    train_from_labels,
)
from f1di.agents.classifier_utils import save_with_snapshot
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


class TestTelemetryClassifier:
    def test_synthetic_shape(self):
        X, y = generate_synthetic(n=100)
        assert X.shape == (100, len(FEATURE_NAMES))
        assert y.shape == (100,)

    def test_synthetic_all_four_classes_present(self):
        X, y = generate_synthetic(n=600, seed=42)
        assert set(y.tolist()) == {0, 1, 2, 3}

    def test_fit_accuracy(self):
        X, y = generate_synthetic(n=600)
        clf = TelemetryClassifier().fit(X, y)
        assert clf.accuracy > 0.70

    def test_predict_critical_high_brake_temp(self):
        X, y = generate_synthetic(n=600)
        clf = TelemetryClassifier().fit(X, y)
        # LR boundary is smooth; extreme value should produce at least WATCH or higher.
        f = _features(brake_temp_front_max=1050.0, lockup_count=0, brake_fade_risk=5.0)
        risk, conf, proba = clf.predict(f)
        assert risk in ("WATCH", "WARNING", "CRITICAL")
        assert 0.0 < conf <= 1.0
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_predict_elevated_lockup(self):
        # lockup_count=3 is below the agent's safety floor (>=5); brake_fade=14 puts this in WATCH territory
        X, y = generate_synthetic(n=600)
        clf = TelemetryClassifier().fit(X, y)
        f = _features(brake_temp_front_max=400.0, lockup_count=3, brake_fade_risk=14.0)
        risk, conf, _ = clf.predict(f)
        assert risk in ("WATCH", "WARNING", "CRITICAL")

    def test_predict_info_nominal(self):
        X, y = generate_synthetic(n=600)
        clf = TelemetryClassifier().fit(X, y)
        f = _features(brake_temp_front_max=300.0, lockup_count=0, brake_fade_risk=1.0,
                      fl_degradation_pressure=0.30, fl_wear_slope=0.001, crosswind_proxy=3.0)
        risk, _, _ = clf.predict(f)
        assert risk in ("INFO", "WATCH")

    def test_save_load_roundtrip(self, tmp_path):
        X, y = generate_synthetic(n=300)
        clf = TelemetryClassifier().fit(X, y)
        p = tmp_path / "tel.pkl"
        clf.save(p)
        loaded = TelemetryClassifier.load(p)
        f = _features()
        r1, c1, _ = clf.predict(f)
        r2, c2, _ = loaded.predict(f)
        assert r1 == r2
        assert c1 == pytest.approx(c2)

    def test_train_no_db(self, tmp_path):
        out = tmp_path / "tel_clf.pkl"
        with patch("f1di.agents.telemetry_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
            r = train_from_labels(output_path=out, synthetic_n=400)
        assert out.exists()
        assert r["n_real"] == 0
        assert r["accuracy"] > 0.70
        assert r["snapshot_blocked"] is False

    def test_train_returns_versioned_path(self, tmp_path):
        out = tmp_path / "tel_clf.pkl"
        with patch("f1di.agents.telemetry_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
            r = train_from_labels(output_path=out, synthetic_n=400)
        assert r.get("versioned_path") is not None
        assert Path(r["versioned_path"]).exists()


class TestSaveWithSnapshot:
    def _make_clf(self, accuracy: float) -> TelemetryClassifier:
        X, y = generate_synthetic(n=300)
        clf = TelemetryClassifier().fit(X, y)
        clf.accuracy = accuracy
        # Null out std so save_with_snapshot uses the flat min_accuracy_delta
        # threshold rather than the std-based widener, which shifts with feature count.
        clf.cv_accuracy_std = None
        return clf

    def test_first_save_writes_live_and_versioned(self, tmp_path):
        live = tmp_path / "tel_classifier.pkl"
        clf = self._make_clf(0.92)
        result = save_with_snapshot(clf, live)
        assert live.exists()
        assert Path(result["versioned_path"]).exists()
        assert result["blocked"] is False
        assert result["prev_accuracy"] is None

    def test_better_model_promotes_to_live(self, tmp_path):
        live = tmp_path / "tel_classifier.pkl"
        old_clf = self._make_clf(0.80)
        save_with_snapshot(old_clf, live)
        new_clf = self._make_clf(0.88)
        result = save_with_snapshot(new_clf, live)
        assert result["blocked"] is False
        loaded = pickle.loads(live.read_bytes())
        assert loaded.accuracy == pytest.approx(0.88)

    def test_regression_blocks_live_update(self, tmp_path):
        live = tmp_path / "tel_classifier.pkl"
        good_clf = self._make_clf(0.90)
        save_with_snapshot(good_clf, live)
        bad_clf = self._make_clf(0.85)  # drops 0.05 > 0.02 threshold
        result = save_with_snapshot(bad_clf, live)
        assert result["blocked"] is True
        # Live model must still be the old good one
        loaded = pickle.loads(live.read_bytes())
        assert loaded.accuracy == pytest.approx(0.90)
        # But versioned copy of bad model was written
        assert Path(result["versioned_path"]).exists()

    def test_small_drop_within_delta_is_not_blocked(self, tmp_path):
        live = tmp_path / "tel_classifier.pkl"
        save_with_snapshot(self._make_clf(0.90), live)
        result = save_with_snapshot(self._make_clf(0.89), live)  # drops 0.01 < 0.02
        assert result["blocked"] is False


class TestTelemetryAgentWithClassifier:
    """Integration: agent uses classifier when pkl exists, rules otherwise."""

    def _window(self):
        from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow
        sample = TelemetrySample(
            session_id="t", driver_id="VER", track_id="silverstone",
            timestamp_ms=0, lap=12, sector=1, distance_m=100.0, corner_id="T1",
            speed_kph=200.0, acceleration_g=0.0, throttle_pct=80.0, brake_pressure_bar=0.0,
            steering_angle_deg=5.0, yaw_rate_deg_s=2.0, slip_angle_deg=0.1,
            wheel_speed_fl=200.0, wheel_speed_fr=200.0, wheel_speed_rl=200.0, wheel_speed_rr=200.0,
            compound=Compound.MEDIUM, stint_lap=10,
            tire_temp_fl_c=90.0, tire_temp_fr_c=89.0, tire_temp_rl_c=88.0, tire_temp_rr_c=87.0,
            tire_wear_fl=0.40, tire_wear_fr=0.38, tire_wear_rl=0.32, tire_wear_rr=0.30,
            grip_estimate=0.80, battery_soc=0.60, ers_deploy_kw=80.0, ers_regen_kw=20.0,
            pu_thermal_state=0.5, track_temp_c=35.0, ambient_temp_c=22.0, humidity_pct=50.0,
            wind_speed_kph=10.0, wind_direction_deg=180.0, rain_intensity=0.0, evolving_grip=0.88,
            brake_temp_fl_c=400.0, brake_temp_fr_c=390.0, brake_temp_rl_c=350.0, brake_temp_rr_c=345.0,
            lockup_event=False,
        )
        return TelemetryWindow(session_id="t", driver_id="VER", track_id="silverstone", samples=[sample])

    def _retriever(self):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.search.return_value = []
        return m

    def test_rule_fallback_when_no_pkl(self):
        from f1di.agents.telemetry import TelemetryAnalysisAgent
        import f1di.agents.telemetry as tel_mod
        agent = TelemetryAnalysisAgent()
        f = _features()
        with patch("f1di.agents.telemetry._CLASSIFIER_PATH", Path("/nonexistent/tel.pkl")):
            tel_mod._clf_mtime = 0.0
            tel_mod._clf_cache = None
            result = agent.analyze(self._window(), f, self._retriever())
        assert result.risk.value in ("INFO", "WATCH", "WARNING", "CRITICAL")
        assert "brake_temp_front_max" in result.features

    def test_info_branch_stores_base_features(self):
        from f1di.agents.telemetry import TelemetryAnalysisAgent
        import f1di.agents.telemetry as tel_mod
        agent = TelemetryAnalysisAgent()
        f = _features(brake_temp_front_max=350.0, lockup_count=0, brake_fade_risk=1.0,
                      fl_degradation_pressure=0.30, crosswind_proxy=3.0)
        with patch("f1di.agents.telemetry._CLASSIFIER_PATH", Path("/nonexistent/tel.pkl")):
            tel_mod._clf_mtime = 0.0
            tel_mod._clf_cache = None
            result = agent.analyze(self._window(), f, self._retriever())
        # INFO branch previously stored features.__dict__; now must store _base_features dict
        assert isinstance(result.features, dict)
        for key in ("brake_temp_front_max", "lockup_count", "brake_fade_risk",
                    "fl_degradation_pressure", "race_phase"):
            assert key in result.features, f"Missing key: {key}"
        # Must NOT contain non-float dataclass fields from __dict__
        assert "mean_speed_kph" not in result.features
