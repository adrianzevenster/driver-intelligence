"""Unit tests for safety car / VSC risk classifier."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from f1di.agents.safety_car_classifier import (
    FEATURE_NAMES,
    SafetyCarClassifier,
    _synthetic_label,
    features_to_array,
    generate_synthetic,
    train_from_labels,
)
from f1di.features.extractor import RaceFeatures


def _features(**overrides) -> RaceFeatures:
    defaults = dict(
        lap=20, sector=2,
        mean_speed_kph=230.0, speed_delta_kph=0.0,
        fl_wear=0.40, fr_wear=0.38, rear_wear_mean=0.30,
        fl_wear_slope=0.001, fr_wear_slope=0.001, rear_wear_slope=0.001,
        axle_imbalance_fl_rl=0.05,
        brake_temp_front_max=350.0, brake_fade_risk=0.0,
        fl_degradation_pressure=0.25,
        battery_soc=0.65, battery_soc_slope=-0.005,
        rain_intensity=0.0, crosswind_proxy=0.5,
        grip_estimate=0.90,
        lockup_count=0, throttle_smoothness=0.85,
        laps_remaining=25.0, stint_fraction=0.5, race_phase=0.4,
    )
    defaults.update(overrides)
    return RaceFeatures(**defaults)


class TestSyntheticLabel:
    def test_sc_deployed_speed(self):
        assert _synthetic_label(60.0, 0.0, 0.0, 0.9, 0, 300.0, 0.85, 0.5) == 3

    def test_extreme_rain_critical(self):
        # rain > 0.75 and grip < 0.50 → CRITICAL
        assert _synthetic_label(200.0, 0.0, 0.85, 0.49, 0, 300.0, 0.85, 0.5) == 3

    def test_low_speed_warning(self):
        assert _synthetic_label(140.0, 0.0, 0.0, 0.9, 0, 300.0, 0.85, 0.5) == 2

    def test_big_delta_warning(self):
        assert _synthetic_label(200.0, -70.0, 0.0, 0.9, 0, 300.0, 0.85, 0.5) == 2

    def test_moderate_rain_watch(self):
        assert _synthetic_label(220.0, 0.0, 0.45, 0.9, 0, 300.0, 0.85, 0.5) == 1

    def test_normal_info(self):
        assert _synthetic_label(240.0, 5.0, 0.05, 0.92, 0, 300.0, 0.85, 0.5) == 0


class TestGenerateSynthetic:
    def test_shape(self):
        X, y = generate_synthetic(n=200, seed=0)
        assert X.shape == (200, len(FEATURE_NAMES))
        assert y.shape == (200,)

    def test_all_classes_present(self):
        X, y = generate_synthetic(n=2000, seed=0)
        assert set(np.unique(y)) == {0, 1, 2, 3}

    def test_deterministic(self):
        X1, y1 = generate_synthetic(n=100, seed=7)
        X2, y2 = generate_synthetic(n=100, seed=7)
        np.testing.assert_array_equal(X1, X2)

    def test_seed_variation(self):
        _, y1 = generate_synthetic(n=100, seed=1)
        _, y2 = generate_synthetic(n=100, seed=2)
        assert not np.array_equal(y1, y2)


class TestFeaturesToArray:
    def test_length(self):
        f = _features()
        arr = features_to_array(f)
        assert len(arr) == len(FEATURE_NAMES)

    def test_values(self):
        f = _features(mean_speed_kph=180.0, rain_intensity=0.3)
        arr = features_to_array(f)
        assert arr[0] == pytest.approx(180.0)
        assert arr[2] == pytest.approx(0.3)


class TestSafetyCarClassifier:
    def test_fit_predict(self):
        X, y = generate_synthetic(n=400, seed=0)
        clf = SafetyCarClassifier().fit(X, y)
        assert clf.accuracy > 0.5
        assert clf.n_train == 400
        assert set(clf.classes_) == {"INFO", "WATCH", "WARNING", "CRITICAL"}

    def test_predict_returns_valid_class(self):
        X, y = generate_synthetic(n=400, seed=0)
        clf = SafetyCarClassifier().fit(X, y)
        f = _features()
        risk, conf, proba = clf.predict(f)
        assert risk in {"INFO", "WATCH", "WARNING", "CRITICAL"}
        assert 0.0 < conf <= 1.0
        assert len(proba) == 4
        assert abs(sum(proba) - 1.0) < 1e-6

    def test_sc_speed_predicts_high_risk(self):
        X, y = generate_synthetic(n=800, seed=0)
        clf = SafetyCarClassifier().fit(X, y)
        f = _features(mean_speed_kph=60.0, speed_delta_kph=-80.0)
        risk, _, _ = clf.predict(f)
        assert risk in {"WARNING", "CRITICAL"}

    def test_normal_conditions_info(self):
        X, y = generate_synthetic(n=800, seed=0)
        clf = SafetyCarClassifier().fit(X, y)
        f = _features(mean_speed_kph=260.0, speed_delta_kph=5.0, rain_intensity=0.0, grip_estimate=0.95)
        risk, _, _ = clf.predict(f)
        assert risk in {"INFO", "WATCH"}

    def test_ood_score(self):
        X, y = generate_synthetic(n=400, seed=0)
        clf = SafetyCarClassifier().fit(X, y)
        f_normal = _features()
        f_extreme = _features(mean_speed_kph=5000.0)
        assert clf.ood_score(f_extreme) > clf.ood_score(f_normal)

    def test_n_real_stored(self):
        X, y = generate_synthetic(n=200, seed=0)
        clf = SafetyCarClassifier().fit(X, y, n_real=7)
        assert clf.n_real == 7

    def test_save_load_roundtrip(self, tmp_path):
        X, y = generate_synthetic(n=300, seed=0)
        clf = SafetyCarClassifier().fit(X, y)
        p = tmp_path / "sc.pkl"
        clf.save(p)
        clf2 = SafetyCarClassifier.load(p)
        f = _features()
        r1, c1, _ = clf.predict(f)
        r2, c2, _ = clf2.predict(f)
        assert r1 == r2
        assert c1 == pytest.approx(c2)

    def test_model_version_set(self):
        clf = SafetyCarClassifier()
        assert clf.model_version == "hgb-v1"
        assert clf.model_type == "HistGradientBoosting"

    def test_brier_score_range(self):
        X, y = generate_synthetic(n=400, seed=0)
        clf = SafetyCarClassifier().fit(X, y)
        assert 0.0 <= clf.brier_score <= 2.0


class TestTrainFromLabels:
    def test_synthetic_only(self, tmp_path):
        out = tmp_path / "sc.pkl"
        with patch("f1di.agents.safety_car_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
            report = train_from_labels(output_path=out)
        assert report["n_real"] == 0
        assert report["accuracy"] > 0.5
        assert out.exists()
        assert report["n_synthetic"] > 0

    def test_blended_with_real(self, tmp_path):
        out = tmp_path / "sc.pkl"
        X_r, y_r = generate_synthetic(n=20, seed=99)
        with patch("f1di.agents.safety_car_classifier._load_labeled_from_db",
                   return_value=(X_r, y_r)):
            report = train_from_labels(output_path=out)
        assert report["n_real"] == 20
        assert report["n_total"] > report["n_synthetic"]

    def test_snapshot_written(self, tmp_path):
        out = tmp_path / "sc.pkl"
        with patch("f1di.agents.safety_car_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
            report = train_from_labels(output_path=out)
        assert report["versioned_path"]
        assert Path(report["versioned_path"]).exists()

    def test_history_written_to_tmp(self, tmp_path):
        out = tmp_path / "sc.pkl"
        with patch("f1di.agents.safety_car_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
            train_from_labels(output_path=out)
        history = tmp_path / "model_history.json"
        assert history.exists()
        import json
        entries = json.loads(history.read_text())
        assert any(e["agent"] == "safety_car" for e in entries)
