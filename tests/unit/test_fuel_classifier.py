"""Unit tests for fuel strategy classifier."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from f1di.agents.fuel_classifier import (
    FEATURE_NAMES,
    FuelClassifier,
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
        lockup_count=0, throttle_smoothness=0.80,
        laps_remaining=20.0, stint_fraction=0.5, race_phase=0.4,
        throttle_mean=72.0, ers_net_deploy_kw=40.0,
    )
    defaults.update(overrides)
    return RaceFeatures(**defaults)


class TestSyntheticLabel:
    def test_high_fuel_pressure_warning(self):
        # throttle=90, ers_net=10, soc=0.2, laps=20, smoothness=0.4 → WARNING
        assert _synthetic_label(90.0, 10.0, 0.20, 20.0, 0.5, 0.40, 210.0, 0.5) == 2

    def test_moderate_pressure_watch(self):
        # fp = 0.65 - 0.06 - 0.105 = 0.485; > 0.40 and laps=10 > 6 → WATCH
        assert _synthetic_label(65.0, 30.0, 0.70, 10.0, 0.5, 0.85, 210.0, 0.5) == 1

    def test_early_race_high_throttle_watch(self):
        # race_phase=0.10 < 0.22 and throttle=84 > 82 and soc=0.50 < 0.60 → WATCH
        assert _synthetic_label(84.0, 100.0, 0.50, 5.0, 0.10, 0.90, 210.0, 0.5) == 1

    def test_normal_info(self):
        # low throttle, good ERS, high SOC, few laps → INFO
        assert _synthetic_label(60.0, 80.0, 0.85, 3.0, 0.8, 0.90, 210.0, 0.5) == 0


class TestGenerateSynthetic:
    def test_shape(self):
        X, y = generate_synthetic(n=200, seed=0)
        assert X.shape == (200, len(FEATURE_NAMES))
        assert y.shape == (200,)

    def test_all_three_classes_present(self):
        X, y = generate_synthetic(n=1000, seed=0)
        assert set(np.unique(y)) == {0, 1, 2}

    def test_deterministic(self):
        X1, y1 = generate_synthetic(n=100, seed=5)
        X2, y2 = generate_synthetic(n=100, seed=5)
        np.testing.assert_array_equal(X1, X2)


class TestFeaturesToArray:
    def test_length(self):
        arr = features_to_array(_features())
        assert len(arr) == len(FEATURE_NAMES)

    def test_throttle_mean_first(self):
        arr = features_to_array(_features(throttle_mean=77.5))
        assert arr[0] == pytest.approx(77.5)

    def test_ers_net_second(self):
        arr = features_to_array(_features(ers_net_deploy_kw=55.0))
        assert arr[1] == pytest.approx(55.0)


class TestFuelClassifier:
    def test_fit_predict_shape(self):
        X, y = generate_synthetic(n=300, seed=0)
        clf = FuelClassifier().fit(X, y)
        assert clf.accuracy > 0.5
        assert set(clf.classes_) == {"INFO", "WATCH", "WARNING"}

    def test_predict_valid_output(self):
        X, y = generate_synthetic(n=300, seed=0)
        clf = FuelClassifier().fit(X, y)
        risk, conf, proba = clf.predict(_features())
        assert risk in {"INFO", "WATCH", "WARNING"}
        assert 0.0 < conf <= 1.0
        assert len(proba) == 3
        assert abs(sum(proba) - 1.0) < 1e-6

    def test_high_throttle_no_ers_warns(self):
        X, y = generate_synthetic(n=600, seed=0)
        clf = FuelClassifier().fit(X, y)
        f = _features(throttle_mean=92.0, ers_net_deploy_kw=5.0, battery_soc=0.20,
                      laps_remaining=25.0, throttle_smoothness=0.35)
        risk, _, _ = clf.predict(f)
        assert risk in {"WATCH", "WARNING"}

    def test_efficient_driving_info(self):
        X, y = generate_synthetic(n=600, seed=0)
        clf = FuelClassifier().fit(X, y)
        f = _features(throttle_mean=58.0, ers_net_deploy_kw=90.0, battery_soc=0.88,
                      laps_remaining=2.0, race_phase=0.95)
        risk, _, _ = clf.predict(f)
        assert risk == "INFO"

    def test_ood_score_extreme_input(self):
        X, y = generate_synthetic(n=300, seed=0)
        clf = FuelClassifier().fit(X, y)
        f_normal  = _features()
        f_extreme = _features(throttle_mean=9999.0)
        assert clf.ood_score(f_extreme) > clf.ood_score(f_normal)

    def test_n_real_stored(self):
        X, y = generate_synthetic(n=200, seed=0)
        clf = FuelClassifier().fit(X, y, n_real=12)
        assert clf.n_real == 12

    def test_save_load_roundtrip(self, tmp_path):
        X, y = generate_synthetic(n=300, seed=0)
        clf = FuelClassifier().fit(X, y)
        p = tmp_path / "fuel.pkl"
        clf.save(p)
        clf2 = FuelClassifier.load(p)
        f = _features()
        r1, c1, _ = clf.predict(f)
        r2, c2, _ = clf2.predict(f)
        assert r1 == r2
        assert c1 == pytest.approx(c2)

    def test_model_version(self):
        assert FuelClassifier().model_version == "hgb-v1"
        assert FuelClassifier().model_type == "HistGradientBoosting"

    def test_brier_score_range(self):
        X, y = generate_synthetic(n=300, seed=0)
        clf = FuelClassifier().fit(X, y)
        assert 0.0 <= clf.brier_score <= 2.0


class TestTrainFromLabels:
    def test_synthetic_only(self, tmp_path):
        out = tmp_path / "fuel.pkl"
        with patch("f1di.agents.fuel_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
            report = train_from_labels(output_path=out)
        assert report["n_real"] == 0
        assert report["accuracy"] > 0.5
        assert out.exists()

    def test_blended_with_real(self, tmp_path):
        out = tmp_path / "fuel.pkl"
        X_r, y_r = generate_synthetic(n=15, seed=99)
        with patch("f1di.agents.fuel_classifier._load_labeled_from_db",
                   return_value=(X_r, y_r)):
            report = train_from_labels(output_path=out)
        assert report["n_real"] == 15
        assert report["n_total"] > report["n_synthetic"]

    def test_history_isolated_to_tmp(self, tmp_path):
        out = tmp_path / "fuel.pkl"
        with patch("f1di.agents.fuel_classifier._load_labeled_from_db",
                   return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
            train_from_labels(output_path=out)
        history = tmp_path / "model_history.json"
        assert history.exists()
        import json
        entries = json.loads(history.read_text())
        assert any(e["agent"] == "fuel" for e in entries)
