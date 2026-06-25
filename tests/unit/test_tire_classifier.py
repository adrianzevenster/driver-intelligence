"""Unit tests for tire strategy logistic regression classifier."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from f1di.agents.tire_classifier import (
    FEATURE_NAMES,
    TireClassifier,
    _synthetic_label,
    features_to_array,
    generate_synthetic,
    train_from_labels,
)
from f1di.features.extractor import RaceFeatures


# ── Helpers ────────────────────────────────────────────────────────────────

def _features(**kw) -> RaceFeatures:
    base = dict(
        lap=15, sector=2, mean_speed_kph=210.0, speed_delta_kph=0.0,
        fl_wear=0.45, fr_wear=0.43, rear_wear_mean=0.35,
        fl_wear_slope=0.001, fr_wear_slope=0.001, rear_wear_slope=0.001,
        axle_imbalance_fl_rl=0.08, brake_temp_front_max=420.0, brake_fade_risk=1.5,
        fl_degradation_pressure=0.38, battery_soc=0.55, battery_soc_slope=-0.002,
        rain_intensity=0.0, crosswind_proxy=4.0, grip_estimate=0.78,
        lockup_count=0, throttle_smoothness=0.88,
        laps_remaining=22.0, stint_fraction=0.55, race_phase=0.45,
    )
    base.update(kw)
    return RaceFeatures(**base)


# ── features_to_array ──────────────────────────────────────────────────────

def test_features_to_array_length():
    f = _features()
    arr = features_to_array(f, 0.5)
    assert arr.shape == (len(FEATURE_NAMES),)


def test_features_to_array_wear_pressure_first():
    f = _features()
    arr = features_to_array(f, 0.77)
    assert arr[0] == pytest.approx(0.77)


def test_features_to_array_grip_second():
    f = _features(grip_estimate=0.62)
    arr = features_to_array(f, 0.5)
    assert arr[1] == pytest.approx(0.62)


def test_features_to_array_race_phase_position():
    f = _features(race_phase=0.73)
    arr = features_to_array(f, 0.5)
    assert arr[FEATURE_NAMES.index("race_phase")] == pytest.approx(0.73)


# ── _synthetic_label ───────────────────────────────────────────────────────

def test_synthetic_label_critical():
    label = _synthetic_label(0.85, 0.55, 0.002, 0.002, 0.001, 0.10)
    assert label == 3  # CRITICAL

def test_synthetic_label_warning_high_wear():
    label = _synthetic_label(0.70, 0.65, 0.002, 0.002, 0.001, 0.08)
    assert label == 2  # WARNING

def test_synthetic_label_warning_projected_cliff():
    # wear_pressure just below warning, but steep slope projects over cliff in 4 laps
    label = _synthetic_label(0.72, 0.80, 0.018, 0.015, 0.005, 0.05)
    # projected = 0.72 + 0.018*4 = 0.792 > 0.78*0.97 ≈ 0.757
    assert label in (2, 3)

def test_synthetic_label_watch_axle():
    label = _synthetic_label(0.42, 0.85, 0.001, 0.001, 0.001, 0.20)
    assert label in (1, 2)  # axle imbalance > 0.12 + wear proxy > 0.25

def test_synthetic_label_info():
    label = _synthetic_label(0.35, 0.88, 0.0005, 0.0005, 0.0005, 0.05)
    assert label == 0  # INFO


# ── generate_synthetic ─────────────────────────────────────────────────────

def test_generate_synthetic_shape():
    X, y = generate_synthetic(n=100)
    assert X.shape == (100, len(FEATURE_NAMES))
    assert y.shape == (100,)


def test_generate_synthetic_all_classes_present():
    X, y = generate_synthetic(n=800, seed=42)
    unique = set(y.tolist())
    assert unique == {0, 1, 2, 3}, f"Expected all 4 classes, got {unique}"


def test_generate_synthetic_deterministic():
    X1, y1 = generate_synthetic(n=50, seed=7)
    X2, y2 = generate_synthetic(n=50, seed=7)
    assert np.allclose(X1, X2)
    assert (y1 == y2).all()


def test_generate_synthetic_different_seeds():
    X1, _ = generate_synthetic(n=50, seed=1)
    X2, _ = generate_synthetic(n=50, seed=2)
    assert not np.allclose(X1, X2)


# ── TireClassifier fit/predict ─────────────────────────────────────────────

def test_tire_classifier_fit_predict_roundtrip():
    X, y = generate_synthetic(n=400, seed=42)
    clf = TireClassifier().fit(X, y)
    assert clf.accuracy > 0.70
    assert set(clf.classes_) == {"INFO", "WATCH", "WARNING", "CRITICAL"}


def test_tire_classifier_predict_returns_valid_risk():
    X, y = generate_synthetic(n=400, seed=42)
    clf = TireClassifier().fit(X, y)
    f = _features(fl_wear=0.82, fr_wear=0.80, grip_estimate=0.52)
    risk_str, conf, proba = clf.predict(f, wear_pressure=0.82)
    assert risk_str in ("INFO", "WATCH", "WARNING", "CRITICAL")
    assert 0.0 < conf <= 1.0
    assert len(proba) == 4
    assert abs(proba.sum() - 1.0) < 1e-6


def test_tire_classifier_critical_prediction():
    X, y = generate_synthetic(n=800, seed=42)
    clf = TireClassifier().fit(X, y)
    f = _features(fl_wear=0.88, fr_wear=0.86, grip_estimate=0.48)
    risk_str, conf, _ = clf.predict(f, wear_pressure=0.88)
    assert risk_str in ("WARNING", "CRITICAL")


def test_tire_classifier_info_prediction():
    X, y = generate_synthetic(n=800, seed=42)
    clf = TireClassifier().fit(X, y)
    f = _features(fl_wear=0.28, fr_wear=0.26, grip_estimate=0.92, laps_remaining=40.0)
    risk_str, _conf, _ = clf.predict(f, wear_pressure=0.28)
    assert risk_str in ("INFO", "WATCH")


def test_tire_classifier_confidence_in_range():
    X, y = generate_synthetic(n=400, seed=42)
    clf = TireClassifier().fit(X, y)
    for wp in (0.30, 0.55, 0.72, 0.88):
        f = _features()
        _, conf, _ = clf.predict(f, wear_pressure=wp)
        assert 0.0 < conf <= 1.0


def test_tire_classifier_save_load(tmp_path):
    X, y = generate_synthetic(n=200, seed=1)
    clf = TireClassifier().fit(X, y)
    path = tmp_path / "tire_clf.pkl"
    clf.save(path)
    loaded = TireClassifier.load(path)
    f = _features()
    r1, c1, _ = clf.predict(f, 0.5)
    r2, c2, _ = loaded.predict(f, 0.5)
    assert r1 == r2
    assert c1 == pytest.approx(c2)


def test_tire_classifier_n_real_stored():
    X, y = generate_synthetic(n=200, seed=1)
    clf = TireClassifier().fit(X, y, n_real=42)
    assert clf.n_real == 42


# ── train_from_labels (no DB) ──────────────────────────────────────────────

def test_train_from_labels_no_db(tmp_path):
    out = tmp_path / "clf.pkl"
    # Patch DB query to return empty arrays
    with patch("f1di.agents.tire_classifier._load_labeled_from_db",
               return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
        report = train_from_labels(output_path=out, synthetic_n=200)

    assert out.exists()
    assert report["n_real"] == 0
    assert report["n_synthetic"] == 200
    assert report["accuracy"] > 0.70
    assert "class_distribution" in report


def test_train_from_labels_with_real_data(tmp_path):
    out = tmp_path / "clf.pkl"
    # 15 real examples → should trigger blending (threshold is 10)
    real_X = np.random.default_rng(0).uniform(0.3, 0.9, (15, len(FEATURE_NAMES)))
    real_y = np.array([3, 3, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 3], dtype=np.int32)
    with patch("f1di.agents.tire_classifier._load_labeled_from_db",
               return_value=(real_X, real_y)):
        report = train_from_labels(output_path=out, synthetic_n=200, real_oversample=3)

    assert report["n_real"] == 15
    # Real rows are blended via sample_weight now, not literal duplication —
    # n_total is just synthetic + real, one row each.
    assert report["n_total"] == 200 + 15
    # Weight ramps from 1.0 at the n_real=10 floor toward the real_oversample
    # cap (3) as n_real grows toward the saturation point.
    from f1di.agents.classifier_utils import REAL_WEIGHT_FLOOR, REAL_WEIGHT_SATURATION
    growth = min(1.0, (15 - REAL_WEIGHT_FLOOR) / (REAL_WEIGHT_SATURATION - REAL_WEIGHT_FLOOR))
    expected_weight = 1.0 + (3 - 1.0) * growth
    # report rounds to 4 decimal places, so allow up to half a ULP at that precision
    assert report["real_sample_weight"] == pytest.approx(expected_weight, abs=5e-5)
    assert report["prior_accuracy"] is not None
    assert report["transfer_lift"] is not None


# ── TireStrategyAgent with classifier ─────────────────────────────────────

def test_tire_agent_uses_classifier_when_available(tmp_path):
    from f1di.agents.tire import TireStrategyAgent

    X, y = generate_synthetic(n=400, seed=42)
    clf = TireClassifier().fit(X, y)
    clf_path = tmp_path / "tire_clf.pkl"
    clf.save(clf_path)

    retriever = MagicMock()
    retriever.search.return_value = []
    agent = TireStrategyAgent()

    with patch("f1di.agents.tire._CLASSIFIER_PATH", Path(clf_path)):
        import f1di.agents.tire as tire_mod
        tire_mod._clf_mtime = 0.0  # force reload
        features = _features(fl_wear=0.82, fr_wear=0.80, grip_estimate=0.52)
        from f1di.agents.thresholds import CircuitThresholds
        t = CircuitThresholds()
        result = agent._classify(clf, t, features, 0.82, 0.85, 0.83, [])

    assert result.risk.value in ("WARNING", "CRITICAL")
    assert 0.0 < result.confidence <= 1.0
    assert result.summary  # non-empty
    assert "wear_pressure" in result.features


def test_tire_agent_falls_back_to_rules_when_no_classifier():
    from f1di.agents.tire import TireStrategyAgent
    from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow

    sample = TelemetrySample(
        session_id="t", driver_id="VER", track_id="monza",
        timestamp_ms=0, lap=10, sector=1, distance_m=0.0, corner_id="T1",
        speed_kph=310.0, acceleration_g=0.0, throttle_pct=95.0, brake_pressure_bar=0.0,
        steering_angle_deg=1.0, yaw_rate_deg_s=0.5, slip_angle_deg=0.0,
        wheel_speed_fl=310.0, wheel_speed_fr=310.0, wheel_speed_rl=310.0, wheel_speed_rr=310.0,
        compound=Compound.HARD, stint_lap=8,
        tire_temp_fl_c=85.0, tire_temp_fr_c=84.0, tire_temp_rl_c=80.0, tire_temp_rr_c=79.0,
        tire_wear_fl=0.35, tire_wear_fr=0.33, tire_wear_rl=0.28, tire_wear_rr=0.26,
        grip_estimate=0.85, battery_soc=0.65, ers_deploy_kw=80.0, ers_regen_kw=25.0,
        pu_thermal_state=0.5, track_temp_c=30.0, ambient_temp_c=20.0, humidity_pct=40.0,
        wind_speed_kph=5.0, wind_direction_deg=0.0, rain_intensity=0.0, evolving_grip=0.90,
        brake_temp_fl_c=300.0, brake_temp_fr_c=295.0, brake_temp_rl_c=250.0, brake_temp_rr_c=245.0,
        lockup_event=False,
    )
    window = TelemetryWindow(session_id="t", driver_id="VER", track_id="monza", samples=[sample])
    retriever = MagicMock()
    retriever.search.return_value = []

    agent = TireStrategyAgent()
    # Patch so no classifier is found
    with patch("f1di.agents.tire._CLASSIFIER_PATH", Path("/nonexistent/path/tire_clf.pkl")):
        import f1di.agents.tire as tire_mod
        tire_mod._clf_mtime = 0.0
        features = _features(fl_wear=0.35, fr_wear=0.33, grip_estimate=0.85)
        result = agent._rule_based(window, features, 0.35, 0.37, 0.35, [])

    assert result.risk.value in ("INFO", "WATCH", "WARNING", "CRITICAL")
    assert result.summary


def test_tire_agent_analyze_attaches_cliff_projection():
    """analyze() (not the internal _classify/_rule_based helpers) must attach
    the Monte Carlo cliff projection regardless of which path produced the
    risk finding.
    """
    from f1di.agents.tire import TireStrategyAgent
    from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow

    samples = [
        TelemetrySample(
            session_id="t", driver_id="VER", track_id="monza",
            timestamp_ms=i * 1000, lap=10 + i, sector=1, distance_m=0.0, corner_id="T1",
            speed_kph=300.0, acceleration_g=0.0, throttle_pct=90.0, brake_pressure_bar=0.0,
            steering_angle_deg=1.0, yaw_rate_deg_s=0.5, slip_angle_deg=0.0,
            wheel_speed_fl=300.0, wheel_speed_fr=300.0, wheel_speed_rl=300.0, wheel_speed_rr=300.0,
            compound=Compound.MEDIUM, stint_lap=10 + i,
            tire_temp_fl_c=95.0, tire_temp_fr_c=94.0, tire_temp_rl_c=90.0, tire_temp_rr_c=89.0,
            tire_wear_fl=0.55 + i * 0.03, tire_wear_fr=0.50 + i * 0.025, tire_wear_rl=0.40, tire_wear_rr=0.38,
            grip_estimate=0.70, battery_soc=0.55, ers_deploy_kw=80.0, ers_regen_kw=25.0,
            pu_thermal_state=0.5, track_temp_c=30.0, ambient_temp_c=20.0, humidity_pct=40.0,
            wind_speed_kph=5.0, wind_direction_deg=0.0, rain_intensity=0.0, evolving_grip=0.90,
            brake_temp_fl_c=300.0, brake_temp_fr_c=295.0, brake_temp_rl_c=250.0, brake_temp_rr_c=245.0,
            lockup_event=False,
        )
        for i in range(5)
    ]
    window = TelemetryWindow(session_id="t", driver_id="VER", track_id="monza", samples=samples)
    retriever = MagicMock()
    retriever.search.return_value = []

    from f1di.features.extractor import extract_features
    features = extract_features(window)

    agent = TireStrategyAgent()
    with patch("f1di.agents.tire._CLASSIFIER_PATH", Path("/nonexistent/path/tire_clf.pkl")):
        import f1di.agents.tire as tire_mod
        tire_mod._clf_mtime = 0.0
        result = agent.analyze(window, features, retriever)

    assert result.cliff_probability_by_lap is not None
    assert len(result.cliff_probability_by_lap) > 0
    # Steeply rising wear in the fixture should produce a confident, near-term call.
    assert result.cliff_eta_laps is not None


def test_tire_agent_full_features_stored_in_finding():
    """Every AgentFinding must include the classifier feature set for flywheel training."""
    from f1di.agents.tire import TireStrategyAgent
    from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow

    sample = TelemetrySample(
        session_id="t", driver_id="LEC", track_id="monaco",
        timestamp_ms=0, lap=25, sector=3, distance_m=0.0, corner_id="T1",
        speed_kph=160.0, acceleration_g=0.0, throttle_pct=60.0, brake_pressure_bar=0.0,
        steering_angle_deg=15.0, yaw_rate_deg_s=5.0, slip_angle_deg=0.5,
        wheel_speed_fl=160.0, wheel_speed_fr=160.0, wheel_speed_rl=160.0, wheel_speed_rr=160.0,
        compound=Compound.SOFT, stint_lap=20,
        tire_temp_fl_c=110.0, tire_temp_fr_c=108.0, tire_temp_rl_c=100.0, tire_temp_rr_c=98.0,
        tire_wear_fl=0.72, tire_wear_fr=0.70, tire_wear_rl=0.60, tire_wear_rr=0.58,
        grip_estimate=0.68, battery_soc=0.45, ers_deploy_kw=50.0, ers_regen_kw=15.0,
        pu_thermal_state=0.6, track_temp_c=42.0, ambient_temp_c=28.0, humidity_pct=55.0,
        wind_speed_kph=3.0, wind_direction_deg=270.0, rain_intensity=0.0, evolving_grip=0.78,
        brake_temp_fl_c=480.0, brake_temp_fr_c=470.0, brake_temp_rl_c=400.0, brake_temp_rr_c=390.0,
        lockup_event=False,
    )
    window = TelemetryWindow(session_id="t", driver_id="LEC", track_id="monaco", samples=[sample])
    retriever = MagicMock()
    retriever.search.return_value = []

    agent = TireStrategyAgent()
    # Force rule-based path (no classifier file)
    with patch("f1di.agents.tire._CLASSIFIER_PATH", Path("/nonexistent/clf.pkl")):
        import f1di.agents.tire as tire_mod
        tire_mod._clf_mtime = 0.0
        from f1di.features.extractor import extract_features
        features = extract_features(window)
        result = agent.analyze(window, features, retriever)

    # Must include the classifier feature set for flywheel training
    for key in ("wear_pressure", "fl_wear_slope", "fr_wear_slope",
                "laps_remaining", "stint_fraction", "race_phase"):
        assert key in result.features, f"Missing feature '{key}' in finding"
