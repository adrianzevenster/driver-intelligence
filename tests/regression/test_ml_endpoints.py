from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def _init_db():
    from f1di.storage.database import get_engine
    get_engine()


def test_judge_correlation_returns_shape_when_no_data():
    _init_db()
    from f1di.api.main import judge_correlation
    result = judge_correlation()
    assert "r" in result
    assert "n" in result
    assert result["n"] >= 0
    if result["n"] < 3:
        assert result["r"] is None
        assert "message" in result


def test_fit_thresholds_empty_telemetry():
    _init_db()
    from f1di.api.main import fit_thresholds_from_telemetry
    result = fit_thresholds_from_telemetry(min_rows=30)
    assert "fitted" in result
    assert "skipped" in result
    assert "total_rows" in result
    assert isinstance(result["fitted"], list)
    assert isinstance(result["skipped"], list)


def test_capture_fixtures_no_incorrect_predictions():
    _init_db()
    from f1di.api.main import capture_fixtures_from_feedback
    result = capture_fixtures_from_feedback(max_cases=50)
    assert "captured" in result
    assert result["captured"] >= 0


def test_retrain_regression_guard_blocks_live_model(tmp_path):
    from f1di.confidence.online import retrain

    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    quality_path = cal_dir / "quality.json"
    live_pkl = cal_dir / "isotonic.pkl"

    # prev_ece deliberately tiny — any real calibrator ECE will exceed it by >0.01
    quality_path.write_text(json.dumps({"ece": 0.0001, "model_path": "old.pkl"}))

    _pairs = [(0.5, 1.0)] * 22  # 22 pairs, above the default min_feedback=20

    with patch("f1di.confidence.online._feedback_pairs", return_value=_pairs):
        result = retrain(
            min_feedback=20,
            calibrator_path=live_pkl,
            quality_path=quality_path,
        )

    assert result["skipped"] is False
    assert result["regression_detected"] is True
    assert result["live_model_unchanged"] is True
    assert not live_pkl.exists(), "live isotonic.pkl must NOT be written on regression"

    versioned = Path(result["versioned_model_path"])
    assert versioned.exists(), "versioned pkl must always be saved for audit"

    updated = json.loads(quality_path.read_text())
    assert updated["regression_detected"] is True


def test_retrain_regression_guard_passes_when_ece_improves(tmp_path):
    from f1di.confidence.online import retrain

    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    quality_path = cal_dir / "quality.json"
    live_pkl = cal_dir / "isotonic.pkl"

    # prev_ece very high — retrain should always beat this
    quality_path.write_text(json.dumps({"ece": 0.99, "model_path": "old.pkl"}))

    _pairs = [(0.7, 1.0)] * 11 + [(0.3, 0.0)] * 11  # 22 well-separated pairs

    with patch("f1di.confidence.online._feedback_pairs", return_value=_pairs):
        result = retrain(
            min_feedback=20,
            calibrator_path=live_pkl,
            quality_path=quality_path,
        )

    assert result["skipped"] is False
    assert result["regression_detected"] is False
    assert live_pkl.exists(), "live isotonic.pkl must be written when ECE passes the guard"
