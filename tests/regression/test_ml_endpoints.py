from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


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


def test_judge_score_pending_payload_for_saved_unscored_insight():
    _init_db()
    from f1di.api.main import get_judge_score
    from f1di.storage.database import db_session
    from f1di.storage.models import InsightRecord

    insight_id = str(uuid.uuid4())
    with db_session() as session:
        session.add(
            InsightRecord(
                insight_id=insight_id,
                session_id="test-session",
                driver_id="VER",
                track_id="silverstone",
                risk="INFO",
                confidence=0.7,
                uncertainty=0.3,
                policy="ALLOW",
                audience="DRIVER",
                recommendation="Maintain current delta.",
                latency_ms=12.0,
            )
        )

    result = get_judge_score(insight_id)

    assert result == {
        "insight_id": insight_id,
        "status": "pending",
        "scored": False,
    }


def test_judge_score_pending_payload_for_transient_generated_insight():
    _init_db()
    from f1di.api.main import _set_judge_state, get_judge_score

    insight_id = str(uuid.uuid4())
    _set_judge_state(insight_id, "pending")

    result = get_judge_score(insight_id)

    assert result == {
        "insight_id": insight_id,
        "status": "pending",
        "scored": False,
    }


def test_judge_score_returns_scored_payload():
    _init_db()
    from f1di.api.main import get_judge_score
    from f1di.storage.database import db_session
    from f1di.storage.models import InsightRecord, JudgeScoreRecord

    insight_id = str(uuid.uuid4())
    with db_session() as session:
        session.add(
            InsightRecord(
                insight_id=insight_id,
                session_id="test-session",
                driver_id="VER",
                track_id="silverstone",
                risk="WARNING",
                confidence=0.8,
                uncertainty=0.2,
                policy="ALLOW",
                audience="DRIVER",
                recommendation="Brake migration is drifting; reduce entry speed.",
                latency_ms=15.0,
            )
        )
        session.add(
            JudgeScoreRecord(
                insight_id=insight_id,
                safety=0.9,
                actionability=0.8,
                register=0.7,
                calibration=0.6,
                mean_score=0.75,
                rationale="Good score.",
            )
        )

    result = get_judge_score(insight_id)

    assert result["insight_id"] == insight_id
    assert result["status"] == "scored"
    assert result["scored"] is True
    assert result["mean_score"] == 0.75
    assert result["rationale"] == "Good score."


def test_fit_thresholds_empty_telemetry():
    _init_db()
    from f1di.api.main import fit_thresholds_from_telemetry
    result = fit_thresholds_from_telemetry(min_rows=30)
    assert "fitted" in result
    assert "skipped" in result
    assert "total_rows" in result
    assert isinstance(result["fitted"], list)
    assert isinstance(result["skipped"], list)


def test_refresh_drift_status_returns_seed_count(monkeypatch):
    from f1di.api.main import refresh_drift_status
    import f1di.observability.drift as drift

    class FakeTracker:
        def seed_from_db(self, limit: int = 200) -> int:
            assert limit == 25
            return 12

        def status(self) -> dict:
            return {
                "ready": False,
                "baseline_size": 12,
                "min_baseline": 50,
                "features": {},
            }

    monkeypatch.setattr(drift, "get_tracker", lambda: FakeTracker())

    result = refresh_drift_status(limit=25)

    assert result["seeded"] == 12
    assert result["baseline_size"] == 12


def test_capture_fixtures_no_incorrect_predictions():
    _init_db()
    from f1di.api.main import capture_fixtures_from_feedback
    result = capture_fixtures_from_feedback(max_cases=50)
    assert "captured" in result
    assert result["captured"] >= 0


def test_retrain_endpoint_converts_internal_failure_to_http_error():
    from fastapi import HTTPException
    from f1di.api.main import retrain_calibrator
    import f1di.confidence.online as online

    def _boom(*, min_feedback: int = 20):
        raise RuntimeError("training backend unavailable")

    with patch.object(online, "retrain", side_effect=_boom):
        with pytest.raises(HTTPException) as exc_info:
            retrain_calibrator()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Calibrator retrain failed: training backend unavailable"


def test_retrain_regression_guard_blocks_live_model(tmp_path):
    from f1di.confidence.online import retrain

    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    quality_path = cal_dir / "quality.json"
    live_pkl = cal_dir / "isotonic.pkl"
    history_path = cal_dir / "model_history.json"

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
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert history[-1]["regression_detected"] is True


def test_retrain_regression_guard_passes_when_ece_improves(tmp_path):
    from f1di.confidence.online import retrain

    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    quality_path = cal_dir / "quality.json"
    live_pkl = cal_dir / "isotonic.pkl"
    history_path = cal_dir / "model_history.json"

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
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert history[-1]["regression_detected"] is False


def test_model_snapshots_endpoint_surfaces_transfer_lift(tmp_path, monkeypatch):
    """model_snapshots() reads data/calibration relative to cwd, so chdir into a
    tmp dir shaped like the real one rather than touching the repo's actual
    classifier pkls.
    """
    monkeypatch.chdir(tmp_path)
    cal_dir = tmp_path / "data" / "calibration"
    cal_dir.mkdir(parents=True)

    from f1di.agents.safety_car_classifier import SafetyCarClassifier, generate_synthetic
    from f1di.agents.classifier_utils import blend_with_transfer

    X_s, y_s = generate_synthetic(n=400)
    X_r, y_r = generate_synthetic(n=60, seed=1)
    blend = blend_with_transfer(
        SafetyCarClassifier._build_pipeline, X_s, y_s, X_r, y_r, n_real=60,
    )
    clf = SafetyCarClassifier().fit(blend["X"], blend["y"], n_real=60, sample_weight=blend["sample_weight"])
    clf.real_sample_weight = blend["real_weight"]
    clf.prior_cv_accuracy = blend["prior_cv"]["cv_accuracy"]

    snap_path = cal_dir / "safety_car_classifier_20260101T000000Z.pkl"
    import pickle
    snap_path.write_bytes(pickle.dumps(clf))

    from f1di.api.main import model_snapshots
    result = model_snapshots("safety_car")

    assert len(result) == 1
    snap = result[0]
    from f1di.agents.classifier_utils import real_sample_weight, REAL_WEIGHT_FLOOR, REAL_WEIGHT_SATURATION
    expected_w = round(real_sample_weight(60, cap=5.0, floor=REAL_WEIGHT_FLOOR, saturation=REAL_WEIGHT_SATURATION), 4)
    assert snap["real_sample_weight"] == pytest.approx(expected_w, abs=5e-5)
    assert snap["transfer_lift"] is not None
    assert snap["cv_fold_accuracies"] is not None
    assert snap["n_real"] == 60
