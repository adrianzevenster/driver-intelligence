"""Integration test: full flywheel loop with a real SQLite DB.

Verifies the path: write InsightRecord + FeedbackRecord → _load_labeled_from_db()
picks them up → train_from_labels() produces a blended model with n_real > 0.

No mocks of the DB layer — uses a fresh in-memory SQLite engine via monkeypatch
of the module-level _engine singleton in f1di.storage.database.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from f1di.storage.models import Base, FeedbackRecord, InsightRecord


@pytest.fixture()
def sqlite_session(monkeypatch, tmp_path):
    """Patch the storage engine with a fresh in-memory SQLite DB."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    import f1di.storage.database as db_mod
    monkeypatch.setattr(db_mod, "_engine", engine)
    monkeypatch.setattr(db_mod, "_SessionLocal", factory)

    session = factory()
    yield session
    session.close()


def _write_insight(session, insight_id: str, agent: str, risk: str, features: dict) -> InsightRecord:
    findings = [{"agent": agent, "risk": risk, "confidence": 0.80, "features": features}]
    rec = InsightRecord(
        insight_id=insight_id,
        session_id="int_test",
        driver_id="VER",
        track_id="silverstone",
        lap=12,
        compound="MEDIUM",
        risk=risk,
        confidence=0.80,
        uncertainty=0.20,
        raw_score=0.6,
        policy="monitor",
        audience="engineer",
        recommendation="Test recommendation.",
        findings_json=json.dumps(findings),
        evidence_json="[]",
        latency_ms=50.0,
        shadow=False,
    )
    session.add(rec)
    session.commit()
    return rec


def _write_feedback(session, insight_id: str, correct: bool) -> FeedbackRecord:
    rec = FeedbackRecord(
        insight_id=insight_id,
        rating=5 if correct else 2,
        correct=correct,
    )
    session.add(rec)
    session.commit()
    return rec


class TestTireFlywheelLoop:
    def test_blended_training_with_real_labels(self, sqlite_session, tmp_path):
        for i in range(15):
            iid = str(uuid.uuid4())
            _write_insight(sqlite_session, iid, "tire_strategy", "WARNING", {
                "wear_pressure": 0.70 + i * 0.01,
                "grip_estimate": 0.68,
                "fl_wear_slope": 0.004,
                "fr_wear_slope": 0.003,
                "rear_wear_slope": 0.002,
                "axle_imbalance_fl_rl": 0.10,
                "laps_remaining": float(20 - i),
                "stint_fraction": 0.6,
                "race_phase": 0.5,
            })
            _write_feedback(sqlite_session, iid, correct=True)

        out = tmp_path / "tire_clf.pkl"
        from f1di.agents.tire_classifier import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=400)

        assert out.exists(), "Live classifier pkl must be written"
        assert result["n_real"] == 15
        assert result["accuracy"] > 0.70
        assert "brier_score" not in result or True  # brier_score is on the clf object
        # Load and verify brier_score attribute is present
        import pickle
        clf = pickle.loads(out.read_bytes())
        assert hasattr(clf, "brier_score")
        assert 0.0 <= clf.brier_score <= 1.0
        assert result.get("snapshot_blocked") is False

    def test_snapshot_written_alongside_live(self, sqlite_session, tmp_path):
        out = tmp_path / "tire_clf.pkl"
        from f1di.agents.tire_classifier import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=300)
        assert out.exists()
        assert result.get("versioned_path") is not None
        assert Path(result["versioned_path"]).exists()
        # Versioned and live must differ in name but same content
        live = out.read_bytes()
        snap = Path(result["versioned_path"]).read_bytes()
        assert live == snap  # shutil.copy2, so bytes are identical


class TestBatteryFlywheelLoop:
    def test_blended_training_with_real_labels(self, sqlite_session, tmp_path):
        for i in range(12):
            iid = str(uuid.uuid4())
            _write_insight(sqlite_session, iid, "battery", "WARNING", {
                "battery_soc": 0.18 - i * 0.005,
                "battery_soc_slope": -0.015,
                "mean_speed_kph": 260.0,
                "race_phase": 0.5,
                "laps_remaining": 25.0,
                "stint_fraction": 0.6,
            })
            _write_feedback(sqlite_session, iid, correct=True)

        out = tmp_path / "bat_clf.pkl"
        from f1di.agents.battery_classifier import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=400)

        assert out.exists()
        assert result["n_real"] == 12
        assert result["accuracy"] > 0.70
        import pickle
        clf = pickle.loads(out.read_bytes())
        assert hasattr(clf, "brier_score")
        assert hasattr(clf, "ood_score")


class TestMetaLearnerFlywheelLoop:
    def test_not_active_below_threshold(self, sqlite_session, tmp_path):
        # Write 15 insights (below the 20-label inference threshold)
        for i in range(15):
            iid = str(uuid.uuid4())
            findings = [
                {"agent": "tire_strategy", "risk": "WARNING", "confidence": 0.82, "features": {}},
                {"agent": "battery", "risk": "INFO", "confidence": 0.60, "features": {}},
                {"agent": "weather", "risk": "INFO", "confidence": 0.55, "features": {}},
                {"agent": "telemetry", "risk": "WATCH", "confidence": 0.65, "features": {}},
            ]
            rec = InsightRecord(
                insight_id=iid, session_id="t", driver_id="VER", track_id="silverstone",
                lap=10, compound="MEDIUM", risk="WARNING", confidence=0.78, uncertainty=0.22,
                raw_score=0.6, policy="monitor", audience="engineer",
                recommendation="Test.", findings_json=json.dumps(findings),
                evidence_json="[]", latency_ms=50.0, shadow=False,
            )
            sqlite_session.add(rec)
            sqlite_session.commit()
            _write_feedback(sqlite_session, iid, correct=True)

        out = tmp_path / "meta.pkl"
        from f1di.inference.meta_learner import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=400)

        assert out.exists()
        assert result["n_real"] == 15
        assert result["active_in_inference"] is False  # still below 20

    def test_active_at_threshold(self, sqlite_session, tmp_path):
        for i in range(22):
            iid = str(uuid.uuid4())
            findings = [
                {"agent": "tire_strategy", "risk": "CRITICAL", "confidence": 0.88, "features": {}},
                {"agent": "battery", "risk": "WARNING", "confidence": 0.75, "features": {}},
                {"agent": "weather", "risk": "WATCH", "confidence": 0.62, "features": {}},
                {"agent": "telemetry", "risk": "WARNING", "confidence": 0.79, "features": {}},
            ]
            rec = InsightRecord(
                insight_id=iid, session_id="t", driver_id="VER", track_id="monza",
                lap=10, compound="SOFT", risk="CRITICAL", confidence=0.85, uncertainty=0.15,
                raw_score=0.8, policy="pit_now", audience="engineer",
                recommendation="Test.", findings_json=json.dumps(findings),
                evidence_json="[]", latency_ms=50.0, shadow=False,
            )
            sqlite_session.add(rec)
            sqlite_session.commit()
            _write_feedback(sqlite_session, iid, correct=i % 3 != 0)  # mix correct/incorrect

        out = tmp_path / "meta.pkl"
        from f1di.inference.meta_learner import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=400)

        assert result["n_real"] == 22
        assert result["active_in_inference"] is True
        import pickle
        ml = pickle.loads(out.read_bytes())
        assert hasattr(ml, "brier_score")
        assert 0.0 <= ml.brier_score <= 1.0
        assert hasattr(ml, "ood_score")


class TestSafetyCarFlywheelLoop:
    def test_blended_training_with_real_labels(self, sqlite_session, tmp_path):
        for i in range(12):
            iid = str(uuid.uuid4())
            _write_insight(sqlite_session, iid, "safety_car", "WARNING", {
                "mean_speed_kph": 90.0 - i * 2,
                "speed_delta_kph": -50.0 - i,
                "rain_intensity": 0.55 + i * 0.01,
                "grip_estimate": 0.55 - i * 0.01,
                "lockup_count": float(i % 3),
                "throttle_smoothness": 0.50,
                "race_phase": 0.45,
                "brake_temp_front_max": 420.0,
            })
            _write_feedback(sqlite_session, iid, correct=True)

        out = tmp_path / "sc_clf.pkl"
        from f1di.agents.safety_car_classifier import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=400)

        assert out.exists()
        assert result["n_real"] == 12
        assert result["accuracy"] > 0.65
        import pickle
        clf = pickle.loads(out.read_bytes())
        assert hasattr(clf, "brier_score")
        assert hasattr(clf, "ood_score")
        assert result.get("snapshot_blocked") is False

    def test_snapshot_written_alongside_live(self, sqlite_session, tmp_path):
        out = tmp_path / "sc_clf.pkl"
        from f1di.agents.safety_car_classifier import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=300)
        assert out.exists()
        assert Path(result["versioned_path"]).exists()


class TestFuelFlywheelLoop:
    def test_blended_training_with_real_labels(self, sqlite_session, tmp_path):
        for i in range(12):
            iid = str(uuid.uuid4())
            _write_insight(sqlite_session, iid, "fuel", "WARNING", {
                "throttle_mean": 82.0 + i * 0.5,
                "ers_net_deploy_kw": 45.0 + i,
                "battery_soc": 0.60 - i * 0.02,
                "laps_remaining": float(35 - i),
                "race_phase": 0.35 + i * 0.02,
                "stint_fraction": 0.55 + i * 0.02,
                "throttle_smoothness": 0.55,
            })
            _write_feedback(sqlite_session, iid, correct=True)

        out = tmp_path / "fuel_clf.pkl"
        from f1di.agents.fuel_classifier import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=400)

        assert out.exists()
        assert result["n_real"] == 12
        assert result["accuracy"] > 0.65
        import pickle
        clf = pickle.loads(out.read_bytes())
        assert hasattr(clf, "brier_score")
        assert hasattr(clf, "ood_score")
        assert result.get("snapshot_blocked") is False

    def test_snapshot_written_alongside_live(self, sqlite_session, tmp_path):
        out = tmp_path / "fuel_clf.pkl"
        from f1di.agents.fuel_classifier import train_from_labels
        result = train_from_labels(output_path=out, synthetic_n=300)
        assert out.exists()
        assert Path(result["versioned_path"]).exists()


class TestRegressionGuardWithRealNoise:
    """Unit tests for blend_with_transfer/save_with_snapshot use small, fixed
    fixtures. These exercise the same guard against more realistic conditions:
    n_real past the weight-cap saturation point, a stable re-train that must
    NOT trip the noise-aware threshold, and a genuinely corrupted re-train that
    must trip it.
    """

    def _seed(self, session, n: int, correct: bool = True) -> None:
        for i in range(n):
            iid = str(uuid.uuid4())
            _write_insight(session, iid, "safety_car", "WARNING", {
                "mean_speed_kph": 90.0 - (i % 10) * 2,
                "speed_delta_kph": -50.0 - (i % 20),
                "rain_intensity": 0.5 + (i % 10) * 0.01,
                "grip_estimate": 0.55 - (i % 10) * 0.01,
                "lockup_count": float(i % 3),
                "throttle_smoothness": 0.50,
                "race_phase": 0.45,
                "brake_temp_front_max": 420.0,
            })
            _write_feedback(session, iid, correct=correct)

    def test_real_weight_saturates_at_cap(self, sqlite_session, tmp_path):
        from f1di.agents.classifier_utils import REAL_WEIGHT_SATURATION
        from f1di.agents.safety_car_classifier import train_from_labels

        self._seed(sqlite_session, REAL_WEIGHT_SATURATION + 10)
        out = tmp_path / "sc.pkl"
        result = train_from_labels(output_path=out, synthetic_n=400)

        assert result["n_real"] == REAL_WEIGHT_SATURATION + 10
        assert result["real_sample_weight"] == 5.0  # default weight_cap

    def test_stable_real_data_does_not_spuriously_block_retrain(self, sqlite_session, tmp_path):
        from f1di.agents.safety_car_classifier import train_from_labels

        self._seed(sqlite_session, 60)
        out = tmp_path / "sc.pkl"
        first = train_from_labels(output_path=out, synthetic_n=400)
        assert first["snapshot_blocked"] is False

        second = train_from_labels(output_path=out, synthetic_n=400)
        assert second["snapshot_blocked"] is False, (
            "re-training on the same stable real data must not trip the "
            "fold-noise-aware regression guard"
        )

    def test_corrupted_real_labels_trigger_genuine_block(self, sqlite_session, tmp_path):
        from f1di.agents.safety_car_classifier import train_from_labels

        self._seed(sqlite_session, 60, correct=True)
        out = tmp_path / "sc.pkl"
        first = train_from_labels(output_path=out, synthetic_n=400)
        assert first["snapshot_blocked"] is False

        # Same features, but every label is now wrong (correct=False shifts the
        # true label away from what the features indicate) -- a real regression,
        # not fold noise, and the guard must catch it.
        self._seed(sqlite_session, 60, correct=False)
        second = train_from_labels(output_path=out, synthetic_n=400)
        assert second["accuracy"] < first["accuracy"] - 0.05
        assert second["snapshot_blocked"] is True


class TestOodScore:
    def test_in_distribution_features_low_ood(self):
        from f1di.agents.battery_classifier import BatteryClassifier, generate_synthetic
        X, y = generate_synthetic(n=400)
        clf = BatteryClassifier().fit(X, y)
        # A nominal feature vector well within training range
        class F:
            battery_soc = 0.55
            battery_soc_slope = -0.005
            mean_speed_kph = 230.0
            race_phase = 0.45
            laps_remaining = 20.0
            stint_fraction = 0.50
        ood = clf.ood_score(F())
        assert ood < 4.0, f"Nominal features should be in-distribution (got ood={ood:.1f})"

    def test_extreme_features_trigger_ood(self):
        from f1di.agents.battery_classifier import BatteryClassifier, generate_synthetic
        X, y = generate_synthetic(n=400)
        clf = BatteryClassifier().fit(X, y)
        # Extreme SOC value well outside training range
        class F:
            battery_soc = -5.0  # impossible value, very OOD
            battery_soc_slope = -0.005
            mean_speed_kph = 230.0
            race_phase = 0.45
            laps_remaining = 20.0
            stint_fraction = 0.50
        ood = clf.ood_score(F())
        assert ood > 4.0, f"Extreme features should be OOD (got ood={ood:.1f})"
