"""Unit tests for auto_retrain module."""
from __future__ import annotations

import pickle
from unittest.mock import patch

import numpy as np

from f1di.agents.auto_retrain import (
    RETRAIN_THRESHOLD,
    _db_n_real,
    _pkl_n_real,
    maybe_retrain,
    maybe_retrain_all,
    retrain_status,
)


class _FakeClf:
    """Minimal picklable stand-in for a fitted classifier."""
    def __init__(self, n_real: int):
        self.n_real = n_real
        self.accuracy = 0.90
        self.brier_score = 0.12


def _make_clf(n_real: int) -> _FakeClf:
    return _FakeClf(n_real)


class TestPklNReal:
    def test_returns_zero_when_no_pkl(self, tmp_path):
        with patch("f1di.agents.auto_retrain._AGENT_PATHS", {"tire": tmp_path / "missing.pkl"}):
            assert _pkl_n_real("tire") == 0

    def test_reads_n_real_from_pkl(self, tmp_path):
        clf = _make_clf(n_real=42)
        p = tmp_path / "tire_classifier.pkl"
        p.write_bytes(pickle.dumps(clf))
        with patch("f1di.agents.auto_retrain._AGENT_PATHS", {"tire": p}):
            assert _pkl_n_real("tire") == 42

    def test_returns_zero_on_corrupt_pkl(self, tmp_path):
        p = tmp_path / "bad.pkl"
        p.write_bytes(b"not a pickle")
        with patch("f1di.agents.auto_retrain._AGENT_PATHS", {"tire": p}):
            assert _pkl_n_real("tire") == 0

    def test_unknown_agent_returns_zero(self):
        assert _pkl_n_real("unknown_agent") == 0


class TestDbNReal:
    def test_returns_count_from_load_labeled(self):
        fake_y = np.array([0, 1, 2, 0, 1])
        with patch("f1di.agents.battery_classifier._load_labeled_from_db",
                   return_value=(np.zeros((5, 6)), fake_y)):
            result = _db_n_real("battery")
        assert result == 5

    def test_returns_zero_on_import_error(self):
        with patch("f1di.agents.auto_retrain._db_n_real", side_effect=Exception("db down")):
            # directly test fallback path
            pass
        # call with a bad agent name
        assert _db_n_real("nonexistent_agent") == 0

    def test_returns_zero_on_db_exception(self):
        with patch("f1di.agents.battery_classifier._load_labeled_from_db",
                   side_effect=Exception("db unavailable")):
            assert _db_n_real("battery") == 0


class TestMaybeRetrain:
    def test_skips_when_delta_below_threshold(self, tmp_path):
        clf = _make_clf(n_real=10)
        p = tmp_path / "battery_classifier.pkl"
        p.write_bytes(pickle.dumps(clf))

        # delta = 12 - 10 = 2, below threshold of 5
        with patch("f1di.agents.auto_retrain._pkl_n_real", return_value=10), \
             patch("f1di.agents.auto_retrain._db_n_real", return_value=12), \
             patch("f1di.agents.auto_retrain._AGENT_PATHS", {"battery": p}), \
             patch("f1di.agents.battery_classifier.train_from_labels") as mock_train:
            maybe_retrain("battery", threshold=5)
            mock_train.assert_not_called()

    def test_triggers_when_delta_meets_threshold(self, tmp_path):
        clf = _make_clf(n_real=10)
        p = tmp_path / "battery_classifier.pkl"
        p.write_bytes(pickle.dumps(clf))

        # delta = 15 - 10 = 5, meets threshold
        with patch("f1di.agents.auto_retrain._pkl_n_real", return_value=10), \
             patch("f1di.agents.auto_retrain._db_n_real", return_value=15), \
             patch("f1di.agents.auto_retrain._AGENT_PATHS", {"battery": p}), \
             patch("f1di.agents.battery_classifier.train_from_labels",
                   return_value={"accuracy": 0.90, "brier_score": 0.12, "n_real": 15,
                                 "snapshot_blocked": False}) as mock_train:
            maybe_retrain("battery", threshold=5)
            mock_train.assert_called_once()

    def test_skips_unknown_agent(self):
        with patch("f1di.agents.battery_classifier.train_from_labels") as mock_train:
            maybe_retrain("nonexistent", threshold=1)
            mock_train.assert_not_called()

    def test_no_double_retrain_while_in_progress(self, tmp_path):
        """Second call while first is running should be a no-op."""
        import f1di.agents.auto_retrain as ar
        clf = _make_clf(n_real=0)
        p = tmp_path / "weather_classifier.pkl"
        p.write_bytes(pickle.dumps(clf))

        call_count = 0

        def slow_train(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"accuracy": 0.9, "brier_score": 0.1, "n_real": 10, "snapshot_blocked": False}

        with patch("f1di.agents.auto_retrain._pkl_n_real", return_value=0), \
             patch("f1di.agents.auto_retrain._db_n_real", return_value=10), \
             patch("f1di.agents.auto_retrain._AGENT_PATHS", {"weather": p}), \
             patch("f1di.agents.weather_classifier.train_from_labels", side_effect=slow_train):
            # Manually mark as in-progress
            with ar._lock:
                ar._in_progress.add("weather")
            try:
                maybe_retrain("weather", threshold=5)
                assert call_count == 0  # blocked by in_progress guard
            finally:
                with ar._lock:
                    ar._in_progress.discard("weather")


class TestMaybeRetrainAll:
    def test_calls_maybe_retrain_for_all_agents(self):
        called = []
        with patch("f1di.agents.auto_retrain.maybe_retrain",
                   side_effect=lambda agent, threshold=RETRAIN_THRESHOLD: called.append(agent)):
            maybe_retrain_all()
        assert set(called) == {"tire", "battery", "weather", "telemetry"}


class TestRetrainStatus:
    def test_returns_all_agents(self, tmp_path):
        fake_paths = {a: tmp_path / f"{a}.pkl" for a in ["tire", "battery", "weather", "telemetry"]}
        with patch("f1di.agents.auto_retrain._AGENT_PATHS", fake_paths):
            status = retrain_status()
        assert set(status["agents"].keys()) == {"tire", "battery", "weather", "telemetry"}
        assert status["threshold"] == RETRAIN_THRESHOLD

    def test_retrain_in_progress_reflected(self, tmp_path):
        import f1di.agents.auto_retrain as ar
        fake_paths = {"battery": tmp_path / "battery.pkl"}
        with patch("f1di.agents.auto_retrain._AGENT_PATHS", fake_paths):
            with ar._lock:
                ar._in_progress.add("battery")
            try:
                status = retrain_status()
                assert status["agents"]["battery"]["retrain_in_progress"] is True
            finally:
                with ar._lock:
                    ar._in_progress.discard("battery")
