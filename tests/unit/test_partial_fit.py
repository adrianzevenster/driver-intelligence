"""Tests for partial_fit_from_labels warm-start classifiers."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch


def _make_real_data(n=20, seed=0, n_classes=3):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 5))
    y = rng.integers(0, n_classes, n, dtype=np.int32)
    return X, y


def test_tire_partial_fit_skips_empty(tmp_path):
    from f1di.agents.tire_classifier import partial_fit_from_labels, FEATURE_NAMES
    empty_X = np.empty((0, len(FEATURE_NAMES)))
    empty_y = np.empty(0, dtype=np.int32)
    with patch("f1di.agents.tire_classifier._load_labeled_from_db", return_value=(empty_X, empty_y)):
        result = partial_fit_from_labels(output_path=tmp_path / "tire_inc.pkl")
    assert result.get("skipped") is True


def test_battery_partial_fit_runs(tmp_path):
    from f1di.agents.battery_classifier import partial_fit_from_labels, FEATURE_NAMES
    n_feats = len(FEATURE_NAMES)
    rng = np.random.default_rng(0)
    X = rng.random((15, n_feats))
    y = rng.integers(0, 3, 15, dtype=np.int32)
    with patch("f1di.agents.battery_classifier._load_labeled_from_db", return_value=(X, y)):
        result = partial_fit_from_labels(output_path=tmp_path / "bat_inc.pkl")
    assert result.get("skipped") is not True
    assert 0.0 <= result["accuracy"] <= 1.0
    assert (tmp_path / "bat_inc.pkl").exists()


def test_weather_partial_fit_runs(tmp_path):
    from f1di.agents.weather_classifier import partial_fit_from_labels, FEATURE_NAMES
    n_feats = len(FEATURE_NAMES)
    rng = np.random.default_rng(1)
    X = rng.random((15, n_feats))
    y = rng.integers(0, 3, 15, dtype=np.int32)
    with patch("f1di.agents.weather_classifier._load_labeled_from_db", return_value=(X, y)):
        result = partial_fit_from_labels(output_path=tmp_path / "wx_inc.pkl")
    assert result.get("incremental") is True


def test_tire_partial_fit_incremental_updates(tmp_path):
    """Second call updates the existing model rather than recreating it."""
    from f1di.agents.tire_classifier import partial_fit_from_labels, FEATURE_NAMES
    n_feats = len(FEATURE_NAMES)
    rng = np.random.default_rng(2)
    X = rng.random((12, n_feats))
    y = rng.integers(0, 4, 12, dtype=np.int32)
    path = tmp_path / "tire2.pkl"
    with patch("f1di.agents.tire_classifier._load_labeled_from_db", return_value=(X, y)):
        partial_fit_from_labels(output_path=path)
        mtime1 = path.stat().st_mtime_ns

    import time; time.sleep(0.05)
    X2 = rng.random((5, n_feats))
    y2 = rng.integers(0, 4, 5, dtype=np.int32)
    with patch("f1di.agents.tire_classifier._load_labeled_from_db", return_value=(X2, y2)):
        result2 = partial_fit_from_labels(output_path=path)
    assert path.stat().st_mtime_ns >= mtime1
    assert result2.get("incremental") is True
