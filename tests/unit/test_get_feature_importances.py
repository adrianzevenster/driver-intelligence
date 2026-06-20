"""Tests for MetaLearner.get_feature_importances()."""
from __future__ import annotations

from f1di.inference.meta_learner import FEATURE_NAMES, MetaLearner, generate_synthetic


def test_get_feature_importances_returns_dict():
    X, y = generate_synthetic(n=300, seed=0)
    meta = MetaLearner().fit(X, y, n_real=25)
    result = meta.get_feature_importances()
    assert isinstance(result, dict)


def test_get_feature_importances_all_features_present():
    X, y = generate_synthetic(n=300, seed=1)
    meta = MetaLearner().fit(X, y, n_real=25)
    result = meta.get_feature_importances()
    assert set(result.keys()) == set(FEATURE_NAMES)


def test_get_feature_importances_normalized():
    X, y = generate_synthetic(n=400, seed=2)
    meta = MetaLearner().fit(X, y, n_real=30)
    result = meta.get_feature_importances()
    total = sum(result.values())
    assert abs(total - 1.0) < 0.01, f"Importances should sum to 1.0, got {total}"


def test_get_feature_importances_nonnegative():
    X, y = generate_synthetic(n=400, seed=3)
    meta = MetaLearner().fit(X, y, n_real=30)
    result = meta.get_feature_importances()
    assert all(v >= 0 for v in result.values()), "All importances should be non-negative"


def test_get_feature_importances_unfitted_returns_empty():
    """An unfitted meta-learner model returns empty dict without crashing."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    meta = MetaLearner.__new__(MetaLearner)
    meta._model = HistGradientBoostingClassifier()
    meta._scaler = StandardScaler()
    meta.n_real = 0
    meta.n_train = 0
    meta.accuracy = 0.0
    # Model is not fitted — feature_importances_ attribute is absent
    result = meta.get_feature_importances()
    assert isinstance(result, dict)
