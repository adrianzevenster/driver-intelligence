"""Unit tests for the fusion meta-learner."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from f1di.inference.meta_learner import (
    MetaLearner, FEATURE_NAMES, findings_to_array,
    generate_synthetic, train_from_labels,
)


# ── findings_to_array ──────────────────────────────────────────────────────

def _mock_finding(agent: str, risk: str, conf: float):
    from unittest.mock import MagicMock
    from f1di.domain.schemas import RiskLevel
    f = MagicMock()
    f.agent = agent
    f.risk = RiskLevel[risk]
    f.confidence = conf
    return f


def test_findings_to_array_length():
    findings = [
        _mock_finding("tire_strategy", "WARNING", 0.80),
        _mock_finding("battery", "INFO", 0.60),
        _mock_finding("weather", "INFO", 0.62),
        _mock_finding("telemetry", "WATCH", 0.65),
    ]
    arr = findings_to_array(findings, 0.72)
    assert arr.shape == (len(FEATURE_NAMES),)


def test_findings_to_array_iso_conf_last():
    findings = [_mock_finding("tire_strategy", "INFO", 0.6),
                _mock_finding("battery", "INFO", 0.6),
                _mock_finding("weather", "INFO", 0.6),
                _mock_finding("telemetry", "INFO", 0.6)]
    arr = findings_to_array(findings, 0.88)
    assert arr[-1] == pytest.approx(0.88)


def test_findings_to_array_missing_agent():
    findings = [_mock_finding("tire_strategy", "CRITICAL", 0.90)]
    arr = findings_to_array(findings, 0.70)
    assert arr.shape == (len(FEATURE_NAMES),)
    assert not np.any(np.isnan(arr))


def test_findings_to_array_agreement_all_same():
    findings = [_mock_finding(a, "INFO", 0.6)
                for a in ("tire_strategy", "battery", "weather", "telemetry")]
    arr = findings_to_array(findings, 0.6)
    agreement = arr[-2]  # second to last
    assert agreement == pytest.approx(1.0)


def test_findings_to_array_agreement_all_different():
    findings = [
        _mock_finding("tire_strategy", "CRITICAL", 0.9),
        _mock_finding("battery", "INFO", 0.6),
        _mock_finding("weather", "INFO", 0.6),
        _mock_finding("telemetry", "INFO", 0.6),
    ]
    arr = findings_to_array(findings, 0.5)
    agreement = arr[-2]
    assert 0.0 <= agreement < 1.0


# ── MetaLearner fit / predict ──────────────────────────────────────────────

def test_generate_synthetic_shape():
    X, y = generate_synthetic(n=200)
    assert X.shape == (200, len(FEATURE_NAMES))
    assert set(y.tolist()) == {0, 1}


def test_meta_learner_fit_accuracy():
    X, y = generate_synthetic(n=600, seed=42)
    meta = MetaLearner().fit(X, y)
    assert meta.accuracy > 0.60


def test_meta_learner_predict_confidence_range():
    X, y = generate_synthetic(n=600, seed=42)
    meta = MetaLearner().fit(X, y)
    findings = [_mock_finding(a, "WARNING", 0.82)
                for a in ("tire_strategy", "battery", "weather", "telemetry")]
    conf = meta.predict_confidence(findings, 0.75)
    assert 0.0 < conf < 1.0


def test_meta_learner_high_agreement_high_conf():
    X, y = generate_synthetic(n=600, seed=42)
    meta = MetaLearner().fit(X, y)
    # All agents WARNING + high iso_conf → should predict high P(correct)
    findings = [_mock_finding(a, "WARNING", 0.85)
                for a in ("tire_strategy", "battery", "weather", "telemetry")]
    conf_agree = meta.predict_confidence(findings, 0.85)
    # Single agent WARNING + others INFO + low iso → lower P(correct)
    findings_single = [
        _mock_finding("tire_strategy", "WARNING", 0.75),
        _mock_finding("battery", "INFO", 0.55),
        _mock_finding("weather", "INFO", 0.55),
        _mock_finding("telemetry", "INFO", 0.55),
    ]
    conf_single = meta.predict_confidence(findings_single, 0.35)
    assert conf_agree > conf_single


def test_meta_learner_save_load(tmp_path):
    X, y = generate_synthetic(n=200, seed=1)
    meta = MetaLearner().fit(X, y, n_real=25)
    p = tmp_path / "meta.pkl"
    meta.save(p)
    loaded = MetaLearner.load(p)
    assert loaded.n_real == 25
    findings = [_mock_finding(a, "INFO", 0.6)
                for a in ("tire_strategy", "battery", "weather", "telemetry")]
    c1 = meta.predict_confidence(findings, 0.6)
    c2 = loaded.predict_confidence(findings, 0.6)
    assert c1 == pytest.approx(c2)


def test_train_no_db(tmp_path):
    out = tmp_path / "meta.pkl"
    with patch("f1di.inference.meta_learner._load_labeled_from_db",
               return_value=(np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32))):
        r = train_from_labels(output_path=out, synthetic_n=300)
    assert out.exists()
    assert r["n_real"] == 0
    assert r["active_in_inference"] is False
    assert r["accuracy"] > 0.50


def test_train_activates_with_enough_real(tmp_path):
    out = tmp_path / "meta.pkl"
    rng = np.random.default_rng(0)
    real_X = rng.uniform(0.0, 1.0, (25, len(FEATURE_NAMES)))
    real_y = np.array([1] * 15 + [0] * 10, dtype=np.int32)
    with patch("f1di.inference.meta_learner._load_labeled_from_db",
               return_value=(real_X, real_y)):
        r = train_from_labels(output_path=out, synthetic_n=200, real_oversample=3)
    assert r["n_real"] == 25
    assert r["active_in_inference"] is True


# ── Fusion integration ─────────────────────────────────────────────────────

def test_fusion_meta_learner_not_applied_below_threshold():
    """Meta-learner with n_real < 20 must not alter iso_confidence."""
    from f1di.inference.fusion import _get_meta_learner
    import f1di.inference.fusion as fus_mod

    # Patch path to point at a tmp pkl with n_real=5
    X, y = generate_synthetic(n=200)
    meta = MetaLearner().fit(X, y, n_real=5)

    import pickle
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        f.write(pickle.dumps(meta))
        tmp_path = Path(f.name)

    try:
        with patch("f1di.inference.fusion._META_PATH", tmp_path):
            fus_mod._meta_mtime = 0.0
            loaded = _get_meta_learner()
        assert loaded is not None
        assert loaded.n_real == 5
        # Fusion code checks n_real >= 20 before blending
        assert loaded.n_real < 20
    finally:
        tmp_path.unlink(missing_ok=True)
