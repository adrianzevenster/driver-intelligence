"""Tests for the SHAP-based explainer."""
from __future__ import annotations

from unittest.mock import MagicMock


def _mock_finding(agent: str, risk: str, conf: float):
    from f1di.domain.schemas import RiskLevel
    f = MagicMock()
    f.agent = agent
    f.risk = RiskLevel[risk]
    f.confidence = conf
    return f


def test_explain_findings_no_shap(monkeypatch):
    """When shap is not installed, returns empty list without crashing."""
    import sys
    monkeypatch.setitem(sys.modules, "shap", None)

    from f1di.inference import explainer
    # Reload to pick up the None'd module
    import importlib
    importlib.reload(explainer)

    findings = [_mock_finding("tire_strategy", "WARNING", 0.8)]
    result = explainer.explain_findings(findings, 0.75)
    assert result == []


def test_explain_findings_no_model(tmp_path, monkeypatch):
    """Returns empty list when meta-learner pkl does not exist."""
    import f1di.inference.explainer as exp_mod
    monkeypatch.setattr(exp_mod, "_META_PATH", tmp_path / "nonexistent.pkl")

    findings = [_mock_finding("tire_strategy", "INFO", 0.6)]
    result = exp_mod.explain_findings(findings, 0.6)
    assert result == []


def test_explain_findings_below_threshold(tmp_path, monkeypatch):
    """Returns empty list when meta-learner has < 20 real labels."""

    from f1di.inference.meta_learner import MetaLearner, generate_synthetic

    X, y = generate_synthetic(n=200, seed=42)
    meta = MetaLearner().fit(X, y, n_real=5)  # below 20
    pkl_path = tmp_path / "meta.pkl"
    meta.save(pkl_path)

    import f1di.inference.explainer as exp_mod
    monkeypatch.setattr(exp_mod, "_META_PATH", pkl_path)

    findings = [_mock_finding("tire_strategy", "WARNING", 0.8)]
    result = exp_mod.explain_findings(findings, 0.75)
    assert result == []
