from __future__ import annotations

import math
import pickle
from pathlib import Path

from sklearn.isotonic import IsotonicRegression

from f1di.domain.schemas import AgentFinding, RiskLevel

RISK_WEIGHT = {RiskLevel.INFO: 0.25, RiskLevel.WATCH: 0.45, RiskLevel.WARNING: 0.70, RiskLevel.CRITICAL: 0.90}


def compute_raw_score(findings: list[AgentFinding]) -> tuple[float, dict[str, float]]:
    if not findings:
        return 0.0, {"agent_agreement": 0.0, "evidence_strength": 0.0, "risk_mean": 0.0}
    risk_values = [RISK_WEIGHT[f.risk] for f in findings]
    mean_risk = sum(risk_values) / len(risk_values)
    dispersion = math.sqrt(sum((x - mean_risk) ** 2 for x in risk_values) / len(risk_values))
    agent_agreement = max(0.0, 1.0 - dispersion)
    model_confidence = sum(f.confidence for f in findings) / len(findings)
    evidence_scores = [e.score for f in findings for e in f.evidence]
    evidence_strength = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.35
    raw = max(0.0, min(1.0, 0.50 * model_confidence + 0.30 * mean_risk + 0.20 * evidence_strength))
    return raw, {
        "agent_agreement": agent_agreement,
        "model_confidence": model_confidence,
        "evidence_strength": evidence_strength,
        "risk_mean": mean_risk,
    }


class ConfidenceCalibrator:
    def __init__(self, model: IsotonicRegression | None = None) -> None:
        self._model = model

    def calibrate(self, findings: list[AgentFinding]) -> tuple[float, float, dict[str, float]]:
        raw, features = compute_raw_score(findings)
        if self._model is not None:
            confidence = float(self._model.predict([features["risk_mean"]])[0])
        else:
            confidence = raw
        confidence = max(0.0, min(1.0, confidence))
        return confidence, 1.0 - confidence, features

    @classmethod
    def fit(cls, X: list[float], y: list[float]) -> ConfidenceCalibrator:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        model.fit(X, y)
        return cls(model=model)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self._model))

    @classmethod
    def load(cls, path: Path) -> ConfidenceCalibrator:
        model = pickle.loads(path.read_bytes())
        return cls(model=model)
