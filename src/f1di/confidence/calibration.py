from __future__ import annotations

import math
import pickle
from pathlib import Path

from sklearn.isotonic import IsotonicRegression

from f1di.domain.schemas import AgentFinding, RiskLevel

RISK_WEIGHT = {RiskLevel.INFO: 0.25, RiskLevel.WATCH: 0.45, RiskLevel.WARNING: 0.70, RiskLevel.CRITICAL: 0.90}

# WARNING threshold — used to decide which side of the agent-signal split applies.
_WARNING_THRESHOLD = RISK_WEIGHT[RiskLevel.WARNING]

_CALIBRATOR_PATH = Path("data/calibration/isotonic.pkl")


def _shared_intermediates(findings: list[AgentFinding]) -> tuple[float, float, float, float, float]:
    """Compute the five shared intermediate values used by both score functions."""
    risk_values = [RISK_WEIGHT[f.risk] for f in findings]
    mean_risk = sum(risk_values) / len(risk_values)
    max_risk = max(risk_values)
    dispersion = math.sqrt(sum((x - mean_risk) ** 2 for x in risk_values) / len(risk_values))
    agent_agreement = max(0.0, 1.0 - dispersion)
    model_confidence = sum(f.confidence for f in findings) / len(findings)
    evidence_scores = [e.score for f in findings for e in f.evidence]
    evidence_strength = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.35
    return mean_risk, max_risk, agent_agreement, model_confidence, evidence_strength


def compute_raw_score(findings: list[AgentFinding]) -> tuple[float, dict[str, float]]:
    """Production scoring formula (heuristic weights)."""
    if not findings:
        return 0.0, {"agent_agreement": 0.0, "evidence_strength": 0.0, "risk_mean": 0.0}
    mean_risk, max_risk, agent_agreement, model_confidence, evidence_strength = _shared_intermediates(findings)
    raw = max(
        0.0,
        min(
            1.0,
            0.30 * max_risk
            + 0.25 * model_confidence
            + 0.20 * mean_risk
            + 0.15 * evidence_strength
            + 0.10 * agent_agreement,
        ),
    )
    return raw, {
        "agent_agreement": agent_agreement,
        "model_confidence": model_confidence,
        "evidence_strength": evidence_strength,
        "risk_mean": mean_risk,
        "risk_max": max_risk,
    }


def compute_raw_score_v2(findings: list[AgentFinding]) -> tuple[float, dict[str, float]]:
    """Challenger scoring formula — fitted weights + context-aware agent signal.

    Key differences from v1:
    - risk_mean weighted up (0.20 → 0.38): sustained mean risk is the strongest predictor.
    - evidence_strength uses min() instead of mean(): with the global-max retriever fix,
      min score across evidence items captures "did all agents find strong matches?" better
      than mean (which is dominated by the always-high top result).
    - evidence_strength weighted down (0.15 → 0.05): retriever scores were near-constant
      under per-query normalisation; this weight reflects residual uncertainty.
    - agent_signal replaces agent_agreement: when the top agent flags WARNING+, divergence
      among agents means one agent correctly identified a real risk (so 1-agreement is used);
      when all agents see low risk, unanimous calm is the trustworthy signal (agreement used).
    """
    if not findings:
        return 0.0, {"agent_agreement": 0.0, "evidence_strength": 0.0, "risk_mean": 0.0}
    mean_risk, max_risk, agent_agreement, model_confidence, _ = _shared_intermediates(findings)
    # Use minimum evidence score: low min means at least one agent's retrieval found nothing
    # relevant, which is a meaningful signal once the retriever gives absolute scores.
    evidence_scores = [e.score for f in findings for e in f.evidence]
    evidence_strength = min(evidence_scores) if evidence_scores else 0.0

    # Divergence is informative when a high-risk outlier agent fires; consensus is
    # informative when all agents agree things are calm.
    if max_risk >= _WARNING_THRESHOLD:
        agent_signal = 1.0 - agent_agreement  # one agent seeing danger while others don't = real signal
    else:
        agent_signal = agent_agreement         # unanimous calm = genuinely safe

    raw = max(
        0.0,
        min(
            1.0,
            0.28 * max_risk
            + 0.38 * mean_risk
            + 0.17 * model_confidence
            + 0.05 * evidence_strength
            + 0.12 * agent_signal,
        ),
    )
    return raw, {
        "agent_agreement": agent_agreement,
        "agent_signal": agent_signal,
        "model_confidence": model_confidence,
        "evidence_strength": evidence_strength,
        "risk_mean": mean_risk,
        "risk_max": max_risk,
    }


class ConfidenceCalibrator:
    def __init__(self, model: IsotonicRegression | None = None) -> None:
        self._model = model

    def calibrate(self, findings: list[AgentFinding]) -> tuple[float, float, dict[str, float], float]:
        raw, features = compute_raw_score(findings)
        if self._model is not None:
            confidence = float(self._model.predict([raw])[0])
        else:
            confidence = raw
        confidence = max(0.0, min(1.0, confidence))
        return confidence, 1.0 - confidence, features, raw

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


class ChallengerCalibrator(ConfidenceCalibrator):
    """Calibrator using the v2 scoring formula. Loads the same isotonic pkl so the
    calibration mapping is held constant and only the raw-score computation differs.
    This isolates the weight change as the variable under test in shadow mode.
    """

    def __init__(self) -> None:
        model: IsotonicRegression | None = None
        if _CALIBRATOR_PATH.exists():
            try:
                model = pickle.loads(_CALIBRATOR_PATH.read_bytes())
            except Exception:
                pass
        super().__init__(model=model)

    def calibrate(self, findings: list[AgentFinding]) -> tuple[float, float, dict[str, float], float]:
        raw, features = compute_raw_score_v2(findings)
        if self._model is not None:
            confidence = float(self._model.predict([raw])[0])
        else:
            confidence = raw
        confidence = max(0.0, min(1.0, confidence))
        return confidence, 1.0 - confidence, features, raw
