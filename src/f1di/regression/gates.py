from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from f1di.domain.schemas import DriverInsight


@dataclass(frozen=True)
class RegressionGate:
    min_schema_adherence: float = 1.0
    min_grounding_score: float = 0.50
    max_latency_ms_p95: float = 250.0
    min_confidence_for_warning: float = 0.65
    min_evidence_score_mean: float = 0.25
    min_risk_variety: int = 2


def grounding_score(insights: list[DriverInsight]) -> float:
    if not insights:
        return 0.0
    return fmean(1.0 if i.evidence else 0.0 for i in insights)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * 0.95))]


def evaluate_gates(insights: list[DriverInsight], gate: RegressionGate = RegressionGate()) -> dict[str, bool | float]:
    warning_conf = [i.confidence for i in insights if i.risk.value in {"WARNING", "CRITICAL"}]
    evidence_scores = [e.score for i in insights for e in i.evidence]
    evidence_score_mean = fmean(evidence_scores) if evidence_scores else 0.0
    risk_variety = len({i.risk for i in insights})
    return {
        "grounding_score": grounding_score(insights),
        "latency_p95_ms": p95([i.latency_ms for i in insights]),
        "warning_confidence_mean": fmean(warning_conf) if warning_conf else 1.0,
        "evidence_score_mean": evidence_score_mean,
        "risk_variety": risk_variety,
        "pass_grounding": grounding_score(insights) >= gate.min_grounding_score,
        "pass_latency": p95([i.latency_ms for i in insights]) <= gate.max_latency_ms_p95,
        "pass_warning_confidence": (fmean(warning_conf) if warning_conf else 1.0) >= gate.min_confidence_for_warning,
        "pass_evidence_quality": evidence_score_mean >= gate.min_evidence_score_mean,
        "pass_risk_variety": risk_variety >= gate.min_risk_variety,
    }
