from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever


class TelemetryAnalysisAgent(RaceAgent):
    name = "telemetry"

    def analyze(self, window: TelemetryWindow, features: RaceFeatures, retriever: HybridMemoryRetriever) -> AgentFinding:
        evidence = multi_source_evidence(
            retriever,
            window.track_id,
            knowledge_query=f"{window.track_id} sector {features.sector} braking lockup instability tire wear",
            fastf1_query=f"{window.track_id} sector fastest lap braking brake temperature",
            jolpica_query=f"{window.track_id} race result fastest lap sector",
        )

        if features.brake_temp_front_max > 950 or features.lockup_count >= 2:
            conf = 0.82
            conf += 0.05 if features.lockup_count >= 2 and features.brake_fade_risk > 12 else 0.0
            conf += 0.03 if features.fl_degradation_pressure > 0.70 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.CRITICAL,
                summary="Front braking envelope is unstable with repeated lockup or excessive temperature.",
                confidence=min(0.92, conf),
                evidence=evidence,
                features={
                    "brake_temp_front_max": features.brake_temp_front_max,
                    "lockup_count": features.lockup_count,
                    "brake_fade_risk": features.brake_fade_risk,
                    "fl_degradation_pressure": features.fl_degradation_pressure,
                },
            )

        if features.fl_degradation_pressure > 0.72 or features.fl_wear_slope > 0.008:
            conf = 0.77
            conf += 0.04 if features.fr_wear_slope > 0.003 else 0.0
            conf += 0.03 if features.fl_degradation_pressure > 0.60 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WARNING,
                summary="Front-left degradation pressure is accelerating beyond the current stint projection.",
                confidence=min(0.86, conf),
                evidence=evidence,
                features={
                    "fl_degradation_pressure": features.fl_degradation_pressure,
                    "fl_wear_slope": features.fl_wear_slope,
                    "fr_wear_slope": features.fr_wear_slope,
                },
            )

        if features.brake_fade_risk > 12.0 or features.crosswind_proxy > 10:
            conf = 0.64
            conf += 0.04 if features.brake_fade_risk > 12 and features.crosswind_proxy > 8 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WATCH,
                summary=(
                    "Brake temperatures are trending upward under sustained load."
                    if features.brake_fade_risk > 12.0
                    else "Crosswind sensitivity is increasing under steering load."
                ),
                confidence=min(0.70, conf),
                evidence=evidence,
                features={"brake_fade_risk": features.brake_fade_risk, "crosswind_proxy": features.crosswind_proxy},
            )

        conf = max(0.48, 0.60 - features.fl_degradation_pressure * 0.10)
        return AgentFinding(
            agent=self.name,
            risk=RiskLevel.INFO,
            summary="Telemetry envelope nominal.",
            confidence=conf,
            evidence=evidence,
            features=features.__dict__,
        )
