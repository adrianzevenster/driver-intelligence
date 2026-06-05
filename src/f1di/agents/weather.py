from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever


class WeatherAgent(RaceAgent):
    name = "weather"

    def analyze(self, window: TelemetryWindow, features: RaceFeatures, retriever: HybridMemoryRetriever) -> AgentFinding:
        evidence = multi_source_evidence(
            retriever,
            window.track_id,
            knowledge_query=f"{window.track_id} rain crossover wind grip track temperature weather",
            fastf1_query=f"{window.track_id} weather rain track temperature air humidity",
            jolpica_query=f"{window.track_id} race wet rain safety car weather",
        )

        if features.rain_intensity >= 0.35:
            conf = 0.76
            conf += 0.04 if features.grip_estimate < 0.65 else 0.0
            conf += 0.02 if features.crosswind_proxy > 8 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WARNING,
                summary="Rain intensity is approaching crossover territory; monitor inter timing.",
                confidence=min(0.84, conf),
                evidence=evidence,
                features={"rain_intensity": features.rain_intensity, "grip_estimate": features.grip_estimate},
            )

        if features.crosswind_proxy > 12:
            conf = 0.67
            conf += 0.04 if features.brake_fade_risk > 8 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WATCH,
                summary="Crosswind is likely affecting braking stability and turn-in confidence.",
                confidence=min(0.73, conf),
                evidence=evidence,
                features={"crosswind_proxy": features.crosswind_proxy, "brake_fade_risk": features.brake_fade_risk},
            )

        conf = max(0.55, 0.65 - features.rain_intensity * 0.20)
        return AgentFinding(
            agent=self.name,
            risk=RiskLevel.INFO,
            summary="Weather signal does not require strategy change.",
            confidence=conf,
            evidence=evidence,
        )
