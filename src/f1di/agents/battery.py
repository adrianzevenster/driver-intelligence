from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.agents import thresholds as _thresh
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever


class BatteryAgent(RaceAgent):
    name = "battery"

    def analyze(self, window: TelemetryWindow, features: RaceFeatures, retriever: HybridMemoryRetriever) -> AgentFinding:
        t = _thresh.get(window.track_id)
        evidence = multi_source_evidence(
            retriever,
            window.track_id,
            knowledge_query=f"{window.track_id} ERS deployment battery SOC exit sector {features.sector}",
            fastf1_query=f"{window.track_id} ERS battery deployment straight DRS speed",
            jolpica_query=f"{window.track_id} race fastest lap power unit",
        )

        if features.battery_soc < t.battery_soc_warning and features.battery_soc_slope < -0.01:
            conf = 0.79
            conf += 0.04 if features.battery_soc_slope < -0.015 else 0.0
            conf += 0.02 if features.mean_speed_kph > 220 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WARNING,
                summary="ERS state of charge is depleting too quickly; reduce deployment before the next high-value straight.",
                confidence=min(0.87, conf),
                evidence=evidence,
                features={"battery_soc": features.battery_soc, "battery_soc_slope": features.battery_soc_slope},
            )

        if features.battery_soc > 0.72 and features.mean_speed_kph < 220:
            conf = 0.69
            conf += 0.03 if features.mean_speed_kph < 200 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WATCH,
                summary="Battery is under-deployed relative to sector speed profile; increase deployment on exit zones.",
                confidence=min(0.74, conf),
                evidence=evidence,
                features={"battery_soc": features.battery_soc, "mean_speed_kph": features.mean_speed_kph},
            )

        conf = max(0.55, 0.65 - abs(features.battery_soc - 0.55) * 0.12)
        return AgentFinding(
            agent=self.name,
            risk=RiskLevel.INFO,
            summary="ERS deployment is consistent with the current tactical envelope.",
            confidence=conf,
            evidence=evidence,
        )
