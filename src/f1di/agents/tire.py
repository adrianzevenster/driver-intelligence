from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.agents import thresholds as _thresh
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever


class TireStrategyAgent(RaceAgent):
    name = "tire_strategy"

    def analyze(self, window: TelemetryWindow, features: RaceFeatures, retriever: HybridMemoryRetriever) -> AgentFinding:
        t = _thresh.get(window.track_id)
        compound = window.latest.compound.value
        evidence = multi_source_evidence(
            retriever,
            window.track_id,
            knowledge_query=f"{window.track_id} tire degradation pit window compound {compound} cliff",
            fastf1_query=f"{window.track_id} {compound} stint average tyre life fastest lap degradation",
            jolpica_query=f"{window.track_id} pit stop race strategy result",
        )
        wear_pressure = max(features.fl_wear, features.fr_wear, features.rear_wear_mean)

        lap_span = window.latest.lap - window.samples[0].lap
        spl = len(window.samples) / lap_span if lap_span > 0 else 1.0
        projected_fl_4laps = features.fl_wear + features.fl_wear_slope * spl * 4
        projected_fr_4laps = features.fr_wear + features.fr_wear_slope * spl * 4
        projected_front_cliff = max(projected_fl_4laps, projected_fr_4laps)

        if wear_pressure > t.wear_critical and features.grip_estimate < 0.62:
            conf = 0.81
            conf += 0.04 if features.fl_wear_slope > 0.003 else 0.0
            conf += 0.04 if features.fr_wear > 0.75 else 0.0
            conf += 0.02 if features.axle_imbalance_fl_rl > 0.12 else 0.0
            axle_flag = features.axle_imbalance_fl_rl > 0.15
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.CRITICAL,
                summary=(
                    "Box window should be opened: tire wear and grip loss indicate imminent performance cliff with axle imbalance."
                    if axle_flag else
                    "Box window should be opened: tire wear and grip loss indicate imminent performance cliff."
                ),
                confidence=min(0.92, conf),
                evidence=evidence,
                features={"wear_pressure": wear_pressure, "grip": features.grip_estimate, "axle_imbalance": features.axle_imbalance_fl_rl},
            )

        if wear_pressure > t.wear_warning:
            conf = 0.74
            conf += 0.04 if features.fl_wear_slope > 0.002 else 0.0
            conf += 0.03 if features.fr_wear_slope > 0.002 else 0.0
            axle_flag = features.axle_imbalance_fl_rl > 0.15
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WARNING,
                summary=(
                    "Prepare pit window within two laps; axle imbalance indicates corner-entry instability risk."
                    if axle_flag else
                    "Prepare pit window within two laps unless track position risk dominates."
                ),
                confidence=min(0.84, conf),
                evidence=evidence,
                features={"wear_pressure": wear_pressure, "grip": features.grip_estimate, "axle_imbalance": features.axle_imbalance_fl_rl},
            )

        if projected_front_cliff > t.wear_critical * 0.97 and features.fl_wear_slope > 0.0:
            conf = 0.68
            conf += 0.04 if features.fr_wear_slope > 0.0015 else 0.0
            conf += 0.03 if features.rear_wear_slope > 0.001 else 0.0
            front_driving = "FL" if projected_fl_4laps >= projected_fr_4laps else "FR"
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WARNING,
                summary=f"Degradation trajectory projects {front_driving} past cliff threshold within four laps; open pit discussion now.",
                confidence=min(0.78, conf),
                evidence=evidence,
                features={
                    "fl_wear": features.fl_wear,
                    "fl_wear_slope": features.fl_wear_slope,
                    "fr_wear_slope": features.fr_wear_slope,
                    "projected_fl_4laps": projected_fl_4laps,
                    "projected_fr_4laps": projected_fr_4laps,
                },
            )

        if features.fr_wear_slope > features.fl_wear_slope + 0.0015 and features.fr_wear > 0.42:
            conf = 0.62
            conf += 0.04 if features.fr_wear > 0.55 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WATCH,
                summary="FR degrading faster than FL; atypical axle loading — monitor for setup-driven imbalance or circuit-specific stress.",
                confidence=min(0.68, conf),
                evidence=evidence,
                features={"fr_wear_slope": features.fr_wear_slope, "fl_wear_slope": features.fl_wear_slope, "fr_wear": features.fr_wear},
            )

        if features.axle_imbalance_fl_rl > 0.12 and features.rear_wear_mean > 0.25:
            conf = 0.63
            conf += 0.04 if features.rear_wear_slope > 0.0015 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WATCH,
                summary="Front-rear axle imbalance is building; monitor for corner-entry instability as stint extends.",
                confidence=min(0.68, conf),
                evidence=evidence,
                features={"axle_imbalance": features.axle_imbalance_fl_rl, "rear_wear_mean": features.rear_wear_mean},
            )

        if features.rear_wear_slope > 0.0022 and features.rear_wear_mean > 0.35:
            conf = 0.61
            conf += 0.03 if features.fl_wear_slope > 0.0015 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WATCH,
                summary="Rear tyre degradation rate is above expected envelope; assess traction and stability into braking zones.",
                confidence=min(0.66, conf),
                evidence=evidence,
                features={"rear_wear_slope": features.rear_wear_slope, "rear_wear_mean": features.rear_wear_mean},
            )

        conf = max(0.50, 0.62 - wear_pressure * 0.12)
        return AgentFinding(
            agent=self.name,
            risk=RiskLevel.INFO,
            summary="Tire state remains within modelled stint limits.",
            confidence=conf,
            evidence=evidence,
        )
