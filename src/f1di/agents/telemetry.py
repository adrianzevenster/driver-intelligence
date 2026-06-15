from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("f1di.agents.telemetry")

from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.agents import thresholds as _thresh
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever

_CLASSIFIER_PATH = Path("data/calibration/telemetry_classifier.pkl")
_clf_cache: object = None
_clf_mtime: float = 0.0


def _get_classifier():
    global _clf_cache, _clf_mtime
    if not _CLASSIFIER_PATH.exists():
        return None
    mtime = _CLASSIFIER_PATH.stat().st_mtime
    if mtime != _clf_mtime:
        from f1di.agents.telemetry_classifier import TelemetryClassifier
        _clf_cache = TelemetryClassifier.load(_CLASSIFIER_PATH)
        _clf_mtime = mtime
    return _clf_cache


def _base_features(features: RaceFeatures) -> dict[str, float]:
    return {
        "brake_temp_front_max": features.brake_temp_front_max,
        "lockup_count": float(features.lockup_count),
        "brake_fade_risk": features.brake_fade_risk,
        "fl_degradation_pressure": features.fl_degradation_pressure,
        "fl_wear_slope": features.fl_wear_slope,
        "fr_wear_slope": features.fr_wear_slope,
        "crosswind_proxy": features.crosswind_proxy,
        "race_phase": features.race_phase,
        "laps_remaining": features.laps_remaining,
    }


class TelemetryAnalysisAgent(RaceAgent):
    name = "telemetry"

    def analyze(
        self,
        window: TelemetryWindow,
        features: RaceFeatures,
        retriever: HybridMemoryRetriever,
    ) -> AgentFinding:
        evidence = multi_source_evidence(
            retriever,
            window.track_id,
            knowledge_query=f"{window.track_id} sector {features.sector} braking lockup instability tire wear",
            fastf1_query=f"{window.track_id} sector fastest lap braking brake temperature",
            jolpica_query=f"{window.track_id} race result fastest lap sector",
        )

        clf = _get_classifier()
        if clf is not None:
            t = _thresh.get(window.track_id)
            return self._classify(features, evidence, clf, t)
        return self._rule_based(window, features, evidence)

    def _classify(self, features: RaceFeatures, evidence, clf, t) -> AgentFinding:
        label, conf, proba = clf.predict(features)
        ood: float | None = None
        if hasattr(clf, "ood_score"):
            ood = float(clf.ood_score(features))
            if ood > 4.0:
                logger.warning("telemetry: OOD features (max_z=%.1f) — confidence penalised", ood)
                conf = conf * 0.85
        # Safety floor: CRITICAL and WARNING rules must not be suppressed to INFO.
        if label == "INFO":
            if features.brake_temp_front_max > t.brake_temp_critical_c or features.lockup_count >= 2:
                label = "WARNING"
                conf = max(conf, 0.65)
            elif features.fl_degradation_pressure > t.fl_degradation_pressure_critical or features.fl_wear_slope > 0.008:
                label = "WATCH"
                conf = max(conf, 0.58)
        elif label == "WATCH":
            if features.brake_temp_front_max > t.brake_temp_critical_c or features.lockup_count >= 2:
                label = "WARNING"
                conf = max(conf, 0.65)
        risk = RiskLevel[label]
        class_probs = {cls: round(float(p), 4) for cls, p in zip(clf.classes_, proba)}
        return AgentFinding(
            agent=self.name,
            risk=risk,
            summary=_summary(risk, features),
            confidence=round(min(0.92, max(0.48, conf)), 4),
            evidence=evidence,
            features=_base_features(features),
            class_probabilities=class_probs, clf_source="classifier",
            ood_score=round(ood, 3) if ood is not None else None,
            ood_flagged=ood is not None and ood > 4.0,
        )

    def _rule_based(
        self,
        window: TelemetryWindow,
        features: RaceFeatures,
        evidence,
    ) -> AgentFinding:
        t = _thresh.get(window.track_id)

        if features.brake_temp_front_max > t.brake_temp_critical_c or features.lockup_count >= 2:
            conf = 0.82
            conf += 0.05 if features.lockup_count >= 2 and features.brake_fade_risk > 12 else 0.0
            conf += 0.03 if features.fl_degradation_pressure > 0.70 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.CRITICAL,
                summary=_summary(RiskLevel.CRITICAL, features),
                confidence=min(0.92, conf),
                evidence=evidence,
                features=_base_features(features),
            )

        if features.fl_degradation_pressure > t.fl_degradation_pressure_critical or features.fl_wear_slope > 0.008:
            conf = 0.77
            conf += 0.04 if features.fr_wear_slope > 0.003 else 0.0
            conf += 0.03 if features.fl_degradation_pressure > t.fl_degradation_pressure_warning else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WARNING,
                summary=_summary(RiskLevel.WARNING, features),
                confidence=min(0.86, conf),
                evidence=evidence,
                features=_base_features(features),
            )

        if features.brake_fade_risk > 12.0 or features.crosswind_proxy > t.crosswind_watch * 0.85:
            conf = 0.64
            conf += 0.04 if features.brake_fade_risk > 12 and features.crosswind_proxy > 8 else 0.0
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WATCH,
                summary=_summary(RiskLevel.WATCH, features),
                confidence=min(0.70, conf),
                evidence=evidence,
                features=_base_features(features),
            )

        conf = max(0.48, 0.60 - features.fl_degradation_pressure * 0.10)
        return AgentFinding(
            agent=self.name,
            risk=RiskLevel.INFO,
            summary="Telemetry envelope nominal.",
            confidence=conf,
            evidence=evidence,
            features=_base_features(features),
        )


def _summary(risk: RiskLevel, features: RaceFeatures) -> str:
    if risk == RiskLevel.CRITICAL:
        return "Front braking envelope is unstable with repeated lockup or excessive temperature."
    if risk == RiskLevel.WARNING:
        return "Front-left degradation pressure is accelerating beyond the current stint projection."
    if risk == RiskLevel.WATCH:
        if features.brake_fade_risk > 12.0:
            return "Brake temperatures are trending upward under sustained load."
        return "Crosswind sensitivity is increasing under steering load."
    return "Telemetry envelope nominal."
