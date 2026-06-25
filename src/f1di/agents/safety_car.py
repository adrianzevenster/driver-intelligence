from __future__ import annotations

import logging
from pathlib import Path

from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever

logger = logging.getLogger("f1di.agents.safety_car")

_CLASSIFIER_PATH = Path("data/calibration/safety_car_classifier.pkl")
_clf_cache: object = None
_clf_mtime: float = 0.0


def _get_classifier():
    global _clf_cache, _clf_mtime
    if not _CLASSIFIER_PATH.exists():
        return None
    try:
        mtime = _CLASSIFIER_PATH.stat().st_mtime
    except OSError:
        return None
    if mtime != _clf_mtime:
        try:
            from f1di.agents.safety_car_classifier import SafetyCarClassifier
            _clf_cache = SafetyCarClassifier.load(_CLASSIFIER_PATH)
            _clf_mtime = mtime
            logger.info(
                "SafetyCarClassifier reloaded: n_real=%d acc=%.3f",
                _clf_cache.n_real, _clf_cache.accuracy,
            )
        except Exception as exc:
            logger.warning("SafetyCarClassifier load failed — using rules: %s", exc)
            _clf_cache = None
            _clf_mtime = mtime
    return _clf_cache


def _base_features(features: RaceFeatures) -> dict[str, float]:
    return {
        "mean_speed_kph":      features.mean_speed_kph,
        "speed_delta_kph":     features.speed_delta_kph,
        "rain_intensity":      features.rain_intensity,
        "grip_estimate":       features.grip_estimate,
        "lockup_count":        float(features.lockup_count),
        "throttle_smoothness": features.throttle_smoothness,
        "race_phase":          features.race_phase,
        "brake_temp_front_max": features.brake_temp_front_max,
    }


def _summary(risk_str: str, features: RaceFeatures) -> tuple[str, dict]:
    base = _base_features(features)

    if risk_str == "CRITICAL":
        if features.mean_speed_kph < 80.0:
            msg = (
                "Safety car or VSC appears deployed: telemetry shows major speed reduction "
                f"({features.mean_speed_kph:.0f} kph). Pit now if window is strategically beneficial."
            )
        else:
            msg = (
                "Safety car conditions imminent: extreme rain and grip loss signal high SC probability. "
                "Prepare immediate pit call."
            )
        return msg, {**base, "speed_kph": features.mean_speed_kph}

    if risk_str == "WARNING":
        if features.rain_intensity > 0.4:
            msg = (
                f"Conditions indicate elevated SC/VSC risk within next 2-3 laps "
                f"(rain={features.rain_intensity:.2f}, grip={features.grip_estimate:.2f}). "
                "Open pit window discussion now."
            )
        else:
            msg = (
                "Speed profile anomaly suggests possible incident ahead. "
                "Monitor sector gaps and prepare pit window."
            )
        return msg, base

    if risk_str == "WATCH":
        if features.rain_intensity > 0.3:
            msg = f"Light-to-moderate rain (intensity={features.rain_intensity:.2f}) raising incident probability. Monitor closely."
        elif features.lockup_count > 0:
            msg = "Braking anomalies detected; may indicate yellow flag zones ahead. Monitor for SC deployment."
        else:
            msg = f"Grip below normal envelope ({features.grip_estimate:.2f}); track conditions raising incident risk."
        return msg, base

    return "No safety car indicators present. Normal racing conditions.", {}


class SafetyCarAgent(RaceAgent):
    name = "safety_car"

    def analyze(self, window: TelemetryWindow, features: RaceFeatures, retriever: HybridMemoryRetriever) -> AgentFinding:
        evidence = multi_source_evidence(
            retriever,
            window.track_id,
            knowledge_query=f"{window.track_id} safety car VSC deployment strategy pit window",
            fastf1_query=f"{window.track_id} safety car virtual safety car incident lap",
            jolpica_query=f"{window.track_id} safety car race result strategy impact",
        )

        clf = _get_classifier()
        if clf is not None:
            return self._classify(clf, features, evidence)

        return self._rule_based(features, evidence)

    def _classify(self, clf, features: RaceFeatures, evidence: list) -> AgentFinding:
        risk_str, conf, proba = clf.predict(features)
        ood: float | None = None
        if hasattr(clf, "ood_score"):
            ood = float(clf.ood_score(features))
            if ood > 4.0:
                logger.warning("safety_car: OOD features (max_z=%.1f) — confidence penalised", ood)
                conf = conf * 0.85

        # Safety floor: hard CRITICAL when speed clearly SC-level
        if risk_str in ("INFO", "WATCH", "WARNING") and features.mean_speed_kph < 80.0:
            risk_str = "CRITICAL"
            conf = max(conf, 0.88)
        elif risk_str == "INFO" and features.rain_intensity > 0.6 and features.grip_estimate < 0.58:
            risk_str = "WARNING"
            conf = max(conf, 0.70)
        elif risk_str == "INFO" and (features.rain_intensity > 0.35 or features.grip_estimate < 0.72):
            risk_str = "WATCH"
            conf = max(conf, 0.58)
        # Safety ceiling: cap WARNING when the triggering conditions are not present.
        # Prevents the classifier from over-predicting WARNING on moderate conditions.
        elif risk_str == "WARNING":
            speed_flag = features.mean_speed_kph < 160.0 or features.speed_delta_kph < -110.0
            rain_flag = features.rain_intensity > 0.5 and features.grip_estimate < 0.65
            if not (speed_flag or rain_flag):
                risk_str = "WATCH"
                conf = min(conf, 0.68)

        conf = min(0.92, max(0.48, conf))
        summary, feat_dict = _summary(risk_str, features)
        class_probs = {cls: round(float(p), 4) for cls, p in zip(clf.classes_, proba)}
        return AgentFinding(
            agent=self.name, risk=RiskLevel[risk_str],
            summary=summary, confidence=conf,
            evidence=evidence, features=feat_dict,
            class_probabilities=class_probs, clf_source="classifier",
            ood_score=round(ood, 3) if ood is not None else None,
            ood_flagged=ood is not None and ood > 4.0,
        )

    def _rule_based(self, features: RaceFeatures, evidence: list) -> AgentFinding:
        base = _base_features(features)

        if features.mean_speed_kph < 80.0:
            return AgentFinding(
                agent=self.name, risk=RiskLevel.CRITICAL,
                summary=(
                    f"Safety car or VSC appears deployed: telemetry shows major speed reduction "
                    f"({features.mean_speed_kph:.0f} kph). Pit now if window is strategically beneficial."
                ),
                confidence=0.90,
                evidence=evidence,
                features={**base, "speed_kph": features.mean_speed_kph},
            )

        if features.rain_intensity > 0.7 and features.grip_estimate < 0.55:
            return AgentFinding(
                agent=self.name, risk=RiskLevel.CRITICAL,
                summary=(
                    "Extreme rain and grip loss signal very high SC probability. "
                    "Prepare immediate pit call."
                ),
                confidence=0.82,
                evidence=evidence,
                features=base,
            )

        if features.mean_speed_kph < 160.0 or features.speed_delta_kph < -60.0:
            conf = 0.75
            conf += 0.04 if features.speed_delta_kph < -80.0 else 0.0
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WARNING,
                summary=(
                    "Speed profile anomaly suggests possible incident ahead. "
                    "Monitor sector gaps and prepare pit window."
                ),
                confidence=min(0.85, conf),
                evidence=evidence,
                features=base,
            )

        if features.rain_intensity > 0.5 and features.grip_estimate < 0.65:
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WARNING,
                summary=(
                    f"Conditions indicate elevated SC/VSC risk within next 2-3 laps "
                    f"(rain={features.rain_intensity:.2f}, grip={features.grip_estimate:.2f}). "
                    "Open pit window discussion now."
                ),
                confidence=0.73,
                evidence=evidence,
                features=base,
            )

        if features.rain_intensity > 0.35 or features.grip_estimate < 0.72:
            conf = 0.63
            conf += 0.04 if features.lockup_count > 0 else 0.0
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WATCH,
                summary=f"Track conditions raising incident risk (rain={features.rain_intensity:.2f}, grip={features.grip_estimate:.2f}). Monitor for SC deployment.",
                confidence=min(0.70, conf),
                evidence=evidence,
                features=base,
            )

        if features.lockup_count >= 2 and features.brake_temp_front_max > 500.0:
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WATCH,
                summary="Braking anomalies detected; may indicate yellow flag zones ahead. Monitor for SC deployment.",
                confidence=0.60,
                evidence=evidence,
                features=base,
            )

        conf = max(0.52, 0.68 - features.rain_intensity * 0.25)
        return AgentFinding(
            agent=self.name, risk=RiskLevel.INFO,
            summary="No safety car indicators present. Normal racing conditions.",
            confidence=conf,
            evidence=evidence,
        )
