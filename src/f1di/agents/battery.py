from __future__ import annotations

import logging

from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.agents import thresholds as _thresh
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever

from f1di.agents.classifier_utils import _CALIBRATION_DIR

logger = logging.getLogger("f1di.agents.battery")
_CLASSIFIER_PATH = _CALIBRATION_DIR / "battery_classifier.pkl"
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
            from f1di.agents.battery_classifier import BatteryClassifier
            _clf_cache = BatteryClassifier.load(_CLASSIFIER_PATH)
            _clf_mtime = mtime
            logger.info("BatteryClassifier reloaded: n_real=%d acc=%.3f", _clf_cache.n_real, _clf_cache.accuracy)
        except Exception as exc:
            logger.warning("BatteryClassifier load failed — using rules: %s", exc)
            _clf_cache = None
            _clf_mtime = mtime
    return _clf_cache


def _base_features(features: RaceFeatures) -> dict[str, float]:
    return {
        "battery_soc": features.battery_soc,
        "battery_soc_slope": features.battery_soc_slope,
        "mean_speed_kph": features.mean_speed_kph,
        "race_phase": features.race_phase,
        "laps_remaining": features.laps_remaining,
        "stint_fraction": features.stint_fraction,
        "circuit_avg_speed_kph": features.circuit_avg_speed_kph,
        "circuit_type_enc": features.circuit_type_enc,
        "race_laps_total": features.race_laps_total,
    }


class BatteryAgent(RaceAgent):
    name = "battery"

    def analyze(self, window: TelemetryWindow, features: RaceFeatures, retriever: HybridMemoryRetriever) -> AgentFinding:
        evidence = multi_source_evidence(
            retriever,
            window.track_id,
            knowledge_query=f"{window.track_id} ERS deployment battery SOC exit sector {features.sector}",
            fastf1_query=f"{window.track_id} ERS battery deployment straight DRS speed",
            jolpica_query=f"{window.track_id} race fastest lap power unit",
        )

        clf = _get_classifier()
        if clf is not None:
            t = _thresh.get(window.track_id)
            risk_str, conf, proba = clf.predict(features)
            ood: float | None = None
            if hasattr(clf, "ood_score"):
                ood = float(clf.ood_score(features))
                if ood > 4.0:
                    logger.warning("battery: OOD features (max_z=%.1f) — confidence penalised", ood)
                    conf = conf * 0.85
            # Safety floor: rules act as a hard lower bound — promote INFO up if needed.
            if risk_str == "INFO":
                if features.battery_soc < t.battery_soc_warning and features.battery_soc_slope < -0.0005:
                    risk_str = "WARNING"
                    conf = max(conf, 0.72)
                elif features.battery_soc > 0.72 and features.mean_speed_kph < 220:
                    risk_str = "WATCH"
                    conf = max(conf, 0.58)
            # Safety ceiling: cap ML predictions to what the rule conditions permit.
            # WARNING requires active depletion (slope < -0.0005 per sample, ~-0.01 per lap);
            # without it the rules would return INFO, so cap back to INFO instead of warning.
            # WATCH for over-charge is only valid when speed is low (< 220 km/h).
            elif risk_str == "WARNING":
                if not (features.battery_soc < t.battery_soc_warning
                        and features.battery_soc_slope < -0.0005):
                    risk_str = "INFO"
                    conf = min(conf, 0.62)
            elif risk_str == "WATCH":
                if features.battery_soc > 0.72 and features.mean_speed_kph >= 220:
                    risk_str = "INFO"
                    conf = min(conf, 0.60)
            conf = min(0.90, max(0.48, conf))
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

        return self._rule_based(window, features, evidence)

    def _rule_based(self, window: TelemetryWindow, features: RaceFeatures, evidence: list) -> AgentFinding:
        t = _thresh.get(window.track_id)
        base = _base_features(features)

        if features.battery_soc < t.battery_soc_warning and features.battery_soc_slope < -0.0005:
            conf = 0.79
            conf += 0.04 if features.battery_soc_slope < -0.0007 else 0.0
            conf += 0.02 if features.mean_speed_kph > 220 else 0.0
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WARNING,
                summary="ERS state of charge is depleting too quickly; reduce deployment before the next high-value straight.",
                confidence=min(0.87, conf), evidence=evidence,
                features={**base, "battery_soc": features.battery_soc, "battery_soc_slope": features.battery_soc_slope},
            )

        if features.battery_soc > 0.72 and features.mean_speed_kph < 220:
            conf = 0.69
            conf += 0.03 if features.mean_speed_kph < 200 else 0.0
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WATCH,
                summary="Battery is under-deployed relative to sector speed profile; increase deployment on exit zones.",
                confidence=min(0.74, conf), evidence=evidence,
                features={**base, "mean_speed_kph": features.mean_speed_kph},
            )

        conf = max(0.55, 0.65 - abs(features.battery_soc - 0.55) * 0.12)
        return AgentFinding(
            agent=self.name, risk=RiskLevel.INFO,
            summary="ERS deployment is consistent with the current tactical envelope.",
            confidence=conf, evidence=evidence,
        )


def _summary(risk_str: str, features: RaceFeatures) -> tuple[str, dict[str, float]]:
    base = _base_features(features)
    if risk_str == "WARNING":
        return (
            "ERS state of charge is depleting too quickly; reduce deployment before the next high-value straight.",
            {**base, "battery_soc_slope": features.battery_soc_slope},
        )
    if risk_str == "WATCH":
        return (
            "Battery is under-deployed relative to sector speed profile; increase deployment on exit zones.",
            {**base, "mean_speed_kph": features.mean_speed_kph},
        )
    return "ERS deployment is consistent with the current tactical envelope.", {}
