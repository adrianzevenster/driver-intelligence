from __future__ import annotations

import logging

from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.agents import thresholds as _thresh
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever

from f1di.agents.classifier_utils import _CALIBRATION_DIR

logger = logging.getLogger("f1di.agents.weather")
_CLASSIFIER_PATH = _CALIBRATION_DIR / "weather_classifier.pkl"
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
            from f1di.agents.weather_classifier import WeatherClassifier
            _clf_cache = WeatherClassifier.load(_CLASSIFIER_PATH)
            _clf_mtime = mtime
            logger.info("WeatherClassifier reloaded: n_real=%d acc=%.3f", _clf_cache.n_real, _clf_cache.accuracy)
        except Exception as exc:
            logger.warning("WeatherClassifier load failed — using rules: %s", exc)
            _clf_cache = None
            _clf_mtime = mtime
    return _clf_cache


def _base_features(features: RaceFeatures) -> dict[str, float]:
    return {
        "rain_intensity": features.rain_intensity,
        "grip_estimate": features.grip_estimate,
        "crosswind_proxy": features.crosswind_proxy,
        "brake_fade_risk": features.brake_fade_risk,
        "race_phase": features.race_phase,
        "circuit_avg_speed_kph": features.circuit_avg_speed_kph,
        "circuit_type_enc": features.circuit_type_enc,
        "race_laps_total": features.race_laps_total,
    }


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

        clf = _get_classifier()
        if clf is not None:
            risk_str, conf, proba = clf.predict(features)
            ood: float | None = None
            if hasattr(clf, "ood_score"):
                ood = float(clf.ood_score(features))
                if ood > 4.0:
                    logger.warning("weather: OOD features (max_z=%.1f) — confidence penalised", ood)
                    conf = conf * 0.85
            # Hard override: rain above the warning threshold must fire at least WARNING.
            # The classifier deprioritises rain when crosswind is low, but a rain ≥ crossover
            # reading is always a strategy-relevant signal regardless of wind direction.
            t = _thresh.get(window.track_id)
            if features.rain_intensity >= t.rain_warning and risk_str in ("INFO", "WATCH"):
                risk_str = "WARNING"
                conf = max(conf, 0.76 + (0.04 if features.grip_estimate < 0.65 else 0.0))
            conf = min(0.88, max(0.48, conf))
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

        if features.rain_intensity >= t.rain_warning:
            conf = 0.76
            conf += 0.04 if features.grip_estimate < 0.65 else 0.0
            conf += 0.02 if features.crosswind_proxy > 8 else 0.0
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WARNING,
                summary="Rain intensity is approaching crossover territory; monitor inter timing.",
                confidence=min(0.84, conf), evidence=evidence,
                features={**base, "rain_intensity": features.rain_intensity, "grip_estimate": features.grip_estimate},
            )

        if features.crosswind_proxy > t.crosswind_watch:
            conf = 0.67
            conf += 0.04 if features.brake_fade_risk > 8 else 0.0
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WATCH,
                summary="Crosswind is likely affecting braking stability and turn-in confidence.",
                confidence=min(0.73, conf), evidence=evidence,
                features={**base, "crosswind_proxy": features.crosswind_proxy, "brake_fade_risk": features.brake_fade_risk},
            )

        conf = max(0.55, 0.65 - features.rain_intensity * 0.20)
        return AgentFinding(
            agent=self.name, risk=RiskLevel.INFO,
            summary="Weather signal does not require strategy change.",
            confidence=conf, evidence=evidence,
        )


def _summary(risk_str: str, features: RaceFeatures) -> tuple[str, dict[str, float]]:
    base = _base_features(features)
    if risk_str == "WARNING":
        return (
            "Rain intensity is approaching crossover territory; monitor inter timing.",
            {**base, "rain_intensity": features.rain_intensity, "grip_estimate": features.grip_estimate},
        )
    if risk_str == "WATCH":
        return (
            "Crosswind is likely affecting braking stability and turn-in confidence.",
            {**base, "crosswind_proxy": features.crosswind_proxy},
        )
    return "Weather signal does not require strategy change.", {}
