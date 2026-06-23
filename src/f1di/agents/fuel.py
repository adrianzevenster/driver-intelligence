from __future__ import annotations

import logging
from pathlib import Path

from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever

logger = logging.getLogger("f1di.agents.fuel")

_CLASSIFIER_PATH = Path("data/calibration/fuel_classifier.pkl")
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
            from f1di.agents.fuel_classifier import FuelClassifier
            _clf_cache = FuelClassifier.load(_CLASSIFIER_PATH)
            _clf_mtime = mtime
            logger.info(
                "FuelClassifier reloaded: n_real=%d acc=%.3f",
                _clf_cache.n_real, _clf_cache.accuracy,
            )
        except Exception as exc:
            logger.warning("FuelClassifier load failed — using rules: %s", exc)
            _clf_cache = None
            _clf_mtime = mtime
    return _clf_cache


def _base_features(features: RaceFeatures) -> dict[str, float]:
    return {
        "throttle_mean":      features.throttle_mean,
        "ers_net_deploy_kw":  features.ers_net_deploy_kw,
        "battery_soc":        features.battery_soc,
        "laps_remaining":     features.laps_remaining,
        "race_phase":         features.race_phase,
        "stint_fraction":     features.stint_fraction,
        "throttle_smoothness": features.throttle_smoothness,
    }


def _summary(risk_str: str, features: RaceFeatures) -> tuple[str, dict]:
    base = _base_features(features)
    if risk_str == "WARNING":
        laps = int(features.laps_remaining)
        msg = (
            f"Fuel consumption rate is above planned load; active save mode required. "
            f"Apply lift-and-coast on straights to recover margin over the remaining {laps} laps."
        )
        return msg, {**base, "throttle_mean": features.throttle_mean}
    if risk_str == "WATCH":
        if features.race_phase < 0.3:
            msg = (
                f"Throttle demand high in opening phase ({features.throttle_mean:.0f}% mean) "
                "on full fuel load. Monitor consumption rate before committing to push lap."
            )
        else:
            msg = (
                f"Fuel burn slightly elevated (throttle={features.throttle_mean:.0f}%, "
                f"ERS net={features.ers_net_deploy_kw:.0f} kW). "
                "Consider increasing lift-and-coast frequency on DRS straights."
            )
        return msg, base
    return "Fuel consumption on plan. No intervention required.", {}


class FuelAgent(RaceAgent):
    name = "fuel"

    def analyze(self, window: TelemetryWindow, features: RaceFeatures, retriever: HybridMemoryRetriever) -> AgentFinding:
        evidence = multi_source_evidence(
            retriever,
            window.track_id,
            knowledge_query=f"{window.track_id} fuel management lift and coast straight strategy",
            fastf1_query=f"{window.track_id} fuel load consumption rate lap time delta",
            jolpica_query=f"{window.track_id} fuel strategy race management",
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
                logger.warning("fuel: OOD features (max_z=%.1f) — confidence penalised", ood)
                conf = conf * 0.85

        fuel_pressure = (
            features.throttle_mean / 100.0
            - features.ers_net_deploy_kw / 500.0
            - features.battery_soc * 0.15
        )
        # Safety floor: promote INFO→WATCH when rule threshold is clearly crossed
        if risk_str == "INFO" and fuel_pressure > 0.55 and features.laps_remaining > 8:
            risk_str = "WATCH"
            conf = max(conf, 0.60)
        # Safety ceiling: WARNING only when the rule threshold is met (pressure > 0.65,
        # laps > 12); looser conditions warrant at most WATCH.
        if risk_str == "WARNING" and not (fuel_pressure > 0.65 and features.laps_remaining > 12):
            risk_str = "WATCH"
            conf = min(conf, 0.68)

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

    def _rule_based(self, features: RaceFeatures, evidence: list) -> AgentFinding:
        base = _base_features(features)
        fuel_pressure = (
            features.throttle_mean / 100.0
            - features.ers_net_deploy_kw / 500.0
            - features.battery_soc * 0.15
        )

        if fuel_pressure > 0.55 and features.laps_remaining > 8 and features.throttle_smoothness < 0.60:
            laps = int(features.laps_remaining)
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WARNING,
                summary=(
                    f"Fuel consumption rate is above planned load; active save mode required. "
                    f"Apply lift-and-coast on straights to recover margin over the remaining {laps} laps."
                ),
                confidence=0.76,
                evidence=evidence,
                features={**base, "fuel_pressure": round(fuel_pressure, 3)},
            )

        if fuel_pressure > 0.40 and features.laps_remaining > 6:
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WATCH,
                summary=(
                    f"Fuel burn slightly elevated (throttle={features.throttle_mean:.0f}%, "
                    f"ERS net={features.ers_net_deploy_kw:.0f} kW). "
                    "Consider increasing lift-and-coast frequency on DRS straights."
                ),
                confidence=0.67,
                evidence=evidence,
                features=base,
            )

        if features.race_phase < 0.25 and features.throttle_mean > 83:
            return AgentFinding(
                agent=self.name, risk=RiskLevel.WATCH,
                summary=(
                    f"Throttle demand high in opening phase ({features.throttle_mean:.0f}% mean) "
                    "on full fuel load. Monitor consumption rate before committing to push lap."
                ),
                confidence=0.62,
                evidence=evidence,
                features=base,
            )

        conf = max(0.54, 0.70 - fuel_pressure * 0.30)
        return AgentFinding(
            agent=self.name, risk=RiskLevel.INFO,
            summary="Fuel consumption on plan. No intervention required.",
            confidence=conf,
            evidence=evidence,
        )
