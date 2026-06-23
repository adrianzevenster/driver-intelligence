from __future__ import annotations

import logging
from pathlib import Path

from f1di.agents.base import RaceAgent, multi_source_evidence
from f1di.agents import thresholds as _thresh
from f1di.domain.schemas import AgentFinding, RiskLevel, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever

logger = logging.getLogger("f1di.agents.tire")

_CLASSIFIER_PATH = Path("data/calibration/tire_classifier.pkl")

# Module-level cache with mtime tracking — reloads automatically when the pkl
# is updated by the flywheel scheduler without requiring a process restart.
_clf_cache: object = None  # TireClassifier | None
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
            from f1di.agents.tire_classifier import TireClassifier
            _clf_cache = TireClassifier.load(_CLASSIFIER_PATH)
            _clf_mtime = mtime
            logger.info(
                "TireClassifier reloaded: n_real=%d acc=%.3f",
                _clf_cache.n_real, _clf_cache.accuracy,
            )
        except Exception as exc:
            logger.warning("TireClassifier load failed — using rules: %s", exc)
            _clf_cache = None
            _clf_mtime = mtime  # don't retry every call on a corrupt file
    return _clf_cache


def _clf_features(features: RaceFeatures, wear_pressure: float) -> dict[str, float]:
    """Full feature set stored in every finding for flywheel training data."""
    return {
        "wear_pressure": wear_pressure,
        "grip_estimate": features.grip_estimate,
        "fl_wear_slope": features.fl_wear_slope,
        "fr_wear_slope": features.fr_wear_slope,
        "rear_wear_slope": features.rear_wear_slope,
        "axle_imbalance_fl_rl": features.axle_imbalance_fl_rl,
        "laps_remaining": features.laps_remaining,
        "stint_fraction": features.stint_fraction,
        "race_phase": features.race_phase,
    }


def _cross_check(finding: AgentFinding, cliff: dict) -> AgentFinding:
    """Adjust confidence and risk based on classifier/projection agreement.

    Three adjustments:
    - Both agree cliff imminent (WARNING/CRITICAL + eta ≤ 4): +0.05 confidence.
    - Projection sees cliff in ≤ 3 laps that classifier missed: upgrade to WARNING.
    - Classifier says CRITICAL but projection sees nothing: −0.05 confidence.

    The clf_agrees_cliff / clf_disagrees_cliff flags land in features so the
    flywheel can learn from signal disagreements over time.
    """
    eta = cliff["eta_laps"]
    risk = finding.risk.value
    updates: dict = {}
    feat = dict(finding.features)

    if risk in ("WARNING", "CRITICAL") and eta is not None and eta <= 4:
        feat["clf_agrees_cliff"] = True
        updates["confidence"] = min(0.94, finding.confidence + 0.05)
    elif risk in ("INFO", "WATCH") and eta is not None and eta <= 3:
        feat["clf_agrees_cliff"] = False
        updates["risk"] = RiskLevel.WARNING
        eta_int = max(1, int(round(eta)))
        updates["summary"] = (
            f"Monte Carlo projection indicates critical wear threshold within "
            f"{eta_int} lap{'s' if eta_int != 1 else ''}: "
            f"tire cliff window opening despite borderline classifier assessment."
        )
        updates["confidence"] = 0.68
    elif risk == "CRITICAL" and eta is None:
        feat["clf_disagrees_cliff"] = True
        updates["confidence"] = max(0.48, finding.confidence - 0.05)

    if feat != dict(finding.features):
        updates["features"] = feat

    return finding.model_copy(update=updates) if updates else finding


def _summary_and_extras(
    risk_str: str,
    features: RaceFeatures,
    wear_pressure: float,
    projected_fl: float,
    projected_fr: float,
) -> tuple[str, dict[str, float]]:
    """Derive a human-readable summary and diagnostic extras from risk + feature values."""
    axle_flag = features.axle_imbalance_fl_rl > 0.15
    base = _clf_features(features, wear_pressure)

    if risk_str == "CRITICAL":
        summary = (
            "Box window should be opened: tire wear and grip loss indicate imminent performance cliff with axle imbalance."
            if axle_flag else
            "Box window should be opened: tire wear and grip loss indicate imminent performance cliff."
        )
        return summary, {**base, "grip": features.grip_estimate, "axle_imbalance": features.axle_imbalance_fl_rl}

    if risk_str == "WARNING":
        if axle_flag:
            summary = "Prepare pit window within two laps; axle imbalance indicates corner-entry instability risk."
        elif max(features.fl_wear_slope, features.fr_wear_slope) > 0.008:
            front = "FL" if projected_fl >= projected_fr else "FR"
            summary = f"Degradation trajectory projects {front} past cliff threshold within four laps; open pit discussion now."
        else:
            summary = "Prepare pit window within two laps unless track position risk dominates."
        return summary, {**base, "grip": features.grip_estimate, "axle_imbalance": features.axle_imbalance_fl_rl}

    if risk_str == "WATCH":
        if features.fr_wear_slope > features.fl_wear_slope + 0.0015:
            summary = "FR degrading faster than FL; atypical axle loading — monitor for setup-driven imbalance or circuit-specific stress."
        elif axle_flag:
            summary = "Front-rear axle imbalance is building; monitor for corner-entry instability as stint extends."
        else:
            summary = "Rear tyre degradation rate is above expected envelope; assess traction and stability into braking zones."
        return summary, base

    # INFO
    return "Tire state remains within modelled stint limits.", {}


class TireStrategyAgent(RaceAgent):
    name = "tire_strategy"

    def analyze(self, window: TelemetryWindow, features: RaceFeatures, retriever: HybridMemoryRetriever) -> AgentFinding:
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
        projected_fl = features.fl_wear + features.fl_wear_slope * spl * 4
        projected_fr = features.fr_wear + features.fr_wear_slope * spl * 4

        clf = _get_classifier()
        if clf is not None:
            t = _thresh.get(window.track_id)
            finding = self._classify(clf, t, features, wear_pressure, projected_fl, projected_fr, evidence)
        else:
            finding = self._rule_based(window, features, wear_pressure, projected_fl, projected_fr, evidence)

        t = _thresh.get(window.track_id)
        from f1di.agents.tire_projection import project_cliff_for_window
        cliff = project_cliff_for_window(window, features, t.wear_critical)
        finding = _cross_check(finding, cliff)
        return finding.model_copy(update={
            "cliff_eta_laps": cliff["eta_laps"],
            "cliff_probability_by_lap": cliff["probability_by_lap"],
        })

    def _classify(
        self,
        clf,
        t,
        features: RaceFeatures,
        wear_pressure: float,
        projected_fl: float,
        projected_fr: float,
        evidence: list,
    ) -> AgentFinding:
        risk_str, conf, proba = clf.predict(features, wear_pressure)
        ood: float | None = None
        if hasattr(clf, "ood_score"):
            ood = float(clf.ood_score(features, wear_pressure))
            if ood > 4.0:
                logger.warning("tire: OOD features (max_z=%.1f) — confidence penalised", ood)
                conf = conf * 0.85
        # Safety floor: classifier acts as a minimum guarantee — rules can only raise.
        # CRITICAL floor first: extreme wear + grip collapse is always critical regardless
        # of what the classifier predicted (catches classifiers that under-fire on rare
        # high-wear embedded fixtures).
        if wear_pressure > t.wear_critical and features.grip_estimate < 0.55:
            risk_str = "CRITICAL"
            conf = max(conf, 0.82)
        elif risk_str == "INFO":
            if wear_pressure > t.wear_critical and features.grip_estimate < 0.62:
                risk_str = "WARNING"
                conf = max(conf, 0.68)
            elif wear_pressure > t.wear_warning and features.grip_estimate < 0.72:
                risk_str = "WATCH"
                conf = max(conf, 0.60)
            elif (max(projected_fl, projected_fr) > t.wear_critical * 0.97
                  and features.fl_wear_slope > 0.0):
                risk_str = "WATCH"
                conf = max(conf, 0.58)
            elif (features.fr_wear_slope > features.fl_wear_slope + 0.001
                  and features.fr_wear > 0.40):
                # FR degrading significantly faster than FL at moderate wear — flag
                # asymmetric loading before it becomes a wear-pressure issue.
                risk_str = "WATCH"
                conf = max(conf, 0.57)
        elif risk_str == "WATCH":
            if wear_pressure > t.wear_critical and features.grip_estimate < 0.62:
                risk_str = "WARNING"
                conf = max(conf, 0.68)
            elif wear_pressure > t.wear_warning and features.grip_estimate < 0.72:
                # Wear above the warning band + grip degrading must be at least WARNING.
                # Guard grip < 0.72 prevents false positives on high-wear/stable-grip tracks
                # (e.g. Silverstone with a lower circuit-specific wear_warning).
                risk_str = "WARNING"
                conf = max(conf, 0.70)
        summary, feat_dict = _summary_and_extras(risk_str, features, wear_pressure, projected_fl, projected_fr)
        conf = min(0.92, max(0.48, conf))
        class_probs = {cls: round(float(p), 4) for cls, p in zip(clf.classes_, proba)}
        return AgentFinding(
            agent=self.name,
            risk=RiskLevel[risk_str],
            summary=summary,
            confidence=conf,
            evidence=evidence,
            features=feat_dict,
            class_probabilities=class_probs, clf_source="classifier",
            ood_score=round(ood, 3) if ood is not None else None,
            ood_flagged=ood is not None and ood > 4.0,
        )

    def _rule_based(
        self,
        window: TelemetryWindow,
        features: RaceFeatures,
        wear_pressure: float,
        projected_fl: float,
        projected_fr: float,
        evidence: list,
    ) -> AgentFinding:
        t = _thresh.get(window.track_id)
        base = _clf_features(features, wear_pressure)

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
                features={**base, "grip": features.grip_estimate, "axle_imbalance": features.axle_imbalance_fl_rl},
            )

        if wear_pressure > t.wear_warning and features.grip_estimate < 0.72:
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
                features={**base, "grip": features.grip_estimate, "axle_imbalance": features.axle_imbalance_fl_rl},
            )

        projected_front_cliff = max(projected_fl, projected_fr)
        if projected_front_cliff > t.wear_critical * 0.97 and features.fl_wear_slope > 0.0:
            conf = 0.68
            conf += 0.04 if features.fr_wear_slope > 0.0015 else 0.0
            conf += 0.03 if features.rear_wear_slope > 0.001 else 0.0
            front_driving = "FL" if projected_fl >= projected_fr else "FR"
            return AgentFinding(
                agent=self.name,
                risk=RiskLevel.WARNING,
                summary=f"Degradation trajectory projects {front_driving} past cliff threshold within four laps; open pit discussion now.",
                confidence=min(0.78, conf),
                evidence=evidence,
                features={
                    **base,
                    "fl_wear": features.fl_wear,
                    "projected_fl_4laps": projected_fl,
                    "projected_fr_4laps": projected_fr,
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
                features={**base, "fr_wear": features.fr_wear},
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
                features={**base, "rear_wear_mean": features.rear_wear_mean},
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
                features={**base, "rear_wear_mean": features.rear_wear_mean},
            )

        conf = max(0.50, 0.62 - wear_pressure * 0.12)
        return AgentFinding(
            agent=self.name,
            risk=RiskLevel.INFO,
            summary="Tire state remains within modelled stint limits.",
            confidence=conf,
            evidence=evidence,
        )
