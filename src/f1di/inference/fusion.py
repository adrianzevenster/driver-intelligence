from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from f1di.agents.battery import BatteryAgent
from f1di.agents.fuel import FuelAgent
from f1di.agents.safety_car import SafetyCarAgent
from f1di.agents.telemetry import TelemetryAnalysisAgent
from f1di.agents.tire import TireStrategyAgent
from f1di.agents.weather import WeatherAgent
from f1di.confidence.calibration import ConfidenceCalibrator, RISK_WEIGHT
from f1di.config.settings import settings
from f1di.domain.schemas import DriverInsight, InsightAudience, RiskLevel, TelemetryWindow
from f1di.features.extractor import extract_features
from f1di.rag import make_retriever
from f1di.rag.store import HybridMemoryRetriever, load_markdown_knowledge

logger = logging.getLogger(__name__)

RISK_ORDER = [RiskLevel.INFO, RiskLevel.WATCH, RiskLevel.WARNING, RiskLevel.CRITICAL]

_CALIBRATOR_PATH = Path("data/calibration/isotonic.pkl")
_META_PATH = Path("data/calibration/meta_learner.pkl")

_meta_cache: object = None
_meta_mtime: float = 0.0


def _get_meta_learner():
    global _meta_cache, _meta_mtime
    if not _META_PATH.exists():
        return None
    try:
        mtime = _META_PATH.stat().st_mtime
    except OSError:
        return None
    if mtime != _meta_mtime:
        try:
            from f1di.inference.meta_learner import MetaLearner
            _meta_cache = MetaLearner.load(_META_PATH)
            _meta_mtime = mtime
        except Exception:
            _meta_cache = None
            _meta_mtime = mtime
    return _meta_cache


class InferenceOrchestrator:
    def __init__(self, retriever=None) -> None:
        self.retriever = retriever or make_retriever()
        kb_path = Path(settings.knowledge_path)
        if kb_path.exists():
            if isinstance(self.retriever, HybridMemoryRetriever):
                if not self.retriever.documents:
                    self.retriever.add_documents(load_markdown_knowledge(kb_path))
            else:
                try:
                    # Qdrant: always upsert so new/updated circuit docs land with correct metadata
                    self.retriever.add_documents(load_markdown_knowledge(kb_path))
                except Exception as exc:
                    logger.warning("knowledge_upsert_failed (retrieval will degrade): %s", exc)
        self.agents = [TelemetryAnalysisAgent(), TireStrategyAgent(), WeatherAgent(), BatteryAgent(), SafetyCarAgent(), FuelAgent()]
        self.calibrator = self._load_calibrator()

    @staticmethod
    def _load_calibrator() -> ConfidenceCalibrator:
        if _CALIBRATOR_PATH.exists():
            try:
                return ConfidenceCalibrator.load(_CALIBRATOR_PATH)
            except Exception:
                pass
        return ConfidenceCalibrator()

    def analyze(
        self,
        window: TelemetryWindow,
        audience: InsightAudience = InsightAudience.DRIVER,
        skip_llm: bool = False,
        record_drift: bool = True,
    ) -> DriverInsight:
        """skip_llm and record_drift exist for batch replay callers (e.g. a
        whole-race lap-by-lap strategy comparison) that need many analyses
        per call: skip_llm avoids paying for an LLM narrative on every lap,
        and record_drift=False keeps replayed laps out of the live drift
        tracker's baseline, which should only reflect real-time traffic.
        """
        start = time.perf_counter()
        features = extract_features(window)

        if record_drift:
            try:
                from f1di.observability.drift import features_as_dict, get_tracker
                get_tracker().update(features_as_dict(features), track_id=window.track_id)
            except Exception:
                pass

        findings = [agent.analyze(window, features, self.retriever) for agent in self.agents]
        highest = max(findings, key=lambda f: RISK_WEIGHT[f.risk])
        confidence, uncertainty, calibration_features, raw_score = self.calibrator.calibrate(findings)

        meta = _get_meta_learner()
        if meta is not None and meta.n_real >= 20:
            meta_conf = meta.predict_confidence(findings, confidence)
            confidence = round(0.6 * meta_conf + 0.4 * confidence, 4)
            uncertainty = round(max(0.0, 1.0 - confidence), 4)

        shap_explanation: list[dict] = []
        try:
            from f1di.inference.explainer import explain_findings
            shap_explanation = explain_findings(findings, confidence)
        except Exception:
            pass

        recommendation = self._rules_recommendation(highest.risk, findings, calibration_features)
        if not skip_llm and settings.llm_backend != "rules" and not settings.deterministic:
            recommendation = self._llm_recommendation(
                window, findings, highest, audience, confidence, recommendation
            )

        supporting = [f.summary for f in findings if f.risk != RiskLevel.INFO] or [
            "No critical deviation from expected race envelope."
        ]
        evidence = []
        seen: set[str] = set()
        for finding in findings:
            for item in finding.evidence:
                if item.source_id not in seen:
                    evidence.append(item)
                    seen.add(item.source_id)

        policy = self._policy(audience, confidence, highest.risk)
        return DriverInsight(
            insight_id=str(uuid.uuid4()),
            session_id=window.session_id,
            driver_id=window.driver_id,
            audience=audience,
            risk=highest.risk,
            recommendation=recommendation,
            confidence=confidence,
            uncertainty=uncertainty,
            raw_score=raw_score,
            supporting_factors=supporting,
            evidence=evidence[:5],
            findings=findings,
            policy=policy,
            latency_ms=(time.perf_counter() - start) * 1000,
            shap_explanation=shap_explanation,
        )

    def _llm_recommendation(
        self,
        window: TelemetryWindow,
        findings,
        highest,
        audience: InsightAudience,
        confidence: float,
        fallback: str,
    ) -> str:
        from f1di.llm.advisor import generate_recommendation

        evidence_snippets = []
        seen: set[str] = set()
        for f in findings:
            for e in f.evidence:
                if e.source_id not in seen:
                    evidence_snippets.append(f"{e.title}: {e.text[:150]}")
                    seen.add(e.source_id)
                    if len(evidence_snippets) >= 3:
                        break
            if len(evidence_snippets) >= 3:
                break

        result = generate_recommendation(
            risk=highest.risk,
            findings=findings,
            audience=audience,
            calibrated_confidence=confidence,
            evidence_snippets=evidence_snippets,
            compound=window.latest.compound.value,
            stint_lap=window.latest.stint_lap,
        )
        return result if result else fallback

    def _policy(self, audience: InsightAudience, confidence: float, risk: RiskLevel) -> str:
        if risk in {RiskLevel.WARNING, RiskLevel.CRITICAL}:
            return "SHOW"
        if audience == InsightAudience.DRIVER and confidence < settings.confidence_min_driver:
            return "ENGINEER_ONLY"
        if confidence < settings.confidence_min_engineer:
            return "SUPPRESS"
        return "SHOW"

    def _rules_recommendation(self, risk: RiskLevel, findings, calibration_features: dict[str, float]) -> str:
        if risk == RiskLevel.CRITICAL:
            sc = next((f for f in findings if f.agent == "safety_car" and f.risk == RiskLevel.CRITICAL), None)
            if sc:
                return "Safety car or VSC likely deployed — pit now if strategically beneficial. Alert driver to back off and close gap to car ahead."
            return "Prioritise stability: reduce push intensity, confirm brake/tire state, and prepare immediate strategy intervention."

        if risk == RiskLevel.WARNING:
            tire = next((f for f in findings if f.agent == "tire_strategy" and f.risk == RiskLevel.WARNING), None)
            if tire:
                if "projected_fl_4laps" in tire.features or "projected_fr_4laps" in tire.features:
                    proj = max(tire.features.get("projected_fl_4laps", 0), tire.features.get("projected_fr_4laps", 0))
                    front = "FR" if tire.features.get("projected_fr_4laps", 0) > tire.features.get("projected_fl_4laps", 0) else "FL"
                    return f"Degradation trajectory projects {front} past cliff in ~4 laps (projected: {proj:.2f}); open pit discussion now and protect the tyre on entry."
                if tire.features.get("axle_imbalance", 0) > 0.15:
                    return "Open pit-window discussion now; pronounced axle imbalance — avoid aggressive kerbs and entry lock-up risk."
                return "Open pit-window discussion now; protect front-left and avoid aggressive entry kerbs."
            weather = next((f for f in findings if f.agent == "weather" and f.risk == RiskLevel.WARNING), None)
            if weather:
                rain = weather.features.get("rain_intensity", 0)
                grip = weather.features.get("grip_estimate", 1.0)
                if grip < 0.65:
                    return "Rain intensity has crossed the intermediate threshold; open compound switch discussion and monitor evolving track conditions."
                return f"Rain intensity {rain:.2f} approaching crossover; prepare intermediate switch discussion and adjust braking reference points."
            battery = next((f for f in findings if f.agent == "battery" and f.risk == RiskLevel.WARNING), None)
            if battery:
                return "ERS depletion rate exceeds recovery capacity; adjust deployment strategy and reduce harvest-zone aggression."
            sc = next((f for f in findings if f.agent == "safety_car" and f.risk == RiskLevel.WARNING), None)
            if sc:
                return "Elevated SC/VSC probability — open pit window discussion now and monitor sector time updates for confirmation."
            fuel = next((f for f in findings if f.agent == "fuel" and f.risk == RiskLevel.WARNING), None)
            if fuel:
                return "Fuel consumption above plan — engage lift-and-coast on straights and reduce deployment aggression."
            return "Adjust driving mode and monitor next telemetry window before escalating to driver comms."

        if risk == RiskLevel.WATCH:
            watch = [f for f in findings if f.risk == RiskLevel.WATCH]
            agents = {f.agent for f in watch}
            if "tire_strategy" in agents:
                tw = next(f for f in watch if f.agent == "tire_strategy")
                if "fr_wear_slope" in tw.features:
                    return "FR wearing asymmetrically; monitor balance and avoid over-reliance on front-right braking stability."
                if "axle_imbalance" in tw.features:
                    return "Monitor front-rear imbalance; avoid aggressive entry kerbs for the next two laps."
                return "Rear degradation above expected envelope; monitor traction out of slow corners."
            if "weather" in agents:
                return "Track conditions evolving; prepare compound switch discussion if rain intensifies."
            if "battery" in agents:
                return "Monitor ERS state; adjust deployment strategy if depletion rate persists beyond this sector."
            if "safety_car" in agents:
                return "Track conditions elevated; monitor for SC/VSC deployment before committing to track position."
            if "fuel" in agents:
                return "Fuel burn slightly elevated; begin light lift-and-coast on long straights and monitor next window."
            return "Keep monitoring; provide engineer-only context unless the next window confirms the trend."

        return "Continue current plan."
