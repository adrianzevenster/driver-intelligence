from __future__ import annotations

from pathlib import Path

from f1di.agents.battery import BatteryAgent
from f1di.agents.telemetry import TelemetryAnalysisAgent
from f1di.agents.tire import TireStrategyAgent
from f1di.agents.weather import WeatherAgent
from f1di.confidence.calibration import ConfidenceCalibrator, compute_raw_score
from f1di.domain.schemas import TelemetryWindow
from f1di.features.extractor import RaceFeatures, extract_features
from f1di.rag.store import HybridMemoryRetriever, load_markdown_knowledge
from f1di.simulator.generator import DriverProfile, IncidentPlan, SyntheticRaceSimulator

_AGENTS = [TelemetryAnalysisAgent(), TireStrategyAgent(), WeatherAgent(), BatteryAgent()]

_WEAR_CLIFF = 0.82
_WEAR_ALERT = 0.65
_GRIP_LOSS = 0.60
_BRAKE_THERMAL = 750.0
_RAIN_CLIFF = 0.42
_SOC_CRITICAL = 0.18


def _incident_kind_for_window(window: TelemetryWindow, incidents: list[IncidentPlan], lookahead_laps: int = 4) -> str | None:
    current_lap = window.latest.lap
    for inc in incidents:
        if 0 <= inc.lap - current_lap <= lookahead_laps:
            return inc.kind
    return None


def _ground_truth_label(window: TelemetryWindow, features: RaceFeatures, incident_kind: str | None = None) -> float:
    latest = window.latest
    max_wear = max(
        latest.tire_wear_fl,
        latest.tire_wear_fr,
        (latest.tire_wear_rl + latest.tire_wear_rr) / 2,
    )

    if incident_kind == "lockup" or any(s.lockup_event for s in window.samples):
        return 0.92
    if latest.brake_temp_fl_c > _BRAKE_THERMAL:
        return 0.90
    if max_wear > _WEAR_CLIFF and latest.grip_estimate < _GRIP_LOSS:
        return 0.85
    if max_wear > _WEAR_ALERT:
        return 0.70
    if latest.rain_intensity >= _RAIN_CLIFF:
        return 0.65

    lap_span = latest.lap - window.samples[0].lap
    spl = len(window.samples) / lap_span if lap_span > 0 else 1.0
    projected_fl_4laps = features.fl_wear + features.fl_wear_slope * spl * 4
    projected_fr_4laps = features.fr_wear + features.fr_wear_slope * spl * 4
    if max(projected_fl_4laps, projected_fr_4laps) > _WEAR_CLIFF * 0.97 and features.fl_wear_slope > 0.0 and max_wear > _WEAR_ALERT * 0.85:
        return 0.65

    if latest.battery_soc < _SOC_CRITICAL and features.battery_soc_slope < -0.01:
        return 0.55

    if max_wear > _WEAR_ALERT * 0.73:
        return 0.40

    return 0.20


def _build_scenarios(per_type: int) -> list[dict]:
    scenarios = []
    for i in range(per_type):
        # nominal
        scenarios.append({
            "profile": DriverProfile(driver_id=f"D{i}"),
            "incidents": [],
            "laps": 8,
        })
        # lockup incident
        scenarios.append({
            "profile": DriverProfile(driver_id=f"L{i}", braking_aggression=1.3, tire_preservation=0.85),
            "incidents": [IncidentPlan(lap=3 + (i % 4), kind="lockup", severity=0.9 + 0.1 * (i % 2))],
            "laps": 8,
        })
        # high wear — sudden degradation
        scenarios.append({
            "profile": DriverProfile(driver_id=f"W{i}", braking_aggression=1.2, tire_preservation=0.75),
            "incidents": [IncidentPlan(lap=4 + (i % 3), kind="sudden_degradation", severity=1.1)],
            "laps": 8,
        })
        # rain builds over a long stint (simulator rain grows with lap number)
        scenarios.append({
            "profile": DriverProfile(driver_id=f"N{i}"),
            "incidents": [],
            "laps": 12,
        })
        # extreme aggression — CRITICAL wear reached quickly
        scenarios.append({
            "profile": DriverProfile(driver_id=f"X{i}", braking_aggression=1.5, tire_preservation=0.65),
            "incidents": [],
            "laps": 6,
        })
    return scenarios


def generate_calibration_dataset(
    n_races: int = 30,
    seed: int = 42,
    knowledge_path: Path = Path("data/knowledge"),
    incident_dataset_path: Path = Path("data/incidents/labeled_dataset.jsonl"),
) -> tuple[list[float], list[float]]:
    sim = SyntheticRaceSimulator(seed=seed)
    retriever = HybridMemoryRetriever()
    if knowledge_path.exists():
        retriever.add_documents(load_markdown_knowledge(knowledge_path))
    per_type = max(1, n_races // 5)
    scenarios = _build_scenarios(per_type)

    X: list[float] = []
    y: list[float] = []
    for sc in scenarios:
        samples = sim.generate_samples(
            session_id=f"calib-{sc['profile'].driver_id}",
            laps=sc["laps"],
            profile=sc["profile"],
            incidents=sc["incidents"],
        )
        for w in sim.rolling_windows(samples, size=12, step=4):
            features = extract_features(w)
            findings = [agent.analyze(w, features, retriever) for agent in _AGENTS]
            raw, _cal_features = compute_raw_score(findings)
            X.append(raw)
            incident_kind = _incident_kind_for_window(w, sc["incidents"])
            y.append(_ground_truth_label(w, features, incident_kind))

    # Augment with real incident labels derived from FastF1 historical data.
    # These are weighted ×2 (vs ×3 for human feedback) since they come from
    # a proxy feature mapping rather than direct calibration score space.
    try:
        from f1di.data.incident_dataset import load_dataset
        X_inc, y_inc = load_dataset(incident_dataset_path)
        if X_inc:
            X += X_inc * 2
            y += y_inc * 2
    except Exception:
        pass

    return X, y


def generate_feature_dataset(
    n_races: int = 30,
    seed: int = 42,
    knowledge_path: Path = Path("data/knowledge"),
) -> tuple[list[dict[str, float]], list[float]]:
    """Like generate_calibration_dataset but returns the per-window calibration feature
    dicts instead of the composite raw score.

    Each feature dict contains:
      agent_agreement, model_confidence, evidence_strength, risk_mean, risk_max

    Used by scripts/fit_weights.py to validate the hardcoded weights in compute_raw_score.
    """
    sim = SyntheticRaceSimulator(seed=seed)
    retriever = HybridMemoryRetriever()
    if knowledge_path.exists():
        retriever.add_documents(load_markdown_knowledge(knowledge_path))
    per_type = max(1, n_races // 5)
    scenarios = _build_scenarios(per_type)

    features_list: list[dict[str, float]] = []
    y: list[float] = []
    for sc in scenarios:
        samples = sim.generate_samples(
            session_id=f"fw-{sc['profile'].driver_id}",
            laps=sc["laps"],
            profile=sc["profile"],
            incidents=sc["incidents"],
        )
        for w in sim.rolling_windows(samples, size=12, step=4):
            feats = extract_features(w)
            findings = [agent.analyze(w, feats, retriever) for agent in _AGENTS]
            _, cal_features = compute_raw_score(findings)
            features_list.append(cal_features)
            incident_kind = _incident_kind_for_window(w, sc["incidents"])
            y.append(_ground_truth_label(w, feats, incident_kind))

    return features_list, y


def calibration_ece(
    calibrator: ConfidenceCalibrator,
    n_bins: int = 5,
    n_races: int = 15,
    seed: int = 999,
) -> float:
    X, y = generate_calibration_dataset(n_races=n_races, seed=seed)
    y_hat = (
        [float(calibrator._model.predict([x])[0]) for x in X]
        if calibrator._model is not None
        else list(X)
    )
    n = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, yh in enumerate(y_hat) if lo <= yh < hi]
        if not idx:
            continue
        pred_conf = sum(y_hat[j] for j in idx) / len(idx)
        true_freq = sum(y[j] for j in idx) / len(idx)
        ece += (len(idx) / n) * abs(pred_conf - true_freq)
    return round(ece, 4)


def calibration_brier(
    calibrator: ConfidenceCalibrator,
    n_races: int = 15,
    seed: int = 999,
) -> float:
    X, y = generate_calibration_dataset(n_races=n_races, seed=seed)
    y_hat = (
        [float(calibrator._model.predict([x])[0]) for x in X]
        if calibrator._model is not None
        else list(X)
    )
    return round(sum((yh - yt) ** 2 for yh, yt in zip(y_hat, y)) / len(y), 4)


def fit_and_save(
    output_path: Path | None = None,
    n_races: int = 30,
    seed: int = 42,
) -> ConfidenceCalibrator:
    from f1di.agents.classifier_utils import _CALIBRATION_DIR
    if output_path is None:
        output_path = _CALIBRATION_DIR / "isotonic.pkl"
    X, y = generate_calibration_dataset(n_races=n_races, seed=seed)
    calibrator = ConfidenceCalibrator.fit(X, y)
    calibrator.save(output_path)
    return calibrator
