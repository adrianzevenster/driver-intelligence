from __future__ import annotations

from pathlib import Path

from f1di.agents.battery import BatteryAgent
from f1di.agents.telemetry import TelemetryAnalysisAgent
from f1di.agents.tire import TireStrategyAgent
from f1di.agents.weather import WeatherAgent
from f1di.agents.thresholds import get as get_thresholds
from f1di.confidence.calibration import ConfidenceCalibrator, compute_raw_score
from f1di.domain.schemas import TelemetryWindow
from f1di.features.extractor import RaceFeatures, extract_features
from f1di.rag.store import HybridMemoryRetriever, load_markdown_knowledge
from f1di.simulator.generator import DriverProfile, IncidentPlan, SyntheticRaceSimulator

_AGENTS = [TelemetryAnalysisAgent(), TireStrategyAgent(), WeatherAgent(), BatteryAgent()]


def _ground_truth_label(window: TelemetryWindow, features: RaceFeatures) -> float:
    t = get_thresholds(window.track_id)
    latest = window.latest
    max_wear = max(
        latest.tire_wear_fl,
        latest.tire_wear_fr,
        (latest.tire_wear_rl + latest.tire_wear_rr) / 2,
    )

    if any(s.lockup_event for s in window.samples) or latest.brake_temp_fl_c > t.brake_temp_critical_c:
        return 0.90
    if max_wear > t.wear_critical and latest.grip_estimate < 0.62:
        return 0.85
    if max_wear > t.wear_warning:
        return 0.70
    if latest.rain_intensity >= t.rain_warning:
        return 0.65

    lap_span = latest.lap - window.samples[0].lap
    spl = len(window.samples) / lap_span if lap_span > 0 else 1.0
    projected_fl_4laps = features.fl_wear + features.fl_wear_slope * spl * 4
    projected_fr_4laps = features.fr_wear + features.fr_wear_slope * spl * 4
    if max(projected_fl_4laps, projected_fr_4laps) > t.wear_critical * 0.97 and features.fl_wear_slope > 0.0 and max_wear > t.wear_warning * 0.85:
        return 0.65

    if latest.battery_soc < t.battery_soc_warning and features.battery_soc_slope < -0.01:
        return 0.55

    if max_wear > t.wear_warning * 0.73:
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
            y.append(_ground_truth_label(w, features))

    return X, y


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
    output_path: Path = Path("data/calibration/isotonic.pkl"),
    n_races: int = 30,
    seed: int = 42,
) -> ConfidenceCalibrator:
    X, y = generate_calibration_dataset(n_races=n_races, seed=seed)
    calibrator = ConfidenceCalibrator.fit(X, y)
    calibrator.save(output_path)
    return calibrator
