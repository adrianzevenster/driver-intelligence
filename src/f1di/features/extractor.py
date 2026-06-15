from __future__ import annotations

import statistics
from dataclasses import dataclass

from f1di.domain.schemas import TelemetryWindow

# Per-compound typical stint length (laps) — used for stint_fraction.
_TYPICAL_STINT_LAPS: dict[str, float] = {
    "SOFT": 18.0,
    "MEDIUM": 26.0,
    "HARD": 35.0,
    "INTERMEDIATE": 22.0,
    "WET": 15.0,
}


@dataclass(frozen=True)
class RaceFeatures:
    lap: int
    sector: int
    mean_speed_kph: float
    speed_delta_kph: float
    fl_wear: float
    fr_wear: float
    rear_wear_mean: float
    fl_wear_slope: float
    fr_wear_slope: float
    rear_wear_slope: float
    axle_imbalance_fl_rl: float
    brake_temp_front_max: float
    brake_fade_risk: float
    fl_degradation_pressure: float
    battery_soc: float
    battery_soc_slope: float
    rain_intensity: float
    crosswind_proxy: float
    grip_estimate: float
    lockup_count: int
    throttle_smoothness: float
    # Race-phase context — default to mid-race neutral values so existing
    # callers that construct RaceFeatures directly remain valid.
    laps_remaining: float = 20.0
    stint_fraction: float = 0.5
    race_phase: float = 0.5   # 0.0 = first lap, 1.0 = final lap


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = statistics.fmean(values)
    denom = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
    return sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values)) / denom


def extract_features(window: TelemetryWindow) -> RaceFeatures:
    samples = window.samples
    latest = window.latest
    speeds = [s.speed_kph for s in samples]
    socs = [s.battery_soc for s in samples]
    fl_wear = [s.tire_wear_fl for s in samples]
    fr_wear = [s.tire_wear_fr for s in samples]
    rear_wear = [(s.tire_wear_rl + s.tire_wear_rr) / 2 for s in samples]
    brake_temp_front = [max(s.brake_temp_fl_c, s.brake_temp_fr_c) for s in samples]
    throttle = [s.throttle_pct for s in samples]
    throttle_delta = [abs(b - a) for a, b in zip(throttle, throttle[1:])]

    total_laps = max(1, window.race_total_laps)
    compound = latest.compound.value
    typical_stint = _TYPICAL_STINT_LAPS.get(compound, 24.0)

    return RaceFeatures(
        lap=latest.lap,
        sector=latest.sector,
        mean_speed_kph=statistics.fmean(speeds),
        speed_delta_kph=speeds[-1] - speeds[0] if len(speeds) > 1 else 0.0,
        fl_wear=latest.tire_wear_fl,
        fr_wear=latest.tire_wear_fr,
        rear_wear_mean=(latest.tire_wear_rl + latest.tire_wear_rr) / 2,
        fl_wear_slope=slope(fl_wear),
        fr_wear_slope=slope(fr_wear),
        rear_wear_slope=slope(rear_wear),
        axle_imbalance_fl_rl=latest.tire_wear_fl - latest.tire_wear_rl,
        brake_temp_front_max=max(latest.brake_temp_fl_c, latest.brake_temp_fr_c),
        brake_fade_risk=max(0.0, slope(brake_temp_front)),
        fl_degradation_pressure=(latest.tire_wear_fl * 0.65) + max(latest.tire_temp_fl_c - 105, 0) / 90,
        battery_soc=latest.battery_soc,
        battery_soc_slope=slope(socs),
        rain_intensity=latest.rain_intensity,
        crosswind_proxy=latest.wind_speed_kph * abs(latest.steering_angle_deg) / 100,
        grip_estimate=latest.grip_estimate,
        lockup_count=sum(1 for s in samples if s.lockup_event),
        throttle_smoothness=1.0 / (1.0 + (statistics.fmean(throttle_delta) if throttle_delta else 0.0)),
        laps_remaining=max(0.0, float(total_laps - latest.lap)),
        stint_fraction=min(1.0, latest.stint_lap / typical_stint),
        race_phase=min(1.0, latest.lap / total_laps),
    )
