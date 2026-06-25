from __future__ import annotations

import statistics
from dataclasses import dataclass

from f1di.domain.schemas import TelemetryWindow

_CIRCUIT_AVG_SPEED: dict[str, float] = {
    "bahrain": 205.0, "jeddah": 250.0,
    "melbourne": 215.0, "albert_park": 215.0,
    "suzuka": 235.0, "shanghai": 215.0, "miami": 220.0,
    "imola": 200.0, "monaco": 140.0,
    "montreal": 210.0, "circuit_gilles_villeneuve": 210.0,
    "barcelona": 210.0, "catalonia": 210.0,
    "red_bull_ring": 215.0, "silverstone": 235.0,
    "hungaroring": 190.0, "spa": 235.0, "zandvoort": 200.0,
    "monza": 250.0, "baku": 215.0, "baku_city_circuit": 215.0,
    "marina_bay": 175.0, "singapore": 175.0, "cota": 195.0,
    "mexico_city": 195.0, "interlagos": 205.0, "las_vegas": 235.0,
    "losail": 225.0, "yas_marina": 210.0, "abu_dhabi": 210.0,
}

# 0.0 = street/temporary circuit, 1.0 = permanent track
_CIRCUIT_TYPE: dict[str, float] = {
    "monaco": 0.0, "baku": 0.0, "baku_city_circuit": 0.0,
    "jeddah": 0.0, "melbourne": 0.0, "albert_park": 0.0,
    "miami": 0.0, "montreal": 0.0, "circuit_gilles_villeneuve": 0.0,
    "marina_bay": 0.0, "singapore": 0.0, "las_vegas": 0.0,
}

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
    throttle_mean: float = 72.0
    ers_net_deploy_kw: float = 40.0
    # EMA-weighted slopes (recent samples weighted higher than older ones).
    # More sensitive to late-stint acceleration than the equal-weight slope().
    # Defaulting to 0.0 keeps existing RaceFeatures constructors valid.
    fl_wear_slope_ema: float = 0.0
    fr_wear_slope_ema: float = 0.0
    circuit_avg_speed_kph: float = 210.0
    circuit_type_enc: float = 1.0  # 0.0=street, 1.0=permanent
    race_laps_total: float = 57.0


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = statistics.fmean(values)
    denom = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
    return sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values)) / denom


def slope_ema(values: list[float], alpha: float = 0.65) -> float:
    """Weighted least-squares slope: recent samples contribute more (weight ∝ alpha^age).

    alpha=0.65 means in a 5-sample window the most recent sample has ~4x the
    influence of the oldest, giving a current reading of the acceleration rate
    rather than the average over the full window.
    """
    if len(values) < 2:
        return 0.0
    n = len(values)
    weights = [alpha ** (n - 1 - i) for i in range(n)]
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]
    x_mean = sum(w * i for w, i in zip(weights, range(n)))
    y_mean = sum(w * v for w, v in zip(weights, values))
    cov = sum(w * (i - x_mean) * (v - y_mean) for w, i, v in zip(weights, range(n), values))
    var = sum(w * (i - x_mean) ** 2 for w, i in zip(weights, range(n)))
    return cov / var if var > 1e-10 else 0.0


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
    track = (window.track_id or "").lower().replace("-", "_").replace(" ", "_")

    ers_net = [s.ers_deploy_kw - s.ers_regen_kw for s in samples]

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
        brake_temp_front_max=max(max(s.brake_temp_fl_c, s.brake_temp_fr_c) for s in samples),
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
        throttle_mean=statistics.fmean(throttle),
        ers_net_deploy_kw=statistics.fmean(ers_net),
        fl_wear_slope_ema=slope_ema(fl_wear),
        fr_wear_slope_ema=slope_ema(fr_wear),
        circuit_avg_speed_kph=_CIRCUIT_AVG_SPEED.get(track, 210.0),
        circuit_type_enc=_CIRCUIT_TYPE.get(track, 1.0),
        race_laps_total=float(total_laps),
    )
