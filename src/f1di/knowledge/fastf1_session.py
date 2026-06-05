"""FastF1-backed session browser — race calendar, lap list, telemetry windows.

FastF1 pulls data from F1's official timing servers and caches locally, so it
works at any time including during live race weekends (unlike OpenF1's free tier).
First load per session downloads ~50-100 MB; subsequent loads are instant from cache.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow
from f1di.knowledge.track_ids import canonical as canonical_track_id

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = str(Path(__file__).parents[3] / "data" / "fastf1_cache")
_CACHE = os.environ.get("F1DI_FASTF1_CACHE", _DEFAULT_CACHE)

_WEAR_RATE: dict[str, float] = {
    "SOFT": 0.028, "MEDIUM": 0.018, "HARD": 0.011,
    "INTERMEDIATE": 0.014, "WET": 0.009,
}
_BASE_TIRE_TEMP: dict[str, float] = {
    "SOFT": 96.0, "MEDIUM": 88.0, "HARD": 80.0,
    "INTERMEDIATE": 72.0, "WET": 60.0,
}


def _ff1():
    import fastf1
    os.makedirs(_CACHE, exist_ok=True)
    fastf1.Cache.enable_cache(_CACHE)
    return fastf1


def get_races(year: int) -> list[dict]:
    ff1 = _ff1()
    schedule = ff1.get_event_schedule(year, include_testing=False)
    races = []
    for _, row in schedule.iterrows():
        races.append({
            "round": int(row["RoundNumber"]),
            "name": str(row["EventName"]),
            "circuit": str(row["Location"]),
            "country": str(row["Country"]),
            "date": str(row["EventDate"])[:10],
        })
    return races


def get_drivers(year: int, round_num: int) -> list[dict]:
    ff1 = _ff1()
    session = ff1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    codes = sorted(session.laps["Driver"].dropna().unique())
    return [{"code": str(c)} for c in codes]


def get_laps(year: int, round_num: int, driver: str) -> list[dict]:
    ff1 = _ff1()
    session = ff1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    driver_laps = session.laps.pick_drivers(driver.upper())
    result = []
    for _, lap in driver_laps.iterrows():
        lt = lap.get("LapTime")
        tl = lap.get("TyreLife")
        result.append({
            "lap_number": int(lap["LapNumber"]),
            "lap_time_s": round(lt.total_seconds(), 3) if pd.notna(lt) else None,
            "compound": str(lap.get("Compound", "UNKNOWN")),
            "tyre_life": int(tl) if pd.notna(tl) else None,
        })
    return sorted(result, key=lambda r: r["lap_number"])


def _build_lap_samples(
    lap_row: pd.Series,
    session_id: str,
    driver: str,
    track_id: str,
    track_temp: float,
    ambient_temp: float,
    humidity: float,
    rainfall: bool,
    n_samples: int,
    time_offset_ms: int,
) -> list[TelemetrySample]:
    """Build telemetry samples for a single lap."""
    compound_str = str(lap_row.get("Compound", "MEDIUM")).upper()
    compound = Compound(compound_str) if compound_str in Compound.__members__ else Compound.MEDIUM
    tl = lap_row.get("TyreLife")
    stint_lap = int(tl) if pd.notna(tl) else 0
    lap_num = int(lap_row["LapNumber"])
    wear = min(0.98, stint_lap * _WEAR_RATE.get(compound.value, 0.018))

    try:
        car_data = lap_row.get_car_data().add_distance()
    except Exception as exc:
        logger.debug("fastf1_no_car_data lap=%d err=%s", lap_num, exc)
        return []

    if car_data is None or len(car_data) == 0:
        return []

    step = max(1, len(car_data) // n_samples)
    rows = list(car_data.iloc[::step].iterrows())
    speeds = [float(r.get("Speed", 200)) for _, r in rows]
    samples = []

    for i, (_, row) in enumerate(rows):
        speed = float(row.get("Speed", 200))
        throttle = float(row.get("Throttle", 50))
        brake = bool(row.get("Brake", False))
        drs = int(row.get("DRS", 0))
        rpm = float(row.get("RPM", 10000))
        gear = int(row.get("nGear", 6))
        dist = float(row.get("Distance", 0))

        prev_speed = speeds[max(0, i - 1)]
        accel_g = (speed - prev_speed) / 3.6 / 9.81 if i > 0 else 0.0
        brake_bar = 90.0 if brake else 0.0
        soc = (
            max(0.25, 0.70 - stint_lap * 0.003)
            if (drs >= 8 or throttle > 90)
            else min(0.95, 0.65 + (100 - throttle) * 0.001)
        )
        base_tire = _BASE_TIRE_TEMP.get(compound.value, 88.0)
        tire_temp = base_tire + (track_temp - 30) * 0.5 + throttle * 0.08
        brake_temp = (350.0 + speed * 0.6) if brake else 320.0
        sample_wear = max(0.0, min(0.98, wear + (i - len(rows)) * _WEAR_RATE.get(compound.value, 0.018) * 0.05))

        samples.append(TelemetrySample(
            session_id=session_id,
            driver_id=driver.upper(),
            track_id=track_id,
            timestamp_ms=time_offset_ms + i * 3700,
            lap=lap_num,
            sector=min(3, max(1, (gear // 3) + 1)),
            distance_m=dist,
            corner_id=None,
            speed_kph=speed,
            acceleration_g=round(accel_g, 3),
            throttle_pct=throttle,
            brake_pressure_bar=brake_bar,
            steering_angle_deg=0.0,
            yaw_rate_deg_s=0.0,
            slip_angle_deg=0.0,
            wheel_speed_fl=speed, wheel_speed_fr=speed,
            wheel_speed_rl=speed, wheel_speed_rr=speed,
            compound=compound,
            stint_lap=stint_lap,
            tire_temp_fl_c=tire_temp, tire_temp_fr_c=tire_temp - 2.0,
            tire_temp_rl_c=tire_temp - 4.0, tire_temp_rr_c=tire_temp - 5.0,
            tire_wear_fl=sample_wear, tire_wear_fr=sample_wear * 0.97,
            tire_wear_rl=sample_wear * 0.92, tire_wear_rr=sample_wear * 0.90,
            grip_estimate=max(0.60, 0.95 - sample_wear * 0.35),
            lockup_event=False,
            battery_soc=round(soc, 3),
            ers_deploy_kw=60.0 if (drs >= 8 or throttle > 85) else 0.0,
            ers_regen_kw=80.0 if brake else 0.0,
            pu_thermal_state=min(0.95, 0.60 + rpm / 15000 * 0.35),
            track_temp_c=track_temp,
            ambient_temp_c=ambient_temp,
            humidity_pct=humidity,
            wind_speed_kph=0.0,
            wind_direction_deg=0.0,
            rain_intensity=0.6 if rainfall else 0.0,
            evolving_grip=0.75 if rainfall else 0.92,
            brake_temp_fl_c=brake_temp, brake_temp_fr_c=brake_temp - 10.0,
            brake_temp_rl_c=brake_temp * 0.7, brake_temp_rr_c=brake_temp * 0.68,
        ))

    return samples


def build_window(
    year: int,
    round_num: int,
    driver: str,
    lap_number: int | None = None,
    n_samples: int = 20,
    window_laps: int = 5,
) -> TelemetryWindow:
    """Build a multi-lap TelemetryWindow from FastF1 data.

    Uses `window_laps` consecutive laps ending at `lap_number` (or the fastest
    lap when omitted). A multi-lap window gives the inference agents trend
    signals (wear slope, SOC drift, brake temps) that a single lap cannot.
    """
    ff1 = _ff1()
    session = ff1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=True, weather=True, messages=False)

    location = session.event.get("Location", "unknown")
    track_id = canonical_track_id(str(location))
    session_id = f"fastf1_{year}_{round_num}"

    driver_laps = session.laps.pick_drivers(driver.upper())
    valid_laps = driver_laps[driver_laps["LapTime"].notna()]

    # Determine end lap of the window
    if lap_number is None:
        anchor = (
            valid_laps.loc[valid_laps["LapTime"].idxmin()]
            if len(valid_laps) > 0
            else driver_laps.iloc[-1]
        )
        end_lap = int(anchor["LapNumber"])
    else:
        end_lap = lap_number

    start_lap = max(1, end_lap - window_laps + 1)

    weather = session.weather_data
    track_temp = float(weather["TrackTemp"].mean()) if len(weather) > 0 else 30.0
    ambient_temp = float(weather["AirTemp"].mean()) if len(weather) > 0 else 22.0
    humidity = float(weather["Humidity"].mean()) if len(weather) > 0 else 50.0
    rainfall = bool(weather["Rainfall"].any()) if len(weather) > 0 else False

    all_samples: list[TelemetrySample] = []
    time_offset = 0

    for lap_n in range(start_lap, end_lap + 1):
        matching = driver_laps[driver_laps["LapNumber"] == lap_n]
        if len(matching) == 0:
            continue
        lap_samples = _build_lap_samples(
            lap_row=matching.iloc[0],
            session_id=session_id,
            driver=driver,
            track_id=track_id,
            track_temp=track_temp,
            ambient_temp=ambient_temp,
            humidity=humidity,
            rainfall=rainfall,
            n_samples=n_samples,
            time_offset_ms=time_offset,
        )
        all_samples.extend(lap_samples)
        time_offset += n_samples * 3700 + 90_000  # ~90 s per lap gap

    if not all_samples:
        raise ValueError(
            f"No telemetry for {driver} laps {start_lap}-{end_lap} in {year} R{round_num}"
        )

    return TelemetryWindow(
        session_id=session_id, driver_id=driver.upper(),
        track_id=track_id, samples=all_samples,
    )
