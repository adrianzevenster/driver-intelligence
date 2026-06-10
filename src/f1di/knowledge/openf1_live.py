"""Live telemetry bridge from the OpenF1 REST API.

Fetches car data, stints, weather, and lap info for a given session + driver
and constructs a TelemetryWindow suitable for the inference engine. Fields not
provided by OpenF1 (tire temps, ERS SOC, steering angle, etc.) are estimated
from available signals using simple physical heuristics.

Works for both live sessions (omit lap_number → latest data) and historical
replay (pass lap_number → window scoped to that lap).
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow
from f1di.knowledge.track_ids import canonical as canonical_track_id

logger = logging.getLogger(__name__)

_BASE = "https://api.openf1.org/v1"
_TIMEOUT = 8.0

_WEAR_RATE: dict[str, float] = {
    "SOFT": 0.028,
    "MEDIUM": 0.018,
    "HARD": 0.011,
    "INTERMEDIATE": 0.014,
    "WET": 0.009,
}
_BASE_TIRE_TEMP: dict[str, float] = {
    "SOFT": 96.0, "MEDIUM": 88.0, "HARD": 80.0,
    "INTERMEDIATE": 72.0, "WET": 60.0,
}


class OpenF1Blocked(RuntimeError):
    """OpenF1 is restricting access during a live session."""


def _get(path: str, **params: Any) -> list[dict]:
    url = f"{_BASE}/{path}"
    try:
        r = httpx.get(url, params=params, timeout=_TIMEOUT)
        if r.status_code in (401, 403) or (
            r.status_code == 422 and "Live F1 session" in r.text
        ):
            raise OpenF1Blocked(r.json().get("detail", "OpenF1 access restricted during live session"))
        r.raise_for_status()
        return r.json()
    except OpenF1Blocked:
        raise
    except Exception as exc:
        logger.warning("openf1_live_fetch_failed", extra={"path": path, "error": str(exc)})
        return []


def get_sessions(*, year: int = 2024, session_type: str = "Race") -> list[dict]:
    """Return sessions for a year ordered newest first."""
    rows = _get("sessions", year=year, session_type=session_type)
    return sorted(rows, key=lambda r: r.get("date_start", ""), reverse=True)


def get_drivers(*, session_key: int) -> list[dict]:
    """Return driver list for a session."""
    return _get("drivers", session_key=session_key)


def get_laps(*, session_key: int, driver_number: int) -> list[dict]:
    """Return completed laps ordered by lap number."""
    rows = _get("laps", session_key=session_key, driver_number=driver_number)
    return sorted(rows, key=lambda r: r.get("lap_number", 0))


def build_window(
    *,
    session_key: int,
    driver_number: int,
    n_samples: int = 12,
    lap_number: int | None = None,
) -> TelemetryWindow:
    """Construct a TelemetryWindow from OpenF1 data.

    When lap_number is None the window covers the most recent n_samples
    car-data points (live mode). When lap_number is specified the window
    is scoped to that lap's time range (replay mode).
    """
    # ── Fetch raw data ────────────────────────────────────────────────────────
    car_rows_all = sorted(
        _get("car_data", session_key=session_key, driver_number=driver_number),
        key=lambda r: r.get("date", ""),
    )
    stint_rows = _get("stints", session_key=session_key, driver_number=driver_number)
    weather_rows = _get("weather", session_key=session_key)
    lap_rows = sorted(
        _get("laps", session_key=session_key, driver_number=driver_number),
        key=lambda r: r.get("lap_number", 0),
    )
    session_rows = _get("sessions", session_key=session_key)
    session_info = session_rows[0] if session_rows else {}

    # ── Select car-data slice ─────────────────────────────────────────────────
    if lap_number is not None:
        target_lap = next((r for r in lap_rows if r.get("lap_number") == lap_number), None)
        next_lap = next((r for r in lap_rows if r.get("lap_number") == lap_number + 1), None)
        if target_lap and target_lap.get("date_start"):
            start = target_lap["date_start"]
            end = next_lap.get("date_start") if next_lap else None
            car_rows = [r for r in car_rows_all if r.get("date", "") >= start]
            if end:
                car_rows = [r for r in car_rows if r.get("date", "") < end]
        else:
            car_rows = car_rows_all[-n_samples:]
        current_lap = lap_number
    else:
        car_rows = car_rows_all[-n_samples:]
        latest_lap = max(lap_rows, key=lambda r: r.get("lap_number", 0), default={})
        current_lap = int(latest_lap.get("lap_number", 1))

    # ── Derive context ────────────────────────────────────────────────────────
    location = session_info.get("location", "unknown")
    track_id = canonical_track_id(location)
    session_id = f"openf1_{session_key}"
    driver_id = str(driver_number)

    # Stint active at the target lap
    lap_stints = [s for s in stint_rows if s.get("lap_start", 1) <= current_lap]
    active_stint = max(lap_stints, key=lambda s: s.get("stint_number", 0), default={})
    raw_compound = (active_stint.get("compound") or "MEDIUM").upper()
    compound = Compound(raw_compound) if raw_compound in Compound.__members__ else Compound.MEDIUM
    stint_lap = max(0, current_lap - int(active_stint.get("lap_start", 1)))

    # Weather closest to the target lap
    latest_weather = max(weather_rows, key=lambda w: w.get("date", ""), default={})
    track_temp = float(latest_weather.get("track_temperature", 30.0))
    ambient_temp = float(latest_weather.get("air_temperature", 22.0))
    humidity = float(latest_weather.get("humidity", 50.0))
    wind_speed_ms = float(latest_weather.get("wind_speed", 0.0))
    wind_dir = float(latest_weather.get("wind_direction", 0.0))
    rainfall = bool(latest_weather.get("rainfall", False))

    wear = min(0.98, stint_lap * _WEAR_RATE.get(compound.value, 0.018))

    # Rain intensity: continuous scale — rainfall flag + wind amplifies spray severity
    if rainfall:
        rain_intensity = min(0.95, 0.60 + max(0.0, wind_speed_ms - 3.0) * 0.025)
    elif humidity > 85:
        rain_intensity = 0.10  # damp/wet track without active rain
    else:
        rain_intensity = 0.0
    evolving_grip = 0.72 if rainfall else (0.87 if humidity > 85 else 0.92)

    # ── Build samples ─────────────────────────────────────────────────────────
    samples: list[TelemetrySample] = []
    speeds = [float(r.get("speed", 200)) for r in car_rows]

    for i, row in enumerate(car_rows):
        speed = float(row.get("speed", 200))
        throttle = float(row.get("throttle", 50))
        brake = int(row.get("brake", 0))
        drs = int(row.get("drs", 0))
        rpm = float(row.get("rpm", 10000))
        gear = int(row.get("n_gear", 6))

        braking = brake > 0
        prev_speed = speeds[max(0, i - 1)]
        accel_g = (speed - prev_speed) / 3.6 / 9.81 if i > 0 else 0.0
        brake_bar = 90.0 if braking else 0.0

        # Lockup: sudden speed drop >12% while braking hard at speed
        speed_drop_ratio = (prev_speed - speed) / max(prev_speed, 1.0) if i > 0 else 0.0
        lockup = braking and speed > 150 and speed_drop_ratio > 0.12

        # Wheel speeds: front wheels carry more braking load → slow faster under braking
        if braking and speed > 80:
            decel_factor = min(0.20, max(0.0, -accel_g) * 0.08)
            wfl = speed * (1.0 - decel_factor)
            wfr = speed * (1.0 - decel_factor * 0.97)
            wrl = speed * (1.0 - decel_factor * 0.30)
            wrr = speed * (1.0 - decel_factor * 0.28)
        else:
            wfl = wfr = wrl = wrr = speed

        if drs == 8 or throttle > 90:
            soc = max(0.25, 0.70 - stint_lap * 0.003)
        else:
            soc = min(0.95, 0.65 + (100 - throttle) * 0.001)
        ers_deploy = 60.0 if (drs == 8 or throttle > 85) else 0.0
        ers_regen = 80.0 if braking else 0.0

        base_tire = _BASE_TIRE_TEMP.get(compound.value, 88.0)
        tire_temp = base_tire + (track_temp - 30) * 0.5 + throttle * 0.08
        if rainfall:
            tire_temp = min(tire_temp, 70.0)  # wet conditions suppress tire temps
        brake_temp = (350.0 + speed * 0.6) if braking else 320.0
        sample_wear = min(0.98, wear + (i - len(car_rows)) * _WEAR_RATE.get(compound.value, 0.018) * 0.05)

        # Sector heuristic: S1 = low-speed entry (gear ≤ 3), S2 = mid (4-5), S3 = high-speed
        sector = 1 if gear <= 3 else (3 if gear >= 7 else 2)

        samples.append(TelemetrySample(
            session_id=session_id,
            driver_id=driver_id,
            track_id=track_id,
            timestamp_ms=i * 3700,
            lap=current_lap,
            sector=sector,
            distance_m=float(current_lap * 5000 + i * 200),
            corner_id=None,
            speed_kph=speed,
            acceleration_g=round(accel_g, 3),
            throttle_pct=throttle,
            brake_pressure_bar=brake_bar,
            steering_angle_deg=0.0,
            yaw_rate_deg_s=0.0,
            slip_angle_deg=0.0,
            wheel_speed_fl=round(wfl, 1),
            wheel_speed_fr=round(wfr, 1),
            wheel_speed_rl=round(wrl, 1),
            wheel_speed_rr=round(wrr, 1),
            compound=compound,
            stint_lap=stint_lap,
            tire_temp_fl_c=tire_temp,
            tire_temp_fr_c=tire_temp - 2.0,
            tire_temp_rl_c=tire_temp - 4.0,
            tire_temp_rr_c=tire_temp - 5.0,
            tire_wear_fl=sample_wear,
            tire_wear_fr=sample_wear * 0.97,
            tire_wear_rl=sample_wear * 0.92,
            tire_wear_rr=sample_wear * 0.90,
            grip_estimate=max(0.55, 0.95 - sample_wear * 0.35 - rain_intensity * 0.15),
            lockup_event=lockup,
            battery_soc=round(soc, 3),
            ers_deploy_kw=ers_deploy,
            ers_regen_kw=ers_regen,
            pu_thermal_state=min(0.95, 0.60 + rpm / 15000 * 0.35),
            track_temp_c=track_temp,
            ambient_temp_c=ambient_temp,
            humidity_pct=humidity,
            wind_speed_kph=wind_speed_ms * 3.6,
            wind_direction_deg=wind_dir,
            rain_intensity=rain_intensity,
            evolving_grip=evolving_grip,
            brake_temp_fl_c=brake_temp,
            brake_temp_fr_c=brake_temp - 10.0,
            brake_temp_rl_c=brake_temp * 0.7,
            brake_temp_rr_c=brake_temp * 0.68,
        ))

    if not samples:
        raise ValueError(f"No car data found for session {session_key} driver {driver_number} lap {lap_number}")

    return TelemetryWindow(
        session_id=session_id,
        driver_id=driver_id,
        track_id=track_id,
        samples=samples,
    )
