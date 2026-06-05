from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from f1di.domain.schemas import Compound, TelemetrySample, TelemetryWindow


@dataclass(frozen=True)
class DriverProfile:
    driver_id: str
    braking_aggression: float = 1.0
    tire_preservation: float = 1.0
    throttle_smoothness: float = 1.0
    battery_bias: float = 1.0
    oversteer_tendency: float = 1.0


@dataclass(frozen=True)
class IncidentPlan:
    lap: int
    kind: str
    severity: float


class SyntheticRaceSimulator:
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def generate_samples(
        self,
        *,
        session_id: str,
        track_id: str = "silverstone",
        profile: DriverProfile = DriverProfile(driver_id="VER"),
        laps: int = 10,
        samples_per_lap: int = 30,
        compound: Compound = Compound.MEDIUM,
        incidents: list[IncidentPlan] | None = None,
    ) -> list[TelemetrySample]:
        incidents = incidents or []
        samples: list[TelemetrySample] = []
        tire_wear = 0.05
        soc = 0.83
        timestamp = 0
        for lap in range(1, laps + 1):
            rain = max(0.0, min(1.0, 0.03 * lap + self.rng.gauss(0, 0.015)))
            for i in range(samples_per_lap):
                phase = i / samples_per_lap
                sector = min(3, int(phase * 3) + 1)
                corner_id = f"T{1 + int(phase * 18)}"
                braking_zone = phase % 0.18 < 0.035
                base_speed = 305 - (95 if braking_zone else 0) + 28 * math.sin(phase * math.tau)
                speed = max(60, base_speed + self.rng.gauss(0, 4))
                brake = max(0, (120 if braking_zone else 8) * profile.braking_aggression + self.rng.gauss(0, 5))
                throttle = max(0, min(100, (35 if braking_zone else 82) + self.rng.gauss(0, 8 / profile.throttle_smoothness)))
                steering = 4 + 27 * math.sin(phase * math.tau * 2) + self.rng.gauss(0, 2)
                wear_rate = 0.0018 * profile.braking_aggression / profile.tire_preservation
                if braking_zone:
                    wear_rate *= 1.7
                tire_wear = min(0.99, tire_wear + wear_rate)
                soc = max(0.05, min(0.95, soc - (0.0025 * profile.battery_bias if throttle > 70 else -0.0015)))
                lockup = False
                brake_temp_add = brake * 3.7
                incident = next((x for x in incidents if x.lap == lap), None)
                if incident and incident.kind == "lockup" and braking_zone and self.rng.random() < 0.22 * incident.severity:
                    lockup = True
                    tire_wear = min(0.99, tire_wear + 0.025 * incident.severity)
                    brake_temp_add += 90 * incident.severity
                if incident and incident.kind == "sudden_degradation":
                    tire_wear = min(0.99, tire_wear + 0.0035 * incident.severity)
                track_temp = 35 + 8 * math.sin(lap / max(laps, 1) * math.pi) - rain * 7
                wind = 8 + 4 * math.sin(lap * 0.7) + self.rng.random() * 4
                grip = max(0.35, min(1.0, 0.95 - tire_wear * 0.42 - rain * 0.35))
                temp_fl = 92 + tire_wear * 35 + brake / 15 + self.rng.gauss(0, 2)
                temp_fr = 90 + tire_wear * 30 + brake / 16 + self.rng.gauss(0, 2)
                sample = TelemetrySample(
                    session_id=session_id,
                    driver_id=profile.driver_id,
                    track_id=track_id,
                    timestamp_ms=timestamp,
                    lap=lap,
                    sector=sector,
                    distance_m=5891 * (lap - 1 + phase),
                    corner_id=corner_id,
                    speed_kph=speed,
                    acceleration_g=(throttle - brake / 2) / 100,
                    throttle_pct=throttle,
                    brake_pressure_bar=brake,
                    steering_angle_deg=steering,
                    yaw_rate_deg_s=steering * speed / 180,
                    slip_angle_deg=abs(steering) / 20 * profile.oversteer_tendency,
                    wheel_speed_fl=speed * (0.98 if lockup else 1.0),
                    wheel_speed_fr=speed * (0.985 if lockup else 1.0),
                    wheel_speed_rl=speed,
                    wheel_speed_rr=speed,
                    compound=compound,
                    stint_lap=lap,
                    tire_temp_fl_c=temp_fl,
                    tire_temp_fr_c=temp_fr,
                    tire_temp_rl_c=temp_fr - 3,
                    tire_temp_rr_c=temp_fr - 4,
                    tire_wear_fl=min(0.99, tire_wear * 1.08),
                    tire_wear_fr=min(0.99, tire_wear),
                    tire_wear_rl=min(0.99, tire_wear * 0.82),
                    tire_wear_rr=min(0.99, tire_wear * 0.80),
                    grip_estimate=grip,
                    lockup_event=lockup,
                    battery_soc=soc,
                    ers_deploy_kw=120 if throttle > 75 else 20,
                    ers_regen_kw=70 if braking_zone else 5,
                    pu_thermal_state=min(1.0, 0.45 + throttle / 180),
                    track_temp_c=track_temp,
                    ambient_temp_c=24 + math.sin(lap / 3),
                    humidity_pct=58 + rain * 25,
                    wind_speed_kph=wind,
                    wind_direction_deg=(240 + lap * 7) % 360,
                    rain_intensity=rain,
                    evolving_grip=max(0.4, 0.88 - rain * 0.4 + lap * 0.002),
                    brake_temp_fl_c=420 + brake_temp_add + tire_wear * 180,
                    brake_temp_fr_c=410 + brake_temp_add + tire_wear * 170,
                    brake_temp_rl_c=360 + brake * 1.7,
                    brake_temp_rr_c=355 + brake * 1.6,
                )
                samples.append(sample)
                timestamp += int(90_000 / samples_per_lap)
        return samples

    def generate_race_samples(
        self,
        *,
        session_id: str,
        track_id: str = "silverstone",
        profiles: list[DriverProfile],
        laps: int = 10,
        samples_per_lap: int = 30,
        compound: Compound = Compound.MEDIUM,
        incidents_by_driver: dict[str, list[IncidentPlan]] | None = None,
    ) -> dict[str, list[TelemetrySample]]:
        incidents_by_driver = incidents_by_driver or {}

        TrackStep = tuple[int, float, int, str, float, float, float, float, float]
        track: list[TrackStep] = []
        for lap in range(1, laps + 1):
            rain = max(0.0, min(1.0, 0.03 * lap + self.rng.gauss(0, 0.015)))
            for i in range(samples_per_lap):
                phase = i / samples_per_lap
                sector = min(3, int(phase * 3) + 1)
                corner_id = f"T{1 + int(phase * 18)}"
                track_temp = 35 + 8 * math.sin(lap / max(laps, 1) * math.pi) - rain * 7
                wind = 8 + 4 * math.sin(lap * 0.7) + self.rng.random() * 4
                wind_dir = (240 + lap * 7) % 360
                evolving = max(0.4, 0.88 - rain * 0.4 + lap * 0.002)
                track.append((lap, phase, sector, corner_id, rain, track_temp, wind, wind_dir, evolving))

        result: dict[str, list[TelemetrySample]] = {}
        for profile in profiles:
            incidents = incidents_by_driver.get(profile.driver_id, [])
            samples: list[TelemetrySample] = []
            tire_wear = 0.05
            soc = 0.83
            timestamp = 0
            for lap, phase, sector, corner_id, rain, track_temp, wind, wind_dir, evolving in track:
                braking_zone = phase % 0.18 < 0.035
                base_speed = 305 - (95 if braking_zone else 0) + 28 * math.sin(phase * math.tau)
                speed = max(60, base_speed + self.rng.gauss(0, 4))
                brake = max(0, (120 if braking_zone else 8) * profile.braking_aggression + self.rng.gauss(0, 5))
                throttle = max(0, min(100, (35 if braking_zone else 82) + self.rng.gauss(0, 8 / profile.throttle_smoothness)))
                steering = 4 + 27 * math.sin(phase * math.tau * 2) + self.rng.gauss(0, 2)
                wear_rate = 0.0018 * profile.braking_aggression / profile.tire_preservation
                if braking_zone:
                    wear_rate *= 1.7
                tire_wear = min(0.99, tire_wear + wear_rate)
                soc = max(0.05, min(0.95, soc - (0.0025 * profile.battery_bias if throttle > 70 else -0.0015)))
                lockup = False
                brake_temp_add = brake * 3.7
                incident = next((x for x in incidents if x.lap == lap), None)
                if incident and incident.kind == "lockup" and braking_zone and self.rng.random() < 0.22 * incident.severity:
                    lockup = True
                    tire_wear = min(0.99, tire_wear + 0.025 * incident.severity)
                    brake_temp_add += 90 * incident.severity
                if incident and incident.kind == "sudden_degradation":
                    tire_wear = min(0.99, tire_wear + 0.0035 * incident.severity)
                grip = max(0.35, min(1.0, 0.95 - tire_wear * 0.42 - rain * 0.35))
                temp_fl = 92 + tire_wear * 35 + brake / 15 + self.rng.gauss(0, 2)
                temp_fr = 90 + tire_wear * 30 + brake / 16 + self.rng.gauss(0, 2)
                samples.append(TelemetrySample(
                    session_id=session_id,
                    driver_id=profile.driver_id,
                    track_id=track_id,
                    timestamp_ms=timestamp,
                    lap=lap,
                    sector=sector,
                    distance_m=5891 * (lap - 1 + phase),
                    corner_id=corner_id,
                    speed_kph=speed,
                    acceleration_g=(throttle - brake / 2) / 100,
                    throttle_pct=throttle,
                    brake_pressure_bar=brake,
                    steering_angle_deg=steering,
                    yaw_rate_deg_s=steering * speed / 180,
                    slip_angle_deg=abs(steering) / 20 * profile.oversteer_tendency,
                    wheel_speed_fl=speed * (0.98 if lockup else 1.0),
                    wheel_speed_fr=speed * (0.985 if lockup else 1.0),
                    wheel_speed_rl=speed,
                    wheel_speed_rr=speed,
                    compound=compound,
                    stint_lap=lap,
                    tire_temp_fl_c=temp_fl,
                    tire_temp_fr_c=temp_fr,
                    tire_temp_rl_c=temp_fr - 3,
                    tire_temp_rr_c=temp_fr - 4,
                    tire_wear_fl=min(0.99, tire_wear * 1.08),
                    tire_wear_fr=min(0.99, tire_wear),
                    tire_wear_rl=min(0.99, tire_wear * 0.82),
                    tire_wear_rr=min(0.99, tire_wear * 0.80),
                    grip_estimate=grip,
                    lockup_event=lockup,
                    battery_soc=soc,
                    ers_deploy_kw=120 if throttle > 75 else 20,
                    ers_regen_kw=70 if braking_zone else 5,
                    pu_thermal_state=min(1.0, 0.45 + throttle / 180),
                    track_temp_c=track_temp,
                    ambient_temp_c=24 + math.sin(lap / 3),
                    humidity_pct=58 + rain * 25,
                    wind_speed_kph=wind,
                    wind_direction_deg=wind_dir,
                    rain_intensity=rain,
                    evolving_grip=evolving,
                    brake_temp_fl_c=420 + brake_temp_add + tire_wear * 180,
                    brake_temp_fr_c=410 + brake_temp_add + tire_wear * 170,
                    brake_temp_rl_c=360 + brake * 1.7,
                    brake_temp_rr_c=355 + brake * 1.6,
                ))
                timestamp += int(90_000 / samples_per_lap)
            result[profile.driver_id] = samples

        return result

    def rolling_windows(self, samples: list[TelemetrySample], *, size: int = 12, step: int = 3) -> list[TelemetryWindow]:
        windows = []
        for idx in range(size, len(samples) + 1, step):
            chunk = samples[idx - size:idx]
            windows.append(TelemetryWindow(session_id=chunk[-1].session_id, driver_id=chunk[-1].driver_id, track_id=chunk[-1].track_id, samples=chunk))
        return windows

    def write_jsonl(self, windows: list[TelemetryWindow], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for window in windows:
                f.write(json.dumps(window.model_dump(mode="json")) + "\n")
