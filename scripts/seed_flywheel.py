#!/usr/bin/env python
"""Seed the flywheel with labeled race scenarios.

Fires realistic telemetry windows at the insights API, then writes
FeedbackRecord rows based on ground-truth risk expectation for each
scenario. Calling this once gives classifiers their first real-label
blend and lets the meta-learner activate at 20 labels.

Usage:
    # Inside the container (or against a running API):
    python scripts/seed_flywheel.py
    python scripts/seed_flywheel.py --api-url http://localhost:8080 --api-key KEY
    python scripts/seed_flywheel.py --dry-run  # no DB writes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"


@dataclass
class Scenario:
    name: str
    expected_risk: str          # INFO / WATCH / WARNING / CRITICAL
    session_id: str
    driver_id: str
    track_id: str
    race_total_laps: int
    lap: int
    compound: str
    stint_lap: int
    tire_wear: float            # 0–1 uniform for simplicity
    grip_estimate: float
    lockup_event: bool
    battery_soc: float
    ers_deploy_kw: float
    ers_regen_kw: float
    pu_thermal_state: float
    rain_intensity: float
    track_temp_c: float
    brake_temp_front: float
    throttle_pct: float
    speed_kph: float


def _sample(s: Scenario) -> dict:
    return {
        "session_id": s.session_id,
        "driver_id": s.driver_id,
        "track_id": s.track_id,
        "timestamp_ms": s.lap * 90_000,
        "lap": s.lap,
        "sector": 2,
        "distance_m": 2_500.0,
        "speed_kph": s.speed_kph,
        "acceleration_g": 0.4,
        "throttle_pct": s.throttle_pct,
        "brake_pressure_bar": 2.0 if s.lockup_event else 0.5,
        "steering_angle_deg": 8.0,
        "yaw_rate_deg_s": 5.0,
        "slip_angle_deg": 2.0 if s.lockup_event else 0.5,
        "wheel_speed_fl": s.speed_kph,
        "wheel_speed_fr": s.speed_kph,
        "wheel_speed_rl": s.speed_kph * (0.88 if s.lockup_event else 1.0),
        "wheel_speed_rr": s.speed_kph * (0.88 if s.lockup_event else 1.0),
        "compound": s.compound,
        "stint_lap": s.stint_lap,
        "tire_temp_fl_c": 98.0 + s.tire_wear * 30,
        "tire_temp_fr_c": 95.0 + s.tire_wear * 28,
        "tire_temp_rl_c": 88.0 + s.tire_wear * 20,
        "tire_temp_rr_c": 85.0 + s.tire_wear * 18,
        "tire_wear_fl": s.tire_wear,
        "tire_wear_fr": s.tire_wear * 0.95,
        "tire_wear_rl": s.tire_wear * 0.80,
        "tire_wear_rr": s.tire_wear * 0.78,
        "grip_estimate": s.grip_estimate,
        "lockup_event": s.lockup_event,
        "battery_soc": s.battery_soc,
        "ers_deploy_kw": s.ers_deploy_kw,
        "ers_regen_kw": s.ers_regen_kw,
        "pu_thermal_state": s.pu_thermal_state,
        "track_temp_c": s.track_temp_c,
        "ambient_temp_c": 20.0,
        "humidity_pct": 40.0 + s.rain_intensity * 50,
        "wind_speed_kph": 8.0,
        "wind_direction_deg": 180.0,
        "rain_intensity": s.rain_intensity,
        "evolving_grip": max(0.30, s.grip_estimate - 0.10),
        "brake_temp_fl_c": s.brake_temp_front,
        "brake_temp_fr_c": s.brake_temp_front + 15,
        "brake_temp_rl_c": s.brake_temp_front * 0.65,
        "brake_temp_rr_c": s.brake_temp_front * 0.62,
    }


def _window(s: Scenario, n_samples: int = 3) -> dict:
    return {
        "session_id": s.session_id,
        "driver_id": s.driver_id,
        "track_id": s.track_id,
        "race_total_laps": s.race_total_laps,
        "samples": [_sample(s) for _ in range(n_samples)],
    }


_SCENARIOS: list[Scenario] = [
    # ── INFO — nominal conditions ────────────────────────────────────────────
    Scenario("Nominal start", "INFO", "seed_2024_1", "VER", "bahrain",
             57, 5, "MEDIUM", 5, 0.12, 0.92, False, 0.90, 110, 75, 0.55, 0.00, 32, 280, 75, 290),
    Scenario("Mid-race cruise", "INFO", "seed_2024_1", "HAM", "silverstone",
             52, 20, "HARD", 12, 0.20, 0.88, False, 0.80, 100, 70, 0.60, 0.00, 36, 310, 72, 295),
    Scenario("Cool conditions nominal", "INFO", "seed_2024_1", "NOR", "monza",
             53, 8, "MEDIUM", 8, 0.15, 0.91, False, 0.88, 115, 80, 0.52, 0.00, 28, 260, 73, 310),
    Scenario("Late race pack fuel", "INFO", "seed_2024_1", "SAI", "spain",
             66, 45, "HARD", 10, 0.35, 0.84, False, 0.65, 90, 60, 0.63, 0.00, 40, 320, 68, 280),
    Scenario("Low wear fresh set", "INFO", "seed_2024_2", "LEC", "monaco",
             78, 12, "SOFT", 4, 0.08, 0.95, False, 0.95, 80, 55, 0.48, 0.00, 30, 240, 65, 180),

    # ── WATCH — elevated but manageable ─────────────────────────────────────
    Scenario("Soft tyre heat build", "WATCH", "seed_2024_2", "PER", "bahrain",
             57, 18, "SOFT", 18, 0.48, 0.79, False, 0.75, 120, 82, 0.68, 0.00, 44, 380, 78, 275),
    Scenario("Light drizzle early", "WATCH", "seed_2024_2", "RUS", "silverstone",
             52, 14, "MEDIUM", 14, 0.28, 0.76, False, 0.72, 105, 72, 0.64, 0.22, 25, 290, 71, 270),
    Scenario("Battery conservation", "WATCH", "seed_2024_2", "ALO", "singapore",
             62, 30, "HARD", 20, 0.40, 0.82, False, 0.42, 75, 55, 0.71, 0.00, 38, 350, 74, 260),
    Scenario("Moderate brake heat", "WATCH", "seed_2024_3", "STR", "canada",
             70, 25, "MEDIUM", 25, 0.38, 0.80, False, 0.70, 100, 68, 0.66, 0.05, 35, 440, 72, 280),
    Scenario("Mid stint soft edge", "WATCH", "seed_2024_3", "GAS", "austria",
             71, 22, "SOFT", 16, 0.52, 0.77, False, 0.78, 118, 79, 0.60, 0.00, 41, 390, 76, 268),
    Scenario("Rain approaching crossover", "WATCH", "seed_2024_3", "OCO", "spa",
             44, 10, "MEDIUM", 10, 0.22, 0.73, False, 0.82, 108, 74, 0.57, 0.30, 22, 300, 70, 280),
    Scenario("High fuel load throttle", "WATCH", "seed_2024_3", "HAD", "mexico",
             71, 3, "HARD", 3, 0.10, 0.87, False, 0.98, 120, 82, 0.50, 0.00, 36, 270, 82, 260),

    # ── WARNING — action required ────────────────────────────────────────────
    Scenario("Critical tyre degradation", "WARNING", "seed_2024_4", "VER", "hungary",
             70, 35, "SOFT", 32, 0.74, 0.60, True, 0.68, 125, 85, 0.72, 0.00, 48, 520, 86, 250),
    Scenario("Heavy rain intervention", "WARNING", "seed_2024_4", "HAM", "spa",
             44, 22, "INTERMEDIATE", 8, 0.30, 0.52, True, 0.60, 80, 55, 0.65, 0.68, 18, 340, 82, 200),
    Scenario("Battery overtemp risk", "WARNING", "seed_2024_4", "NOR", "abu_dhabi",
             58, 40, "MEDIUM", 28, 0.55, 0.71, False, 0.28, 140, 85, 0.88, 0.00, 42, 380, 74, 270),
    Scenario("Lockup with worn tyre", "WARNING", "seed_2024_4", "LEC", "monza",
             53, 42, "SOFT", 38, 0.82, 0.55, True, 0.72, 130, 87, 0.69, 0.00, 50, 560, 88, 255),
    Scenario("SC period high risk", "WARNING", "seed_2024_5", "SAI", "hungary",
             70, 28, "MEDIUM", 15, 0.45, 0.49, False, 0.80, 50, 35, 0.58, 0.55, 20, 280, 58, 120),
    Scenario("Fuel burn over plan", "WARNING", "seed_2024_5", "PER", "mexico",
             71, 55, "HARD", 42, 0.62, 0.68, False, 0.55, 128, 84, 0.70, 0.00, 45, 360, 87, 265),
    Scenario("Wet track SC", "WARNING", "seed_2024_5", "RUS", "canada",
             70, 18, "INTERMEDIATE", 5, 0.18, 0.44, True, 0.65, 60, 40, 0.62, 0.72, 17, 295, 70, 130),
    Scenario("Thermal overload ERS", "WARNING", "seed_2024_5", "ALO", "bahrain",
             57, 50, "HARD", 38, 0.68, 0.67, False, 0.32, 145, 88, 0.91, 0.00, 43, 400, 77, 260),

    # ── CRITICAL — immediate ─────────────────────────────────────────────────
    Scenario("Imminent tyre failure", "CRITICAL", "seed_2024_6", "VER", "silverstone",
             52, 38, "SOFT", 36, 0.92, 0.42, True, 0.60, 125, 84, 0.74, 0.00, 52, 640, 90, 240),
    Scenario("Full wet SC deploy", "CRITICAL", "seed_2024_6", "HAM", "spa",
             44, 15, "WET", 3, 0.15, 0.35, True, 0.70, 45, 28, 0.60, 0.88, 14, 260, 60, 90),
    Scenario("Battery critical SOC", "CRITICAL", "seed_2024_6", "NOR", "singapore",
             62, 55, "HARD", 40, 0.72, 0.66, False, 0.12, 155, 90, 0.95, 0.00, 44, 410, 75, 255),
    Scenario("Multi-incident SC", "CRITICAL", "seed_2024_6", "LEC", "monaco",
             78, 35, "SOFT", 22, 0.60, 0.33, True, 0.55, 35, 22, 0.68, 0.62, 18, 290, 55, 75),
    Scenario("Extreme deg lockup rain", "CRITICAL", "seed_2024_6", "SAI", "brazil",
             71, 45, "INTERMEDIATE", 18, 0.78, 0.38, True, 0.48, 55, 35, 0.78, 0.78, 16, 310, 82, 140),
]

_RISK_ORDER = ["INFO", "WATCH", "WARNING", "CRITICAL"]
_RISK_COLOR = {
    "INFO": "\033[94m", "WATCH": _YELLOW,
    "WARNING": "\033[38;5;208m", "CRITICAL": _RED,
}


def _rc(risk: str) -> str:
    return _RISK_COLOR.get(risk, _RESET)


def _label_correct(expected: str, got: str) -> bool | None:
    """Decide whether to label the model's answer as correct.

    Returns None to skip (no label written for ambiguous cases).
    """
    ei = _RISK_ORDER.index(expected)
    gi = _RISK_ORDER.index(got)
    diff = gi - ei

    if diff == 0:
        return True        # exact match → correct
    if diff == 1:
        return True        # one level high → acceptable (conservative is fine)
    if diff == -1:
        return None        # one level low → ambiguous, skip
    if diff <= -2:
        return False       # model missed a major risk → incorrect
    # diff >= 2: over-alarmed by 2+
    return False


def _post_insight(api_url: str, api_key: str, window: dict) -> dict | None:
    body = json.dumps(window).encode()
    req = urllib.request.Request(
        f"{api_url}/v1/insights",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"    {_RED}HTTP {e.code}: {e.read()[:200]}{_RESET}")
        return None
    except Exception as exc:
        print(f"    {_RED}Error: {exc}{_RESET}")
        return None


def _write_feedback(insight_id: str, correct: bool, comment: str) -> None:
    from f1di.storage.database import db_session
    from f1di.storage.models import FeedbackRecord
    with db_session() as session:
        session.add(FeedbackRecord(
            insight_id=insight_id,
            rating=5 if correct else 1,
            correct=correct,
            comment=comment,
            submitted_by="seed_flywheel",
        ))
        session.commit()


def main(api_url: str, api_key: str, dry_run: bool) -> None:
    n_correct = n_incorrect = n_skip = n_fail = 0

    print(f"\n{_BOLD}Flywheel seed — {len(_SCENARIOS)} scenarios{_RESET}")
    if dry_run:
        print(f"{_YELLOW}DRY RUN — no DB writes{_RESET}")
    print()

    for i, sc in enumerate(_SCENARIOS, 1):
        print(f"  [{i:2d}/{len(_SCENARIOS)}] {_BOLD}{sc.name:<34}{_RESET} expected={_rc(sc.expected_risk)}{sc.expected_risk}{_RESET}", end="  ", flush=True)

        result = _post_insight(api_url, api_key, _window(sc))
        if not result:
            print(f"{_RED}FAIL{_RESET}")
            n_fail += 1
            continue

        got_risk  = result.get("risk", "INFO")
        insight_id = result.get("insight_id", "")
        confidence = result.get("confidence", 0)

        label = _label_correct(sc.expected_risk, got_risk)
        if label is True:
            tag = f"{_GREEN}✓ {got_risk:<8}{_RESET}"
        elif label is False:
            tag = f"{_RED}✗ {got_risk:<8}{_RESET}"
        else:
            tag = f"{_DIM}~ {got_risk:<8}{_RESET}"

        print(f"got={tag} conf={confidence:.0%}", end="  ")

        if label is None:
            print(f"{_DIM}skip{_RESET}")
            n_skip += 1
        elif dry_run:
            print(f"{_DIM}(dry){_RESET}")
            if label:
                n_correct += 1
            else:
                n_incorrect += 1
        else:
            comment = f"seed_flywheel expected={sc.expected_risk} got={got_risk} scenario={sc.name}"
            _write_feedback(insight_id, label, comment)
            if label:
                n_correct += 1
                print(f"{_GREEN}labeled correct{_RESET}")
            else:
                n_incorrect += 1
                print(f"{_RED}labeled incorrect{_RESET}")

        time.sleep(0.3)  # don't hammer the API

    total = n_correct + n_incorrect
    print(f"\n{_BOLD}Summary{_RESET}")
    print(f"  {_GREEN}Correct:   {n_correct}{_RESET}")
    print(f"  {_RED}Incorrect: {n_incorrect}{_RESET}")
    print(f"  {_DIM}Skipped:   {n_skip}  Failed: {n_fail}{_RESET}")
    print(f"  {_BOLD}Total labeled: {total}{_RESET}")

    if total >= 20 and not dry_run:
        print(f"\n  {_GREEN}≥20 labels — triggering auto-retrain…{_RESET}")
        try:
            from f1di.agents.auto_retrain import maybe_retrain_all
            maybe_retrain_all()
            print(f"  {_GREEN}Auto-retrain check complete.{_RESET}")
        except Exception as exc:
            print(f"  {_YELLOW}Auto-retrain skipped: {exc}{_RESET}")
    elif total < 20 and not dry_run:
        print(f"\n  {_YELLOW}Only {total} labels written — need {20 - total} more to activate meta-learner.{_RESET}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed flywheel with labeled race scenarios")
    parser.add_argument("--api-url",  default="http://localhost:8080", help="API base URL")
    parser.add_argument("--api-key",  default="", help="X-API-Key header value")
    parser.add_argument("--dry-run",  action="store_true", help="Don't write feedback records")
    args = parser.parse_args()

    main(args.api_url, args.api_key, args.dry_run)
