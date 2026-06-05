from __future__ import annotations

import argparse
from pathlib import Path

from f1di.simulator.generator import DriverProfile, IncidentPlan, SyntheticRaceSimulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/scenarios/synthetic_race.jsonl")
    parser.add_argument("--laps", type=int, default=12)
    args = parser.parse_args()
    sim = SyntheticRaceSimulator(seed=42)
    samples = sim.generate_samples(
        session_id="synthetic-silverstone-001",
        profile=DriverProfile(driver_id="DRV-01", braking_aggression=1.12, tire_preservation=0.92),
        laps=args.laps,
        incidents=[IncidentPlan(lap=6, kind="lockup", severity=1.0), IncidentPlan(lap=8, kind="sudden_degradation", severity=0.8)],
    )
    windows = sim.rolling_windows(samples)
    sim.write_jsonl(windows, Path(args.out))
    print(f"wrote {len(windows)} windows to {args.out}")


if __name__ == "__main__":
    main()
