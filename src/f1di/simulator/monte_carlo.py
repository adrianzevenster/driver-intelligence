from __future__ import annotations

import logging
import time

import numpy as np

from f1di.domain.schemas import (
    PredictionPoint,
    RaceProjection,
    StrategyComparison,
    StrategyScenario,
    TelemetryWindow,
)
from f1di.features.extractor import RaceFeatures

logger = logging.getLogger("f1di.simulator.monte_carlo")

_RACE_LAPS: dict[str, int] = {
    "monaco": 78,
    "mexico_city": 71,
    "interlagos": 71,
    "spielberg": 71,
    "budapest": 70,
    "montreal": 70,
    "barcelona": 66,
    "imola": 63,
    "miami": 57,
    "lusail": 57,
    "bahrain": 57,
    "las_vegas": 50,
    "jeddah": 50,
    "abu_dhabi": 58,
    "suzuka": 53,
    "monza": 53,
    "silverstone": 52,
    "spa": 44,
    "zandvoort": 72,
    "singapore": 62,
    "baku": 51,
    "austin": 56,
    "melbourne": 58,
    "shanghai": 56,
}

# Approximate race-pace average lap times per circuit (seconds)
_BASE_LAP_TIME_S: dict[str, float] = {
    "monaco": 75.0,
    "singapore": 102.0,
    "spa": 107.0,
    "baku": 104.0,
    "austin": 96.0,
    "las_vegas": 95.0,
    "shanghai": 95.0,
    "miami": 91.0,
    "bahrain": 93.0,
    "jeddah": 90.0,
    "suzuka": 90.0,
    "silverstone": 89.0,
    "abu_dhabi": 87.0,
    "lusail": 85.0,
    "monza": 82.0,
    "mexico_city": 81.0,
    "budapest": 81.0,
    "barcelona": 80.0,
    "imola": 80.0,
    "melbourne": 79.0,
    "zandvoort": 74.0,
    "montreal": 74.0,
    "interlagos": 71.0,
    "spielberg": 68.0,
}
_DEFAULT_LAP_TIME_S = 90.0
_PIT_LOSS_S = 22.0       # time lost to slow pit lane vs. racing line
_FRESH_WEAR_INITIAL = 0.01
_FRESH_WEAR_SLOPE = 0.0018


class MonteCarloSimulator:
    def __init__(self, iterations: int = 500) -> None:
        self.iterations = iterations

    def _simulate_stint(
        self,
        start_fl: float,
        start_fr: float,
        slope_fl: float,
        slope_fr: float,
        n_laps: int,
        base_lap_time: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (wear_fl, wear_fr, lap_times) each shaped (iterations, n_laps)."""
        if n_laps <= 0:
            empty = np.zeros((self.iterations, 0))
            return empty, empty.copy(), empty.copy()

        wear_fl = np.zeros((self.iterations, n_laps))
        wear_fr = np.zeros((self.iterations, n_laps))
        lap_times = np.zeros((self.iterations, n_laps))

        for i in range(self.iterations):
            curr_fl, curr_fr = start_fl, start_fr
            for lap_idx in range(n_laps):
                curr_fl = min(1.0, curr_fl + slope_fl * np.random.normal(1.0, 0.05))
                curr_fr = min(1.0, curr_fr + slope_fr * np.random.normal(1.0, 0.05))
                max_wear = max(curr_fl, curr_fr)
                wear_penalty = (max_wear - 0.8) * 15.0 if max_wear > 0.8 else 0.0
                lap_times[i, lap_idx] = base_lap_time + max_wear * 2.0 + wear_penalty + np.random.normal(0, 0.3)
                wear_fl[i, lap_idx] = curr_fl
                wear_fr[i, lap_idx] = curr_fr

        return wear_fl, wear_fr, lap_times

    def _cliff_lap(self, mean_fl: np.ndarray, offset: int) -> int | None:
        """Returns the first absolute lap number where projected mean FL wear exceeds 0.85."""
        for idx, w in enumerate(mean_fl):
            if w > 0.85:
                return offset + idx + 1
        return None

    def project_race(self, window: TelemetryWindow, features: RaceFeatures) -> RaceProjection:
        start_time = time.perf_counter()
        track_id = window.track_id
        current_lap = window.latest.lap
        total_laps = _RACE_LAPS.get(track_id, 53)
        remaining = max(0, total_laps - current_lap)
        base_lap_time = _BASE_LAP_TIME_S.get(track_id, _DEFAULT_LAP_TIME_S)

        if remaining == 0:
            return RaceProjection(
                session_id=window.session_id,
                driver_id=window.driver_id,
                track_id=track_id,
                current_lap=current_lap,
                remaining_laps=0,
                projections=[],
                summary="Race completed.",
                confidence=1.0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        sim_wear_fl, sim_wear_fr, sim_lap_times = self._simulate_stint(
            features.fl_wear, features.fr_wear,
            features.fl_wear_slope, features.fr_wear_slope,
            remaining, base_lap_time,
        )

        projections = []
        for l_idx in range(remaining):
            lap_num = current_lap + l_idx + 1
            if lap_num % 5 == 0 or l_idx >= remaining - 3:
                times = sim_lap_times[:, l_idx]
                projections.append(
                    PredictionPoint(
                        lap=lap_num,
                        p10_time_s=float(np.percentile(times, 10)),
                        p50_time_s=float(np.percentile(times, 50)),
                        p90_time_s=float(np.percentile(times, 90)),
                        wear_fl=float(np.mean(sim_wear_fl[:, l_idx])),
                        wear_fr=float(np.mean(sim_wear_fr[:, l_idx])),
                        grip=float(max(0.4, features.grip_estimate - np.mean(sim_wear_fl[:, l_idx]) * 0.3)),
                    )
                )

        final_mean_wear = float(np.mean(sim_wear_fl[:, -1]))
        summary = f"Projected end-of-race FL wear: {final_mean_wear:.1%}. "
        if final_mean_wear > 0.85:
            summary += "Critical performance cliff predicted before finish. Strategy: Consider one-stop fallback."
        else:
            summary += "Stint trajectory remains viable until the checkered flag."

        return RaceProjection(
            session_id=window.session_id,
            driver_id=window.driver_id,
            track_id=track_id,
            current_lap=current_lap,
            remaining_laps=remaining,
            projections=projections,
            summary=summary,
            confidence=0.85,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

    def compare_strategies(self, window: TelemetryWindow, features: RaceFeatures) -> StrategyComparison:
        start_time = time.perf_counter()
        track_id = window.track_id
        current_lap = window.latest.lap
        total_laps = _RACE_LAPS.get(track_id, 53)
        remaining = max(0, total_laps - current_lap)
        base_lap_time = _BASE_LAP_TIME_S.get(track_id, _DEFAULT_LAP_TIME_S)

        def _run_stay_out() -> dict:
            wfl, _, times = self._simulate_stint(
                features.fl_wear, features.fr_wear,
                features.fl_wear_slope, features.fr_wear_slope,
                remaining, base_lap_time,
            )
            mean_fl = np.mean(wfl, axis=0) if remaining > 0 else np.array([])
            return {
                "total_time_s": float(np.mean(np.sum(times, axis=1))) if remaining > 0 else 0.0,
                "cliff_lap": self._cliff_lap(mean_fl, current_lap),
                "end_wear_fl": float(mean_fl[-1]) if remaining > 0 else features.fl_wear,
                "pit_lap": None,
            }

        def _run_pit(stay_laps: int) -> dict:
            fresh_laps = remaining - stay_laps - 1

            if fresh_laps < 0:
                return _run_stay_out()

            old_time = 0.0
            old_cliff = None
            if stay_laps > 0:
                wfl_old, _, times_old = self._simulate_stint(
                    features.fl_wear, features.fr_wear,
                    features.fl_wear_slope, features.fr_wear_slope,
                    stay_laps, base_lap_time,
                )
                old_time = float(np.mean(np.sum(times_old, axis=1)))
                old_cliff = self._cliff_lap(np.mean(wfl_old, axis=0), current_lap)

            fresh_time = 0.0
            end_wear = _FRESH_WEAR_INITIAL
            if fresh_laps > 0:
                wfl_new, _, times_new = self._simulate_stint(
                    _FRESH_WEAR_INITIAL, _FRESH_WEAR_INITIAL,
                    _FRESH_WEAR_SLOPE, _FRESH_WEAR_SLOPE,
                    fresh_laps, base_lap_time,
                )
                fresh_time = float(np.mean(np.sum(times_new, axis=1)))
                end_wear = float(np.mean(wfl_new[:, -1]))

            return {
                "total_time_s": old_time + base_lap_time + _PIT_LOSS_S + fresh_time,
                "cliff_lap": old_cliff,
                "end_wear_fl": end_wear,
                "pit_lap": current_lap + stay_laps + 1,
            }

        candidates: list[tuple[str, dict]] = [("Stay out", _run_stay_out())]
        if remaining >= 2:
            candidates.append(("Pit this lap", _run_pit(0)))
        if remaining >= 5:
            delay = min(3, remaining // 2)
            candidates.append((f"Pit in {delay} laps", _run_pit(delay)))

        best_time = min(r["total_time_s"] for _, r in candidates)

        scenarios = [
            StrategyScenario(
                label=label,
                pit_lap=r["pit_lap"],
                total_time_s=round(r["total_time_s"], 2),
                delta_s=round(r["total_time_s"] - best_time, 2),
                cliff_lap=r["cliff_lap"],
                end_wear_fl=round(r["end_wear_fl"], 3),
                recommended=(r["total_time_s"] == best_time),
            )
            for label, r in candidates
        ]

        best = next(s for s in scenarios if s.recommended)
        stay_out = next((s for s in scenarios if s.pit_lap is None), None)

        if best.pit_lap is None:
            rec = f"Stay out — tires project safely to the end (~{best.end_wear_fl:.0%} FL wear)."
        else:
            saving = (stay_out.total_time_s - best.total_time_s) if stay_out else 0.0
            rec = f"Pit on lap {best.pit_lap} — projected {saving:.1f}s faster than staying out."

        return StrategyComparison(
            session_id=window.session_id,
            driver_id=window.driver_id,
            track_id=track_id,
            current_lap=current_lap,
            remaining_laps=remaining,
            scenarios=scenarios,
            recommendation=rec,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )
