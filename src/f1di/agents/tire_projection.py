"""Monte Carlo projection of when tire wear crosses a circuit's critical
threshold, given the current wear level and its observed rate of change.

tire.py already computes a single deterministic point projection
(wear + slope * 4 laps). This module replaces that point estimate with a
distribution: perturb the observed per-lap wear rate with Gaussian noise and
propagate thousands of trajectories forward, so a caller gets "the cliff is
N laps away with P% probability" instead of one number with no sense of how
much to trust it.

This is intentionally cheap — a few thousand rows of vectorized numpy, not a
physics simulator — because the only thing being modeled is uncertainty in a
linear extrapolation of an already-noisy slope estimate, not vehicle dynamics.

Non-linear degradation: real tire wear accelerates as a stint progresses —
the slope observed at stint_fraction=0.5 will be lower than the true
degradation rate later in the stint. Backtest data showed a systematic
+9.2-lap positive bias (cliff arrived earlier than predicted) at typical
confident-call stint fractions (~0.5-0.7), implying the effective slope is
~1.7x the measured instantaneous slope at mid-stint. The _DEGRADATION_ACCEL
factor corrects this by scaling the slope proportionally to stint progress.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from f1di.domain.schemas import TelemetryWindow
    from f1di.features.extractor import RaceFeatures

# Noise model: sigma = max(noise_frac * |slope|, noise_floor). The floor
# matters because early in a stint the observed slope is often near zero
# (not enough wear has accumulated yet), and zero slope should not imply
# zero forecast uncertainty.
_DEFAULT_NOISE_FRAC = 0.35
_DEFAULT_NOISE_FLOOR = 0.003
_MAX_HORIZON_LAPS = 30

# Multiplicative acceleration factor for tire degradation slope. Real F1 tire
# wear is super-linear: the slope measured at stint_fraction f is roughly
# (1 + _DEGRADATION_ACCEL * f) times the slope at stint start. Calibrated
# from the cliff-projection backtest (2024, rounds 1-5): median signed error
# was +9.2 laps (cliff arrived sooner than predicted), consistent with a
# ~1.75x slope underestimate at median confident-call stint_fraction ≈ 0.55.
# Solving: 1 + k * 0.55 = 1.75 → k ≈ 1.36, rounded to 1.4.
_DEGRADATION_ACCEL: float = 1.4


def project_cliff(
    *,
    fl_wear: float,
    fr_wear: float,
    fl_wear_slope: float,
    fr_wear_slope: float,
    samples_per_lap: float,
    wear_critical: float,
    laps_remaining: float,
    stint_fraction: float = 0.5,
    n_sims: int = 2000,
    noise_frac: float = _DEFAULT_NOISE_FRAC,
    noise_floor: float = _DEFAULT_NOISE_FLOOR,
    seed: int | None = None,
) -> dict:
    """Monte Carlo projection of when wear crosses `wear_critical`.

    Projects forward whichever of FL/FR currently has higher wear (the
    binding tire) — rear wear isn't included since the existing cliff logic
    in tire.py is FL/FR-driven (axle imbalance is the front-rear signal, not
    a separate rear cliff).

    Returns:
        {
            "eta_laps": float | None,           # median first-crossing lap
            "probability_by_lap": dict[int, float],  # P(crossed by lap N)
            "n_sims": int,
            "horizon_laps": int,
        }
        eta_laps is None when fewer than half the simulated trajectories
        cross the threshold within the horizon — i.e. not a confident call,
        rather than a wrong number presented with false confidence.
    """
    horizon = max(1, min(int(round(laps_remaining)) if laps_remaining > 0 else _MAX_HORIZON_LAPS, _MAX_HORIZON_LAPS))

    if fl_wear >= fr_wear:
        binding_wear, binding_slope_per_sample = fl_wear, fl_wear_slope
    else:
        binding_wear, binding_slope_per_sample = fr_wear, fr_wear_slope

    raw_slope_per_lap = binding_slope_per_sample * samples_per_lap
    sf = max(0.0, min(1.0, stint_fraction))
    # Apply non-linear acceleration: degradation rate rises with stint progress.
    # This corrects the systematic positive bias observed in the cliff backtest.
    binding_slope_per_lap = raw_slope_per_lap * (1.0 + _DEGRADATION_ACCEL * sf)

    rng = np.random.default_rng(seed)
    sigma = max(abs(binding_slope_per_lap) * noise_frac, noise_floor)
    sampled_slopes = rng.normal(binding_slope_per_lap, sigma, size=n_sims)

    laps = np.arange(1, horizon + 1)
    trajectories = binding_wear + sampled_slopes[:, None] * laps[None, :]  # (n_sims, horizon)
    crossed = trajectories >= wear_critical

    any_cross = crossed.any(axis=1)
    first_cross_idx = np.where(any_cross, crossed.argmax(axis=1), -1)

    probability_by_lap = {int(lap): float(crossed[:, i].mean()) for i, lap in enumerate(laps)}

    crossing_idx = first_cross_idx[first_cross_idx >= 0]
    eta_laps = float(np.median(laps[crossing_idx])) if len(crossing_idx) >= n_sims * 0.5 else None

    return {
        "eta_laps": eta_laps,
        "probability_by_lap": probability_by_lap,
        "n_sims": n_sims,
        "horizon_laps": horizon,
    }


def project_cliff_for_window(
    window: "TelemetryWindow", features: "RaceFeatures", wear_critical: float, **kwargs,
) -> dict:
    """project_cliff(), deriving samples_per_lap the same way tire.py does
    (samples in the window / laps spanned), so callers with a window+features
    pair (e.g. comparing two drivers for an undercut) don't duplicate that
    calculation.
    """
    lap_span = window.latest.lap - window.samples[0].lap
    samples_per_lap = len(window.samples) / lap_span if lap_span > 0 else 1.0
    # Prefer EMA slopes (recency-weighted) over equal-weight slopes for the
    # projection; fall back to the standard slope if the EMA field is absent
    # (e.g. RaceFeatures constructed directly in tests without EMA fields).
    fl_slope = features.fl_wear_slope_ema if getattr(features, "fl_wear_slope_ema", 0.0) != 0.0 else features.fl_wear_slope
    fr_slope = features.fr_wear_slope_ema if getattr(features, "fr_wear_slope_ema", 0.0) != 0.0 else features.fr_wear_slope
    return project_cliff(
        fl_wear=features.fl_wear, fr_wear=features.fr_wear,
        fl_wear_slope=fl_slope, fr_wear_slope=fr_slope,
        samples_per_lap=samples_per_lap, wear_critical=wear_critical,
        laps_remaining=features.laps_remaining,
        stint_fraction=features.stint_fraction,
        **kwargs,
    )
