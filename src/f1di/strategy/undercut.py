"""Undercut-window estimate: comparing two drivers' tire-cliff projections to
estimate whether pitting now would put `driver` ahead of `rival` once the
pit-lane loss and fresh-tire pace gain are accounted for.

Probability model (v2):
    The undercut succeeds when the total fresh-tire pace advantage accumulated
    while the rival stays on old tires exceeds the pit-lane time loss. The
    rival's additional lap count is sampled from their tire-cliff Monte Carlo
    distribution (so uncertainty in when they're forced to stop propagates into
    the probability). The pace gain per lap grows quadratically as the rival's
    tires further degrade each lap they stay out:

        gain(N) = G × (N × wear_rival + slope_eff_rival × N×(N+1)/2)

    Break-even N_q (quadratic) is the lap count where gain(N_q) = pit_loss_s,
    solved analytically. P(success) = fraction of rival-cliff simulations where
    N > N_q (i.e., rival stays out long enough that the undercut pays off).

    v1 used a linear gain model and asked "is the rival still out at break-even?",
    which returned ~0 for every scenario once wear rates were calibrated (rival
    cliff at ~3 laps, linear break-even at ~18 laps). The quadratic model reduces
    break-even significantly (rival's last laps add more advantage than early ones),
    and Monte Carlo sampling over the full crossing distribution correctly handles
    the probability distribution rather than a single median.

Residual limitation:
    The backtest showed ~19% of undercuts succeed in practice. The model correctly
    predicts near-zero for most scenarios from tire data alone. The gap between
    model output and observed base rate represents exogenous success drivers
    (traffic, safety-car timing, rival team strategy calls) that are not visible
    in telemetry. `undercut_success_probability` should be read as the probability
    attributable to tire wear dynamics only — add qualitative judgment for
    strategic context.
"""
from __future__ import annotations

import math

import numpy as np

from f1di.agents.tire_projection import _DEGRADATION_ACCEL, project_cliff_for_window

PIT_LOSS_S = 22.0
FRESH_TIRE_PACE_GAIN_S_PER_WEAR = 2.0
_N_MC = 2000


def _slope_eff_per_lap(features, window) -> float:
    """Degradation-adjusted slope in wear units per lap, matching the
    acceleration model used inside project_cliff_for_window."""
    lap_span = window.latest.lap - window.samples[0].lap
    spl = len(window.samples) / lap_span if lap_span > 0 else 1.0
    fl = getattr(features, "fl_wear_slope_ema", 0.0) or features.fl_wear_slope
    fr = getattr(features, "fr_wear_slope_ema", 0.0) or features.fr_wear_slope
    raw_per_lap = max(fl, fr) * spl
    sf = max(0.0, min(1.0, features.stint_fraction))
    return raw_per_lap * (1.0 + _DEGRADATION_ACCEL * sf)


def _break_even_quad(pit_loss_s: float, wear_rival: float, slope_eff: float, G: float) -> float | None:
    """Quadratic break-even: smallest N where gain(N) ≥ pit_loss_s.

    gain(N) = G × (N × wear_rival + slope_eff × N×(N+1)/2)
            = (G×slope_eff/2)×N² + (G×wear_rival + G×slope_eff/2)×N - pit_loss_s

    Returns None when no real positive root exists (gain never reaches pit_loss_s).
    """
    a = G * slope_eff / 2.0
    b = G * wear_rival + G * slope_eff / 2.0
    c = -pit_loss_s
    if abs(a) < 1e-9:
        return (pit_loss_s / b) if b > 1e-6 else None
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    root = (-b + math.sqrt(disc)) / (2.0 * a)
    return root if root > 0 else None


def _mc_success_prob(
    cliff_rival: dict,
    wear_rival: float,
    slope_eff_rival: float,
    pit_loss_s: float,
    gap_s: float,
    G: float,
    seed: int = 0,
) -> float:
    """P(undercut succeeds) via Monte Carlo over rival cliff-crossing distribution.

    Samples rival's additional lap count N from the cliff projection's probability_by_lap
    CDF (treating "never crosses within horizon" as N = horizon). For each sample,
    computes the quadratic cumulative pace gain and marks it a success when
    gain(N) > pit_loss_s + gap_s.
    """
    horizon = cliff_rival["horizon_laps"]
    probs = cliff_rival["probability_by_lap"]
    laps_arr = np.array(sorted(probs.keys()), dtype=np.float64)
    cdf = np.array([probs[int(k)] for k in laps_arr], dtype=np.float64)

    rng = np.random.default_rng(seed)
    u = rng.uniform(size=_N_MC)

    # Invert CDF: find smallest lap k where cdf[k] >= u; if none, rival hits horizon.
    # Clip before indexing — np.where evaluates both branches unconditionally.
    idx = np.searchsorted(cdf, u, side="left")
    idx_safe = np.clip(idx, 0, len(laps_arr) - 1)
    rival_n = np.where(idx < len(laps_arr), laps_arr[idx_safe], float(horizon))

    threshold = pit_loss_s + gap_s
    gain = G * (rival_n * wear_rival + slope_eff_rival * rival_n * (rival_n + 1.0) / 2.0)
    return float(np.mean(gain > threshold))


def undercut_window(
    year: int,
    round_num: int,
    driver: str,
    rival: str,
    lap_number: int,
    session_type: str = "R",
    pit_loss_s: float | None = None,
    fresh_tire_gain_s_per_wear: float = FRESH_TIRE_PACE_GAIN_S_PER_WEAR,
    gap_s: float = 0.0,
) -> dict:
    """Estimate whether `driver` pitting now would beat `rival` to the punch.

    pit_loss_s: override the circuit-specific pit-lane time loss. When None
    (default), the value is loaded from CircuitThresholds for the circuit,
    which is calibrated from FastF1 PitInTime/PitOutTime measurements.

    gap_s: current on-track time gap from driver to rival in seconds. Positive
    means rival is ahead (typical undercut scenario). Used to tighten the success
    condition — a 5-second gap means the fresh-tire advantage must overcome both
    the pit-lane loss AND the track gap. Default 0 assumes they are side by side.

    Returns a dict with:
        driver/rival wear states and cliff projections,
        laps_to_break_even (linear, for transparency),
        laps_to_break_even_quad (quadratic — what the probability uses),
        undercut_success_probability (MC estimate, tire-dynamics component only),
        model_caveat (honest description of what the number means).
    """
    from f1di.agents.thresholds import get as get_thresholds
    from f1di.features.extractor import extract_features
    from f1di.knowledge.fastf1_session import build_window

    window_driver = build_window(
        year=year, round_num=round_num, driver=driver,
        lap_number=lap_number, session_type=session_type,
    )
    window_rival = build_window(
        year=year, round_num=round_num, driver=rival,
        lap_number=lap_number, session_type=session_type,
    )
    features_driver = extract_features(window_driver)
    features_rival = extract_features(window_rival)

    t = get_thresholds(window_driver.track_id)
    effective_pit_loss_s = pit_loss_s if pit_loss_s is not None else t.pit_loss_s
    cliff_driver = project_cliff_for_window(window_driver, features_driver, t.wear_critical)
    cliff_rival = project_cliff_for_window(window_rival, features_rival, t.wear_critical)

    wear_driver = max(features_driver.fl_wear, features_driver.fr_wear, features_driver.rear_wear_mean)
    wear_rival = max(features_rival.fl_wear, features_rival.fr_wear, features_rival.rear_wear_mean)
    slope_eff_rival = _slope_eff_per_lap(features_rival, window_rival)

    G = fresh_tire_gain_s_per_wear
    fresh_gain_s_per_lap = wear_rival * G  # linear rate at current wear (kept for output)

    if fresh_gain_s_per_lap <= 1e-6:
        laps_to_break_even = None
        laps_to_break_even_quad = None
        undercut_success_probability = 0.0
    else:
        laps_to_break_even = effective_pit_loss_s / fresh_gain_s_per_lap
        laps_to_break_even_quad = _break_even_quad(effective_pit_loss_s, wear_rival, slope_eff_rival, G)
        undercut_success_probability = _mc_success_prob(
            cliff_rival, wear_rival, slope_eff_rival,
            effective_pit_loss_s, gap_s, G,
        )

    return {
        "year": year,
        "round_num": round_num,
        "session_type": session_type.upper(),
        "lap": lap_number,
        "driver": driver.upper(),
        "rival": rival.upper(),
        "driver_current_wear": round(wear_driver, 4),
        "rival_current_wear": round(wear_rival, 4),
        "driver_cliff_eta_laps": cliff_driver["eta_laps"],
        "rival_cliff_eta_laps": cliff_rival["eta_laps"],
        "pit_loss_s": round(effective_pit_loss_s, 2),
        "gap_s": gap_s,
        "fresh_tire_gain_s_per_lap": round(fresh_gain_s_per_lap, 3),
        "slope_eff_rival_per_lap": round(slope_eff_rival, 5),
        "laps_to_break_even": round(laps_to_break_even, 2) if laps_to_break_even is not None else None,
        "laps_to_break_even_quad": round(laps_to_break_even_quad, 2) if laps_to_break_even_quad is not None else None,
        "undercut_success_probability": round(undercut_success_probability, 4),
        "model_caveat": (
            "Heuristic probability from tire-wear dynamics only (quadratic gain MC over rival "
            "cliff distribution). Does not model traffic, safety-car timing, or rival team "
            "strategy. Backtest shows ~19% of undercuts succeed in practice; output near zero "
            "means the wear trajectory alone does not justify the stop — add qualitative "
            "context for strategic factors not visible in telemetry."
        ),
    }
