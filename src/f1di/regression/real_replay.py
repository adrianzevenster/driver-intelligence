from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from f1di.confidence.calibration import compute_raw_score
from f1di.domain.schemas import Compound, DriverInsight, RiskLevel, TelemetrySample, TelemetryWindow
from f1di.inference.fusion import InferenceOrchestrator

RISK_RANK = {
    RiskLevel.INFO: 0,
    RiskLevel.WATCH: 1,
    RiskLevel.WARNING: 2,
    RiskLevel.CRITICAL: 3,
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def window_from_case(case: dict[str, Any]) -> TelemetryWindow:
    if "window" in case:
        return TelemetryWindow.model_validate(case["window"])
    return build_window(case)


def build_window(case: dict[str, Any], samples: int = 12) -> TelemetryWindow:
    profile = str(case["profile"])
    compound = Compound(str(case["compound"]))
    lap = int(case["lap"])
    stint_lap = int(case["stint_lap"])
    track_id = str(case["track_id"])
    driver_id = str(case["driver_id"])
    session_id = f"real-replay-{case['case_id']}"

    generated: list[TelemetrySample] = []
    for idx in range(samples):
        t = idx / max(samples - 1, 1)
        phase = 0.18 + 0.52 * t
        braking = idx in {2, 6, 10}
        sector = min(3, max(1, int(phase * 3) + 1))
        speed = 292.0 - (104.0 if braking else 0.0) + 18.0 * math.sin(t * math.tau)
        throttle = 18.0 if braking else 86.0
        brake_bar = 112.0 if braking else 6.0
        steering = 7.0 + 18.0 * math.sin(t * math.tau * 1.7)

        fl_wear = 0.34 + 0.02 * t
        fr_wear = 0.32 + 0.018 * t
        rl_wear = 0.27 + 0.012 * t
        rr_wear = 0.26 + 0.012 * t
        grip = 0.82
        rain = 0.02
        battery_soc = 0.62 - 0.002 * idx
        lockup = False
        front_brake_temp = 500.0 + 22.0 * t
        tire_temp = 91.0 + 5.0 * t
        wind_speed = 12.0

        if profile == "front_left_cliff":
            fl_wear = 0.58 + 0.014 * idx
            fr_wear = 0.55 + 0.012 * idx
            rl_wear = 0.42 + 0.006 * idx
            rr_wear = 0.41 + 0.006 * idx
            grip = 0.65 - 0.005 * idx
            tire_temp = 104.0 + 1.1 * idx
        elif profile == "rain_crossover":
            rain = 0.20 + 0.018 * idx
            grip = 0.76 - 0.014 * idx
            wind_speed = 25.0
        elif profile == "ers_depletion":
            battery_soc = 0.34 - 0.014 * idx
            speed += 12.0
        elif profile == "brake_lockup":
            fl_wear = 0.68 + 0.012 * idx
            fr_wear = 0.64 + 0.010 * idx
            rl_wear = 0.50 + 0.005 * idx
            rr_wear = 0.49 + 0.005 * idx
            grip = 0.63 - 0.006 * idx
            lockup = idx in {8, 10, 11}
            front_brake_temp = 880.0 + 15.0 * idx
            brake_bar = 130.0 if braking or lockup else 12.0
            tire_temp = 107.0 + 1.0 * idx
        elif profile == "high_wear_stable_grip":
            fl_wear = 0.57 + 0.003 * idx
            fr_wear = 0.55 + 0.002 * idx
            rl_wear = 0.44 + 0.001 * idx
            rr_wear = 0.43 + 0.001 * idx
            grip = 0.76
            tire_temp = 96.0 + 0.2 * idx
        elif profile == "damp_no_crossover":
            rain = 0.09 + 0.004 * idx
            grip = 0.79 - 0.002 * idx
            wind_speed = 16.0
        elif profile == "low_battery_recovering":
            battery_soc = 0.18 + 0.008 * idx
            throttle = 52.0 if not braking else throttle
            speed -= 18.0
        elif profile == "single_lockup_recovered":
            lockup = idx == 5
            front_brake_temp = 650.0 + (85.0 if lockup else 4.0 * idx)
            brake_bar = 118.0 if lockup else brake_bar
        elif profile == "critical_cliff_imminent":
            # Extreme wear + collapsing grip — tire CRITICAL + telemetry WARNING guaranteed
            fl_wear = min(0.99, 0.82 + 0.010 * idx)
            fr_wear = min(0.99, 0.79 + 0.008 * idx)
            rl_wear = min(0.99, 0.68 + 0.005 * idx)
            rr_wear = min(0.99, 0.66 + 0.004 * idx)
            grip = max(0.35, 0.58 - 0.006 * idx)
            tire_temp = 112.0 + 1.2 * idx
        elif profile == "multi_stress":
            # Rain at crossover + front wear above warning — weather + tire both fire
            fl_wear = min(0.99, 0.54 + 0.014 * idx)
            fr_wear = min(0.99, 0.52 + 0.012 * idx)
            rl_wear = min(0.99, 0.38 + 0.007 * idx)
            rr_wear = min(0.99, 0.36 + 0.006 * idx)
            grip = max(0.40, 0.70 - 0.018 * idx)
            rain = 0.30 + 0.010 * idx
            wind_speed = 22.0
            tire_temp = 89.0 + 0.8 * idx
        elif profile == "cold_restart":
            # Safety-car restart: multiple lockups on cold tires — telemetry CRITICAL
            fl_wear = min(0.99, 0.36 + 0.004 * idx)
            fr_wear = min(0.99, 0.34 + 0.003 * idx)
            rl_wear = min(0.99, 0.28 + 0.002 * idx)
            rr_wear = min(0.99, 0.27 + 0.002 * idx)
            grip = max(0.55, 0.72 - 0.006 * idx)
            tire_temp = 62.0 + 5.0 * idx
            lockup = idx in {1, 2, 4}
            front_brake_temp = 920.0 + 12.0 * idx if lockup else 560.0 + 6.0 * idx
            brake_bar = 125.0 if braking or lockup else 6.0

        generated.append(
            TelemetrySample(
                session_id=session_id,
                driver_id=driver_id,
                track_id=track_id,
                timestamp_ms=idx * 3500,
                lap=lap,
                sector=sector,
                distance_m=5891.0 * (lap - 1 + phase),
                corner_id=f"T{1 + int(phase * 18)}",
                speed_kph=max(55.0, speed),
                acceleration_g=(throttle - brake_bar / 2) / 100,
                throttle_pct=throttle,
                brake_pressure_bar=brake_bar,
                steering_angle_deg=steering,
                yaw_rate_deg_s=steering * max(55.0, speed) / 190,
                slip_angle_deg=abs(steering) / 18,
                wheel_speed_fl=max(55.0, speed) * (0.965 if lockup else 1.0),
                wheel_speed_fr=max(55.0, speed) * (0.970 if lockup else 1.0),
                wheel_speed_rl=max(55.0, speed),
                wheel_speed_rr=max(55.0, speed),
                compound=compound,
                stint_lap=stint_lap,
                tire_temp_fl_c=tire_temp,
                tire_temp_fr_c=tire_temp - 2.0,
                tire_temp_rl_c=tire_temp - 5.0,
                tire_temp_rr_c=tire_temp - 6.0,
                tire_wear_fl=min(0.99, fl_wear),
                tire_wear_fr=min(0.99, fr_wear),
                tire_wear_rl=min(0.99, rl_wear),
                tire_wear_rr=min(0.99, rr_wear),
                grip_estimate=max(0.35, grip),
                lockup_event=lockup,
                battery_soc=max(0.05, battery_soc),
                ers_deploy_kw=125.0 if throttle > 80 else 20.0,
                ers_regen_kw=80.0 if braking else 5.0,
                pu_thermal_state=min(1.0, 0.55 + throttle / 220),
                track_temp_c=39.0 - rain * 7,
                ambient_temp_c=25.0,
                humidity_pct=min(100.0, 58.0 + rain * 50),
                wind_speed_kph=wind_speed,
                wind_direction_deg=235.0,
                rain_intensity=rain,
                evolving_grip=max(0.40, 0.88 - rain * 0.45),
                brake_temp_fl_c=front_brake_temp,
                brake_temp_fr_c=front_brake_temp - 8.0,
                brake_temp_rl_c=front_brake_temp * 0.67,
                brake_temp_rr_c=front_brake_temp * 0.65,
            )
        )

    return TelemetryWindow(
        session_id=session_id,
        driver_id=driver_id,
        track_id=track_id,
        samples=generated,
    )


def _meets_min_risk(insight: DriverInsight, expected: str) -> bool:
    return RISK_RANK[insight.risk] >= RISK_RANK[RiskLevel(expected)]


def _meets_max_risk(insight: DriverInsight, expected: str) -> bool:
    return RISK_RANK[insight.risk] <= RISK_RANK[RiskLevel(expected)]


def evaluate_cases(
    cases: list[dict[str, Any]],
    orchestrator: InferenceOrchestrator,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        insight = orchestrator.analyze(window_from_case(case), skip_llm=True, record_drift=False)
        expected_agents = set(case.get("expected_agents", []))
        expected_sources = set(case.get("expected_sources", []))
        expected_policy = case.get("expected_policy")
        active_agents = {f.agent for f in insight.findings if RISK_RANK[f.risk] >= RISK_RANK[RiskLevel.WATCH]}
        evidence_sources = {e.source_id for e in insight.evidence}
        _raw, debug_features = compute_raw_score(insight.findings)

        min_risk_ok = (
            _meets_min_risk(insight, str(case["expected_min_risk"]))
            if "expected_min_risk" in case
            else True
        )
        max_risk_ok = (
            _meets_max_risk(insight, str(case["expected_max_risk"]))
            if "expected_max_risk" in case
            else True
        )
        agents_ok = expected_agents.issubset(active_agents)
        evidence_ok = bool(insight.evidence)
        sources_ok = expected_sources.issubset(evidence_sources)
        policy_ok = insight.policy == expected_policy if expected_policy else True
        passed = min_risk_ok and max_risk_ok and agents_ok and evidence_ok and sources_ok and policy_ok

        rows.append({
            "case_id": case["case_id"],
            "class": case.get("class", case.get("profile", "unknown")),
            "source": case.get("source", {}),
            "label": case.get("label", {}),
            "expected_min_risk": case.get("expected_min_risk"),
            "expected_max_risk": case.get("expected_max_risk"),
            "observed_risk": insight.risk.value,
            "observed_policy": insight.policy,
            "expected_policy": expected_policy,
            "confidence": round(insight.confidence, 4),
            "active_agents": sorted(active_agents),
            "expected_agents": sorted(expected_agents),
            "evidence_sources": sorted(evidence_sources),
            "expected_sources": sorted(expected_sources),
            "evidence_count": len(insight.evidence),
            "calibration_debug": {
                "risk_mean": round(debug_features["risk_mean"], 4),
                "risk_max": round(debug_features["risk_max"], 4),
                "agent_agreement": round(debug_features["agent_agreement"], 4),
                "evidence_strength": round(debug_features["evidence_strength"], 4),
                "model_confidence": round(debug_features["model_confidence"], 4),
            },
            "latency_ms": round(insight.latency_ms, 3),
            "checks": {
                "risk_min": min_risk_ok,
                "risk_max": max_risk_ok,
                "agents": agents_ok,
                "evidence": evidence_ok,
                "sources": sources_ok,
                "policy": policy_ok,
            },
            "pass": passed,
        })

    positives = [r for r in rows if r["expected_min_risk"]]
    nominal = [r for r in rows if r["expected_max_risk"]]
    agent_expected = [r for r in rows if r["expected_agents"]]
    source_expected = [r for r in rows if r["expected_sources"]]
    policy_expected = [r for r in rows if r["expected_policy"]]
    by_class = {}
    for cls in sorted({str(r["class"]) for r in rows}):
        cls_rows = [r for r in rows if r["class"] == cls]
        cls_positive = [r for r in cls_rows if r["expected_min_risk"]]
        by_class[cls] = {
            "cases": len(cls_rows),
            "positive_cases": len(cls_positive),
            "passed": sum(1 for r in cls_rows if r["pass"]),
            "recall": round(
                sum(1 for r in cls_positive if r["pass"]) / max(len(cls_positive), 1),
                4,
            ) if cls_positive else None,
        }
    recall = sum(1 for r in positives if r["pass"]) / max(len(positives), 1)
    false_positive_ok = all(r["pass"] for r in nominal)
    evidence_ok = all(r["evidence_count"] > 0 for r in rows)
    source_ok = all(set(r["expected_sources"]).issubset(set(r["evidence_sources"])) for r in rows)
    agent_activation_rate = (
        sum(1 for r in agent_expected if r["checks"]["agents"]) / max(len(agent_expected), 1)
    )
    source_retrieval_rate = (
        sum(1 for r in source_expected if r["checks"]["sources"]) / max(len(source_expected), 1)
    )
    policy_correctness = (
        sum(1 for r in policy_expected if r["checks"]["policy"]) / max(len(policy_expected), 1)
    )
    false_positive_rate = (
        sum(1 for r in nominal if not r["checks"]["risk_max"]) / max(len(nominal), 1)
    )

    return {
        "cases": rows,
        "by_class": by_class,
        "case_recall": round(recall, 4),
        "agent_activation_rate": round(agent_activation_rate, 4),
        "source_retrieval_rate": round(source_retrieval_rate, 4),
        "policy_correctness": round(policy_correctness, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "nominal_cases": len(nominal),
        "positive_cases": len(positives),
        "pass_case_recall": recall >= 1.0,
        "pass_nominal_false_positive": false_positive_ok,
        "pass_agent_activation": agent_activation_rate >= 1.0,
        "pass_evidence": evidence_ok,
        "pass_expected_sources": source_ok,
        "pass_policy_correctness": policy_correctness >= 1.0,
    }


def run_real_replay_gate(
    fixture_path: Path,
    output_report: Path,
    orchestrator: InferenceOrchestrator,
) -> dict[str, Any]:
    report = evaluate_cases(load_cases(fixture_path), orchestrator)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = [name for name, value in report.items() if name.startswith("pass_") and value is not True]
    if failed:
        raise SystemExit(f"Real replay gates failed: {', '.join(failed)}")
    return report
