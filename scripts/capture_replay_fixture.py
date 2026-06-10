#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from f1di.domain.schemas import TelemetryWindow


def _write_case(path: Path, case: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    existing.append(case)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def _base_case(args: argparse.Namespace, window: TelemetryWindow) -> dict:
    case = {
        "case_id": args.case_id,
        "class": args.case_class,
        "source": {
            "type": args.source_type,
            "series": args.series,
            "event": args.event,
            "session": args.session,
            "year": args.year,
            "round": args.round,
            "driver": args.driver,
            "lap": args.lap,
            "labeling_date": args.labeling_date,
            "labeler": args.labeler,
            "provenance_note": args.provenance_note,
        },
        "label": {
            "rationale": args.label_rationale,
            "outcome": args.label_outcome,
        },
        "expected_sources": args.expected_source,
        "window": window.model_dump(mode="json"),
    }
    if args.expected_min_risk:
        case["expected_min_risk"] = args.expected_min_risk
    if args.expected_max_risk:
        case["expected_max_risk"] = args.expected_max_risk
    if args.expected_agent:
        case["expected_agents"] = args.expected_agent
    if args.expected_policy:
        case["expected_policy"] = args.expected_policy
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a labeled replay case into the evaluation fixture.")
    parser.add_argument("--provider", choices=["fastf1", "openf1"], required=True)
    parser.add_argument("--output", default="data/fixtures/real_replay_eval.json")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-class", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--session-key", type=int)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--driver-number", type=int)
    parser.add_argument("--lap", type=int)
    parser.add_argument("--event", required=True)
    parser.add_argument("--session", default="Race")
    parser.add_argument("--label-rationale", required=True)
    parser.add_argument("--label-outcome", required=True)
    parser.add_argument("--expected-min-risk")
    parser.add_argument("--expected-max-risk")
    parser.add_argument("--expected-agent", action="append", default=[])
    parser.add_argument("--expected-source", action="append", default=[])
    parser.add_argument("--expected-policy")
    parser.add_argument("--labeler", default="manual")
    parser.add_argument("--labeling-date", default="2026-06-06")
    parser.add_argument("--provenance-note", default="Captured from provider API/cache.")
    args = parser.parse_args()

    if args.provider == "fastf1":
        if args.round is None:
            raise SystemExit("--round is required for --provider fastf1")
        from f1di.knowledge.fastf1_session import build_window

        window = build_window(
            year=args.year,
            round_num=args.round,
            driver=args.driver,
            lap_number=args.lap,
        )
        args.source_type = "captured_fastf1"
        args.series = "fastf1"
    else:
        if args.session_key is None or args.driver_number is None:
            raise SystemExit("--session-key and --driver-number are required for --provider openf1")
        from f1di.knowledge.openf1_live import build_window

        window = build_window(
            session_key=args.session_key,
            driver_number=args.driver_number,
            lap_number=args.lap,
        )
        args.source_type = "captured_openf1"
        args.series = "openf1"

    _write_case(Path(args.output), _base_case(args, window))


if __name__ == "__main__":
    main()

