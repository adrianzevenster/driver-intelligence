#!/usr/bin/env python
"""Race weekend labelling CLI.

Walk through insights stored in the DB for a given session and driver,
display each agent's finding, and record correct/incorrect feedback.
Each labelled insight feeds directly into the auto-retrain flywheel.

Usage:
    uv run python scripts/label_race.py --session monaco_2026 --driver VER
    uv run python scripts/label_race.py --session bahrain_2026  # all drivers
    uv run python scripts/label_race.py --session monaco_2026 --driver VER --dry-run
    uv run python scripts/label_race.py --list-sessions

Controls (per insight):
    y / 1  → correct (rating=5)
    n / 0  → incorrect (rating=1)
    s      → skip (no label written)
    q      → quit (save progress so far)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"

_RISK_COLOR = {
    "INFO":     "\033[94m",
    "WATCH":    _YELLOW,
    "WARNING":  "\033[38;5;208m",
    "CRITICAL": _RED,
}

_AGENT_LABEL = {
    "tire_strategy": "Tire Strategy",
    "battery":       "Battery / ERS",
    "weather":       "Weather",
    "telemetry":     "Telemetry",
    "safety_car":    "Safety Car",
    "fuel":          "Fuel Strategy",
}


def _rc(risk: str) -> str:
    return _RISK_COLOR.get(risk, _RESET)


def _print_insight(idx: int, total: int, rec) -> None:
    findings = rec.findings
    print()
    print(f"{_BOLD}{'─'*60}{_RESET}")
    print(
        f"{_BOLD}[{idx}/{total}]{_RESET}  "
        f"Session: {_CYAN}{rec.session_id}{_RESET}  "
        f"Driver: {_BOLD}{rec.driver_id}{_RESET}  "
        f"Lap: {rec.lap or '?'}  "
        f"Risk: {_rc(rec.risk)}{_BOLD}{rec.risk}{_RESET}"
    )
    print(f"{_DIM}insight_id: {rec.insight_id}{_RESET}")
    print()
    print(f"  {_BOLD}Recommendation:{_RESET} {rec.recommendation}")
    print()
    print(f"  {_BOLD}Agent findings:{_RESET}")
    for f in sorted(findings, key=lambda x: ["CRITICAL", "WARNING", "WATCH", "INFO"].index(x.get("risk", "INFO"))):
        agent = _AGENT_LABEL.get(f.get("agent", ""), f.get("agent", "?"))
        risk  = f.get("risk", "INFO")
        conf  = f.get("confidence", 0)
        summ  = f.get("summary", "")[:90]
        src   = " [LR]" if f.get("clf_source") else ""
        ood   = " [OOD]" if f.get("ood_flagged") else ""
        print(f"    {_rc(risk)}{risk:<10}{_RESET} {agent:<18} {conf*100:>4.0f}%{src}{ood}")
        print(f"    {_DIM}{summ}…{_RESET}")


def _prompt() -> str:
    while True:
        try:
            raw = input(f"\n  Label → {_GREEN}[y]es{_RESET}/{_RED}[n]o{_RESET}/{_YELLOW}[s]kip{_RESET}/{_DIM}[q]uit{_RESET}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if raw in {"y", "1", "yes"}:
            return "y"
        if raw in {"n", "0", "no"}:
            return "n"
        if raw in {"s", "skip", ""}:
            return "s"
        if raw in {"q", "quit", "exit"}:
            return "q"
        print("  Please enter y, n, s, or q.")


def _write_feedback(session, insight_id: str, correct: bool, submitted_by: str) -> None:
    from f1di.storage.models import FeedbackRecord
    fb = FeedbackRecord(
        insight_id=insight_id,
        rating=5 if correct else 1,
        correct=correct,
        submitted_by=submitted_by,
    )
    session.add(fb)
    session.commit()


def _list_sessions() -> None:
    from sqlalchemy import select, func
    from f1di.storage.database import db_session, init_db
    from f1di.storage.models import InsightRecord
    init_db()
    with db_session() as session:
        rows = session.execute(
            select(
                InsightRecord.session_id,
                InsightRecord.driver_id,
                func.count(InsightRecord.id).label("n"),
            ).group_by(InsightRecord.session_id, InsightRecord.driver_id)
            .order_by(InsightRecord.session_id, InsightRecord.driver_id)
        ).all()
    if not rows:
        print("No insights in DB yet.")
        return
    print(f"\n{'Session':<40} {'Driver':<8} {'Insights':>8}")
    print("─" * 60)
    for session_id, driver_id, n in rows:
        print(f"{session_id:<40} {driver_id:<8} {n:>8}")
    print()


def main(session_id: str, driver_id: str | None, dry_run: bool, submitted_by: str) -> None:
    from sqlalchemy import select
    from f1di.storage.database import db_session, init_db
    from f1di.storage.models import FeedbackRecord, InsightRecord
    init_db()

    with db_session() as session:
        q = select(InsightRecord).where(InsightRecord.session_id == session_id)
        if driver_id:
            q = q.where(InsightRecord.driver_id == driver_id)
        q = q.order_by(InsightRecord.driver_id, InsightRecord.lap)
        rows = session.execute(q).scalars().all()

        if not rows:
            print(f"No insights found for session '{session_id}'" +
                  (f" driver '{driver_id}'" if driver_id else "") + ".")
            return

        # Skip already-labelled
        existing = {
            fb.insight_id
            for fb in session.execute(
                select(FeedbackRecord.insight_id)
                .where(FeedbackRecord.insight_id.in_([r.insight_id for r in rows]))
            ).scalars()
        }
        unlabelled = [r for r in rows if r.insight_id not in existing]
        already    = len(rows) - len(unlabelled)

        print(f"\n{_BOLD}Race weekend labelling{_RESET} — session: {_CYAN}{session_id}{_RESET}")
        if driver_id:
            print(f"Driver filter: {_BOLD}{driver_id}{_RESET}")
        print(f"{len(rows)} insights total · {already} already labelled · {_BOLD}{len(unlabelled)} to review{_RESET}")
        if dry_run:
            print(f"{_YELLOW}DRY RUN — no feedback will be written{_RESET}")

        n_correct = n_incorrect = n_skip = 0

        for idx, rec in enumerate(unlabelled, 1):
            _print_insight(idx, len(unlabelled), rec)
            action = _prompt()
            if action == "q":
                print(f"\n{_DIM}Quit after {idx-1} reviewed.{_RESET}")
                break
            if action == "s":
                n_skip += 1
                continue
            correct = action == "y"
            if correct:
                n_correct += 1
            else:
                n_incorrect += 1
            if not dry_run:
                _write_feedback(session, rec.insight_id, correct, submitted_by)

        print(f"\n{_BOLD}Summary{_RESET}")
        print(f"  {_GREEN}Correct:   {n_correct}{_RESET}")
        print(f"  {_RED}Incorrect: {n_incorrect}{_RESET}")
        print(f"  {_DIM}Skipped:   {n_skip}{_RESET}")

        total_written = n_correct + n_incorrect
        if total_written > 0 and not dry_run:
            print(f"\n  {total_written} labels written — triggering auto-retrain check...")
            try:
                from f1di.storage.database import init_db as _init
                _init()
                from f1di.agents.auto_retrain import maybe_retrain_all
                maybe_retrain_all()
                print(f"  {_GREEN}Auto-retrain check complete.{_RESET}")
            except Exception as exc:
                print(f"  {_YELLOW}Auto-retrain check skipped: {exc}{_RESET}")
        elif dry_run and total_written > 0:
            print(f"\n  {_DIM}(dry run — {total_written} labels not written){_RESET}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Race weekend labelling CLI")
    parser.add_argument("--session",     required=False, help="Session ID to label")
    parser.add_argument("--driver",      required=False, help="Filter to a single driver code (e.g. VER)")
    parser.add_argument("--dry-run",     action="store_true", help="Preview without writing feedback")
    parser.add_argument("--list-sessions", action="store_true", help="List available sessions and exit")
    parser.add_argument("--submitted-by", default="engineer", help="Label author (default: engineer)")
    args = parser.parse_args()

    if args.list_sessions:
        _list_sessions()
        sys.exit(0)

    if not args.session:
        parser.error("--session is required (or use --list-sessions to see available sessions)")

    main(
        session_id=args.session,
        driver_id=args.driver,
        dry_run=args.dry_run,
        submitted_by=args.submitted_by,
    )
