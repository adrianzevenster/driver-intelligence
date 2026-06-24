#!/usr/bin/env python
"""Replay historical races through the inference pipeline, then auto-label outcomes.

For each requested race this script:
  1. Loads FastF1 lap data and discovers all drivers.
  2. Builds per-lap TelemetryWindows via build_all_lap_windows().
  3. Runs InferenceOrchestrator.analyze() (skip_llm, no drift recording)
     and stores each result as an InsightRecord in the database.
  4. Calls label_race() to match stored insights against actual race
     incidents and write FeedbackRecord rows.
  5. Optionally triggers maybe_retrain_all() after all races are done.

Sessions already replayed are skipped (idempotent via IngestionRecord).
Run this once against 2023 + 2024 to seed the flywheel with thousands of
real labeled examples, then retrain — that is the step that activates
genuine ML value on top of the rule-based baseline.

Usage:
    uv run python scripts/replay_and_label.py --years 2024,2023
    uv run python scripts/replay_and_label.py --years 2024 --rounds 1,2,3
    uv run python scripts/replay_and_label.py --years 2025 --drivers VER,NOR
    uv run python scripts/replay_and_label.py --years 2024 --dry-run
    uv run python scripts/replay_and_label.py --years 2024 --no-retrain
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _build_with_backoff(year: int, round_num: int, drv: str, build_fn) -> dict | None:
    """Call build_fn with retry/backoff on FastF1 rate-limit errors.

    FastF1 enforces 500 calls/hour. Each driver's car_data fetch costs ~5 calls.
    On RateLimitExceededError we wait progressively longer (60s → 120s → 300s)
    before retrying, up to 3 attempts total.
    """
    from fastf1.exceptions import RateLimitExceededError
    delays = [60, 120, 300]
    for attempt, wait in enumerate(delays + [None]):
        try:
            return build_fn(year, round_num, drv)
        except RateLimitExceededError:
            if wait is None:
                _log(f"    {_RED}{drv}: rate limit — giving up after {len(delays)} retries{_RESET}")
                return None
            _log(f"    {_YELLOW}{drv}: rate limit hit — waiting {wait}s before retry {attempt + 1}/{len(delays)}…{_RESET}")
            time.sleep(wait)
        except Exception as exc:
            _log(f"    {_YELLOW}{drv}: window build failed — {exc}{_RESET}")
            return None


def _get_race_drivers(year: int, round_num: int) -> list[str]:
    """Return list of driver codes that completed at least one timed lap."""
    import fastf1
    import os
    cache_dir = str(Path(__file__).parents[1] / "data" / "fastf1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    valid = session.laps[session.laps["LapTime"].notna()]
    return sorted(valid["Driver"].unique().tolist())


def replay_race(
    year: int,
    round_num: int,
    *,
    driver_filter: list[str] | None,
    dry_run: bool,
) -> dict:
    """Replay one race: build windows, run inference, store InsightRecords.

    Returns a summary dict with counts.
    """
    from f1di.knowledge.fastf1_session import build_all_lap_windows
    from f1di.inference.fusion import InferenceOrchestrator
    from f1di.storage.database import db_session
    from f1di.storage.repository import save_insight, already_ingested, mark_ingested

    source = "replay"

    with db_session() as sess:
        if already_ingested(sess, source=source, year=year, round_num=round_num):
            _log(f"  {_DIM}already replayed — skipping{_RESET}")
            return {"skipped": True}

    try:
        drivers = _get_race_drivers(year, round_num)
    except Exception as exc:
        _log(f"  {_RED}could not load driver list: {exc}{_RESET}")
        return {"error": str(exc)}

    if driver_filter:
        drivers = [d for d in drivers if d.upper() in {f.upper() for f in driver_filter}]

    _log(f"  drivers: {', '.join(drivers)}")

    try:
        orchestrator = InferenceOrchestrator()
    except Exception:
        from f1di.rag.store import HybridMemoryRetriever
        from f1di.knowledge.fastf1_session import _DEFAULT_CACHE  # noqa: F401
        from pathlib import Path as _Path
        from f1di.rag.store import load_markdown_knowledge
        retriever = HybridMemoryRetriever()
        kb = _Path("data/knowledge")
        if kb.exists():
            retriever.add_documents(load_markdown_knowledge(kb))
        orchestrator = InferenceOrchestrator(retriever=retriever)

    total_windows = 0
    total_saved = 0

    for drv in drivers:
        windows = _build_with_backoff(year, round_num, drv, build_all_lap_windows)
        if windows is None:
            continue

        if not windows:
            _log(f"    {_DIM}{drv}: no valid laps{_RESET}")
            continue

        driver_insights = []
        for lap_n, window in sorted(windows.items()):
            try:
                insight = orchestrator.analyze(window, skip_llm=True, record_drift=False)
                driver_insights.append((window, insight))
                total_windows += 1
            except Exception as exc:
                _log(f"    {_YELLOW}{drv} lap {lap_n}: analyze failed — {exc}{_RESET}")

        if not driver_insights or dry_run:
            if dry_run:
                total_saved += len(driver_insights)
            _log(f"    {drv}: {len(driver_insights)} windows {'(dry run)' if dry_run else ''}")
            continue

        drv_saved = 0
        for window, insight in driver_insights:
            try:
                with db_session() as sess:
                    save_insight(sess, insight, window)
                drv_saved += 1
            except Exception:
                pass  # duplicate or constraint violation — skip
        if drv_saved:
            total_saved += drv_saved
            _log(f"    {_GREEN}{drv}: {drv_saved} insights saved{_RESET}")
        else:
            _log(f"    {_DIM}{drv}: 0 new insights (already ingested?){_RESET}")

    all_drivers_failed = (total_windows == 0 and total_saved == 0)

    if not dry_run and total_saved > 0 and not all_drivers_failed:
        try:
            with db_session() as sess:
                mark_ingested(
                    sess,
                    source=source,
                    year=year,
                    round_num=round_num,
                    documents_added=total_saved,
                )
        except Exception:
            # Another concurrent run may have already marked this race; ignore.
            pass
    elif not dry_run and all_drivers_failed:
        _log(f"  {_YELLOW}No insights saved — race will be retried on next run{_RESET}")

    return {"n_windows": total_windows, "n_saved": total_saved}


def label_race_outcomes(year: int, round_num: int, *, dry_run: bool) -> dict:
    """Run the outcome labeler for one race."""
    from f1di.data.outcome_labeler import label_race
    from dataclasses import asdict
    try:
        result = label_race(year=year, round_num=round_num, dry_run=dry_run)
        return asdict(result)
    except Exception as exc:
        _log(f"  {_RED}outcome labeler failed: {exc}{_RESET}")
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay historical races and auto-label outcomes")
    parser.add_argument("--years",    required=True, help="Comma-separated years, e.g. 2024,2023")
    parser.add_argument("--rounds",   default="",    help="Comma-separated round numbers; default=all")
    parser.add_argument("--drivers",  default="",    help="Comma-separated driver codes to filter; default=all")
    parser.add_argument("--dry-run",  action="store_true", help="Run inference but do not write to DB")
    parser.add_argument("--no-retrain", action="store_true", help="Skip the final retrain trigger")
    parser.add_argument("--label-only", action="store_true",
                        help="Skip replay (assume InsightRecords already exist), only run outcome labeler")
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    round_filter = [int(r.strip()) for r in args.rounds.split(",") if r.strip()]
    driver_filter = [d.strip().upper() for d in args.drivers.split(",") if d.strip()] or None

    from f1di.knowledge.fastf1_session import get_races

    grand_total_insights = 0
    grand_total_labels   = 0

    for year in years:
        _log(f"\n{_BOLD}=== {year} ==={_RESET}")
        try:
            races = get_races(year)
        except Exception as exc:
            _log(f"  {_RED}could not load race calendar for {year}: {exc}{_RESET}")
            continue

        for race in races:
            round_num  = race["round"]
            event_name = race["name"]

            if round_filter and round_num not in round_filter:
                continue

            _log(f"\n{_CYAN}R{round_num:02d} {event_name}{_RESET}")

            # ── Step 1: replay (unless --label-only) ──────────────────────
            if not args.label_only:
                t0 = time.perf_counter()
                replay_result = replay_race(
                    year, round_num,
                    driver_filter=driver_filter,
                    dry_run=args.dry_run,
                )
                elapsed = time.perf_counter() - t0

                if replay_result.get("skipped"):
                    pass
                elif "error" in replay_result:
                    _log(f"  {_RED}replay error: {replay_result['error']}{_RESET}")
                    continue
                else:
                    n_saved = replay_result.get("n_saved", 0)
                    grand_total_insights += n_saved
                    _log(
                        f"  replay: {replay_result.get('n_windows', 0)} windows → "
                        f"{n_saved} insights in {elapsed:.1f}s"
                        + (" (dry run)" if args.dry_run else "")
                    )

            # ── Step 2: outcome labeling ───────────────────────────────────
            _log("  labeling outcomes …")
            t0 = time.perf_counter()
            label_result = label_race_outcomes(year, round_num, dry_run=args.dry_run)
            elapsed = time.perf_counter() - t0

            if "error" not in label_result:
                n_correct   = label_result.get("n_labeled_correct",   0)
                n_incorrect = label_result.get("n_labeled_incorrect",  0)
                n_examined  = label_result.get("n_insights_examined",  0)
                n_incidents = len(label_result.get("incidents_found",  []))
                grand_total_labels += n_correct + n_incorrect
                _log(
                    f"  labels: examined={n_examined}  "
                    f"{_GREEN}correct={n_correct}{_RESET}  "
                    f"{_RED}incorrect={n_incorrect}{_RESET}  "
                    f"incidents={n_incidents}  ({elapsed:.1f}s)"
                    + (" (dry run)" if args.dry_run else "")
                )
            else:
                _log(f"  {_YELLOW}outcome labeler: {label_result['error']}{_RESET}")

    # ── Final summary ──────────────────────────────────────────────────────
    _log(f"\n{_BOLD}Done.{_RESET}")
    _log(f"  Total insights replayed : {grand_total_insights}")
    _log(f"  Total outcome labels    : {grand_total_labels}")

    if args.dry_run:
        _log(f"  {_YELLOW}Dry run — no data written.{_RESET}")
        return

    if grand_total_labels == 0:
        _log(f"  {_DIM}No new labels — retrain skipped.{_RESET}")
        return

    if args.no_retrain:
        _log(f"  {_DIM}--no-retrain set — skipping retrain.{_RESET}")
        return

    _log("\n  Triggering retrain …")
    try:
        from f1di.agents.auto_retrain import maybe_retrain_all
        maybe_retrain_all()
        _log(f"  {_GREEN}Retrain complete.{_RESET}")
    except Exception as exc:
        _log(f"  {_YELLOW}Retrain failed: {exc}{_RESET}")


if __name__ == "__main__":
    main()
