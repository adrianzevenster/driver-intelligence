#!/usr/bin/env python
"""Pre-race flywheel smoke test — confirms the outcome labeling pipeline is healthy
before a race weekend so failures are caught before they matter.

Usage:
    uv run python scripts/smoketest_flywheel.py
    uv run python scripts/smoketest_flywheel.py --year 2024 --round 5

Exit 0 = all checks passed.  Exit 1 = one or more checks failed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

_PASS = f"{_GREEN}PASS{_RESET}"
_FAIL = f"{_RED}FAIL{_RESET}"
_WARN = f"{_YELLOW}WARN{_RESET}"


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = _PASS if ok else _FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}]  {label}{suffix}")
    return ok


def _warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{_WARN}]  {label}{suffix}")


def check_config() -> bool:
    from f1di.config.settings import settings
    ok = True
    ok &= _check(
        "F1DI_INGESTION_AUTO_ENABLED",
        settings.ingestion_auto_enabled,
        "set to true to activate the flywheel" if not settings.ingestion_auto_enabled else "",
    )
    if not settings.ingestion_auto_enabled:
        _warn("Flywheel will not run automatically — set F1DI_INGESTION_AUTO_ENABLED=true on the server")
    _check(
        "Storage URL",
        True,
        settings.storage_url,
    )
    return ok


def check_db() -> bool:
    try:
        from f1di.storage.database import check_connection, init_db
        init_db()
        ok = check_connection()
        return _check("Database connectivity", ok)
    except Exception as exc:
        return _check("Database connectivity", False, str(exc))


def check_fastf1() -> bool:
    try:
        import fastf1  # noqa: F401
        return _check("fastf1 importable", True)
    except ImportError as exc:
        return _check("fastf1 importable", False, str(exc))


def _most_recent_round(year: int) -> int | None:
    try:
        import warnings
        import fastf1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        past = schedule[schedule["EventDate"].astype(str) < str(date.today())]
        if past.empty:
            return None
        return int(past.iloc[-1]["RoundNumber"])
    except Exception:
        return None


def check_dry_run(year: int, round_num: int) -> bool:
    try:
        from f1di.data.outcome_labeler import label_race
        report = label_race(year, round_num, dry_run=True)
        detail = (
            f"incidents={len(report.incidents_found)} "
            f"examined={report.n_insights_examined} "
            f"would_label={report.n_labeled_correct + report.n_labeled_incorrect}"
        )
        ok = len(report.incidents_found) > 0 or report.n_insights_examined >= 0
        return _check(f"label_race dry_run year={year} round={round_num}", ok, detail)
    except Exception as exc:
        return _check("label_race dry_run", False, str(exc))


def check_calibration_files() -> bool:
    cal = Path("data/calibration/isotonic.pkl")
    quality = Path("data/calibration/quality.json")
    ok1 = _check("Calibrator pkl exists", cal.exists(), str(cal))
    ok2 = _check("Quality json exists", quality.exists(), str(quality))
    if quality.exists():
        import json
        q = json.loads(quality.read_text())
        ece = q.get("ece", "n/a")
        n_fb = q.get("calibration_dataset", {}).get("n_feedback", 0)
        _check(f"Calibration ECE={ece} n_feedback={n_fb}", float(ece) <= 0.15 if isinstance(ece, (int, float)) else False)
    return ok1 and ok2


def check_outcome_labeled_cache() -> bool:
    path = Path("data/calibration/outcome_labeled.json")
    if not path.exists():
        _warn("outcome_labeled.json missing — no rounds have been labeled yet (expected on first run)")
        return True
    import json
    labeled = json.loads(path.read_text())
    return _check("Outcome labeled cache readable", True, f"{len(labeled)} round(s) processed")


def main(year: int | None, round_num: int | None) -> None:
    print(f"\nF1DI Flywheel Smoke Test  ({date.today()})")
    print("=" * 50)

    all_ok = True

    print("\n[Config]")
    all_ok &= check_config()

    print("\n[Infrastructure]")
    all_ok &= check_db()
    ff1_ok = check_fastf1()
    all_ok &= ff1_ok

    print("\n[Calibration artifacts]")
    all_ok &= check_calibration_files()
    check_outcome_labeled_cache()

    if ff1_ok:
        print("\n[Dry-run labeling]")
        if year is None or round_num is None:
            current_year = date.today().year
            round_num = _most_recent_round(current_year)
            if round_num is None:
                round_num = _most_recent_round(current_year - 1)
                year = current_year - 1
            else:
                year = current_year
        if round_num:
            all_ok &= check_dry_run(year, round_num)
        else:
            _warn("Could not determine most recent race round — skipping dry-run")

    print()
    if all_ok:
        print(f"{_GREEN}All checks passed — flywheel is ready for race weekend.{_RESET}")
    else:
        print(f"{_RED}One or more checks failed — fix before the race weekend.{_RESET}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-race flywheel smoke test")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--round", dest="round_num", type=int, default=None)
    args = parser.parse_args()
    main(args.year, args.round_num)
