from __future__ import annotations

import logging
import os
import warnings

from datetime import date
from typing import Any

from f1di.knowledge.track_ids import canonical as canonical_track_id
from f1di.rag.store import KnowledgeDocument

from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = str(Path(__file__).parents[3] / "data" / "fastf1_cache")


def _fmt_laptime(td: Any) -> str:
    """Format a pandas Timedelta as M:SS.mmm."""
    try:
        total_s = td.total_seconds()
        mins = int(total_s // 60)
        secs = total_s % 60
        return f"{mins}:{secs:06.3f}"
    except Exception:
        return "?"


def _build_document(year: int, event_name: str, event_round: int) -> KnowledgeDocument:
    import fastf1
    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fastf1.Cache.enable_cache(_CACHE_DIR)
        session = fastf1.get_session(year, event_round, "R")
        session.load(telemetry=False, weather=True, messages=False, laps=True)

    circuit = session.event.get("OfficialEventName", event_name)
    location = session.event.get("Location", event_name)
    country = session.event.get("Country", "")
    race_date = str(session.event.get("EventDate", ""))[:10]
    track_id = canonical_track_id(location)

    laps = session.laps.copy()
    # Drop incomplete laps (safety car in-laps, pit-in/out)
    valid = laps[laps["LapTime"].notna() & laps["IsAccurate"]] if "IsAccurate" in laps.columns else laps[laps["LapTime"].notna()]

    # ── Compound performance ────────────────────────────────────────────
    compound_lines: list[str] = []
    for compound, grp in valid.groupby("Compound", sort=True):
        if len(grp) < 3:
            continue
        avg = grp["LapTime"].mean()
        best = grp["LapTime"].min()
        best_driver = grp.loc[grp["LapTime"].idxmin(), "Driver"]
        best_lap_num = int(grp.loc[grp["LapTime"].idxmin(), "LapNumber"])
        tyre_life_avg = grp["TyreLife"].mean() if "TyreLife" in grp.columns else None
        life_tag = f" | avg tyre life {tyre_life_avg:.0f} laps" if tyre_life_avg else ""
        compound_lines.append(
            f"  {compound}: avg {_fmt_laptime(avg)} | fastest {_fmt_laptime(best)} ({best_driver} L{best_lap_num}){life_tag}"
        )

    # ── Stint structure per driver ──────────────────────────────────────
    stint_lines: list[str] = []
    if "Stint" in valid.columns:
        drv_stints: dict[str, list[str]] = {}
        for (drv, stint_num), grp in valid.groupby(["Driver", "Stint"], sort=True):
            compound = grp["Compound"].iloc[0] if "Compound" in grp.columns else "?"
            lap_start = int(grp["LapNumber"].min())
            lap_end = int(grp["LapNumber"].max())
            best = _fmt_laptime(grp["LapTime"].min())
            drv_stints.setdefault(drv, []).append(f"{compound} L{lap_start}–{lap_end} (best {best})")
        for drv, parts in sorted(drv_stints.items()):
            stops = len(parts) - 1
            stint_lines.append(f"  {drv}: {' → '.join(parts)} | {stops} stop{'s' if stops != 1 else ''}")

    # ── Fastest laps by driver ──────────────────────────────────────────
    fast_lines: list[str] = []
    for drv, grp in valid.groupby("Driver", sort=True):
        best_row = grp.loc[grp["LapTime"].idxmin()]
        s1 = _fmt_laptime(best_row["Sector1Time"]) if "Sector1Time" in best_row and pd.notna(best_row["Sector1Time"]) else "?"
        s2 = _fmt_laptime(best_row["Sector2Time"]) if "Sector2Time" in best_row and pd.notna(best_row["Sector2Time"]) else "?"
        s3 = _fmt_laptime(best_row["Sector3Time"]) if "Sector3Time" in best_row and pd.notna(best_row["Sector3Time"]) else "?"
        compound = best_row.get("Compound", "?") if "Compound" in best_row.index else "?"
        lap_n = int(best_row["LapNumber"])
        fast_lines.append(
            f"  {drv}: {_fmt_laptime(best_row['LapTime'])} L{lap_n} [{compound}] | S1 {s1} S2 {s2} S3 {s3}"
        )
    fast_lines = fast_lines[:20]  # cap at 20 drivers

    # ── Weather ─────────────────────────────────────────────────────────
    weather_text = "No weather data."
    wd = session.weather_data
    if wd is not None and len(wd) > 0:
        tt = wd["TrackTemp"].dropna() if "TrackTemp" in wd.columns else pd.Series([], dtype=float)
        at = wd["AirTemp"].dropna() if "AirTemp" in wd.columns else pd.Series([], dtype=float)
        rain = bool(wd["Rainfall"].any()) if "Rainfall" in wd.columns else False
        wind = wd["WindSpeed"].mean() if "WindSpeed" in wd.columns else None
        parts = []
        if len(tt):
            parts.append(f"Track temp {tt.min():.0f}–{tt.max():.0f}°C")
        if len(at):
            parts.append(f"Air {at.mean():.0f}°C")
        if wind:
            parts.append(f"Wind {wind:.0f} km/h")
        parts.append(f"Rain: {'Yes' if rain else 'No'}")
        weather_text = " | ".join(parts)

    title = f"{year} {event_name} — Lap Analysis"
    sections = [
        f"# {title}",
        f"Circuit: {circuit} | Location: {location}, {country} | Date: {race_date}",
        "",
        "## Compound Performance",
    ]
    sections += compound_lines or ["  No compound data."]
    sections += ["", "## Driver Stint Structure"]
    sections += stint_lines or ["  No stint data."]
    sections += ["", "## Fastest Laps by Driver"]
    sections += fast_lines or ["  No lap data."]
    sections += ["", "## Weather", weather_text]

    return KnowledgeDocument(
        source_id=f"fastf1_{year}_{event_round}",
        title=title,
        text="\n".join(sections),
        metadata={
            "track_id": track_id,
            "year": str(year),
            "source": "fastf1",
            "event_round": str(event_round),
        },
    )


def _build_qualifying_document(year: int, event_name: str, event_round: int) -> KnowledgeDocument | None:
    """Build a qualifying-focused document with best lap per driver and sector breakdown."""
    import fastf1
    import pandas as pd

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fastf1.Cache.enable_cache(_CACHE_DIR)
        try:
            session = fastf1.get_session(year, event_round, "Q")
            session.load(telemetry=False, weather=False, messages=False, laps=True)
        except Exception as exc:
            logger.debug("fastf1_qualifying_unavailable", extra={"year": year, "round": event_round, "error": str(exc)})
            return None

    location = session.event.get("Location", event_name)
    country = session.event.get("Country", "")
    race_date = str(session.event.get("EventDate", ""))[:10]
    track_id = canonical_track_id(location)

    laps = session.laps.copy()
    if laps.empty:
        return None

    # Personal best lap per driver (flag or fallback to min LapTime)
    if "IsPersonalBest" in laps.columns:
        best_laps = laps[laps["IsPersonalBest"]].copy()
    else:
        valid = laps[laps["LapTime"].notna()]
        best_laps = valid.loc[valid.groupby("Driver")["LapTime"].idxmin()].copy() if not valid.empty else laps

    best_laps = best_laps.dropna(subset=["LapTime"]).sort_values("LapTime")
    if best_laps.empty:
        return None

    lap_lines: list[str] = []
    for _, row in best_laps.head(20).iterrows():
        drv = row.get("Driver", "?")
        lap_time = _fmt_laptime(row["LapTime"])
        compound = row.get("Compound", "?") if "Compound" in row.index else "?"
        s1 = _fmt_laptime(row["Sector1Time"]) if "Sector1Time" in row.index and pd.notna(row.get("Sector1Time")) else "?"
        s2 = _fmt_laptime(row["Sector2Time"]) if "Sector2Time" in row.index and pd.notna(row.get("Sector2Time")) else "?"
        s3 = _fmt_laptime(row["Sector3Time"]) if "Sector3Time" in row.index and pd.notna(row.get("Sector3Time")) else "?"
        lap_lines.append(f"  {drv}: {lap_time} [{compound}] | S1 {s1} S2 {s2} S3 {s3}")

    compounds = sorted(best_laps["Compound"].dropna().unique().tolist()) if "Compound" in best_laps.columns else []

    # Theoretical best lap (minimum per sector across all drivers)
    theo_lines: list[str] = []
    for sector, col in [("S1", "Sector1Time"), ("S2", "Sector2Time"), ("S3", "Sector3Time")]:
        if col in laps.columns:
            valid_s = laps[laps[col].notna()]
            if not valid_s.empty:
                best_row = valid_s.loc[valid_s[col].idxmin()]
                theo_lines.append(f"  {sector}: {_fmt_laptime(best_row[col])} ({best_row.get('Driver', '?')})")

    title = f"{year} {event_name} — Qualifying"
    sections = [
        f"# {title}",
        f"Circuit: {location}, {country} | Date: {race_date}",
        f"Compounds used in qualifying: {', '.join(compounds) if compounds else 'unknown'}",
        "",
        "## Best Qualifying Laps by Driver",
    ]
    sections += lap_lines
    if theo_lines:
        sections += ["", "## Theoretical Best (fastest sector per driver combined)"]
        sections += theo_lines

    return KnowledgeDocument(
        source_id=f"fastf1_q_{year}_{event_round}",
        title=title,
        text="\n".join(sections),
        metadata={
            "track_id": track_id,
            "year": str(year),
            "source": "fastf1",
            "session_type": "qualifying",
            "event_round": str(event_round),
        },
    )


def ingest(retriever, *, years: list[int] | None = None, n_per_year: int = 5, include_qualifying: bool = True) -> list[str]:
    import fastf1

    if years is None:
        current = date.today().year
        years = [current, current - 1]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fastf1.Cache.enable_cache(_CACHE_DIR)

    all_events: list[tuple[int, str, int]] = []
    for yr in years:
        try:
            schedule = fastf1.get_event_schedule(yr, include_testing=False)
            # Filter to races that have already happened
            today_str = str(date.today())
            past = schedule[schedule["EventDate"].astype(str) <= today_str]
            past = past.tail(n_per_year)  # most recent N
            for _, row in past.iterrows():
                all_events.append((yr, row["EventName"], int(row["RoundNumber"])))
            logger.info("fastf1_schedule_fetched", extra={"year": yr, "count": len(past)})
        except Exception as exc:
            logger.warning("fastf1_schedule_failed", extra={"year": yr, "error": str(exc)})

    docs: list[KnowledgeDocument] = []
    ingested: list[str] = []

    # FastF1 loads are IO-bound but not thread-safe for the cache; use sequential
    for yr, name, rnd in all_events:
        # Race session
        try:
            doc = _build_document(yr, name, rnd)
            docs.append(doc)
            ingested.append(doc.title)
        except Exception as exc:
            logger.warning("fastf1_race_skipped", extra={"error": f"{yr} {name}: {exc}"})

        # Qualifying session
        if include_qualifying:
            try:
                qdoc = _build_qualifying_document(yr, name, rnd)
                if qdoc:
                    docs.append(qdoc)
                    ingested.append(qdoc.title)
            except Exception as exc:
                logger.debug("fastf1_quali_skipped", extra={"error": f"{yr} {name}: {exc}"})

    if docs:
        retriever.add_documents(docs)

    return ingested
