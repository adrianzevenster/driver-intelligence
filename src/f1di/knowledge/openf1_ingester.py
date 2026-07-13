from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import httpx

from f1di.rag.store import KnowledgeDocument

logger = logging.getLogger(__name__)

_BASE = "https://api.openf1.org/v1"
_TIMEOUT = 20.0


def _get(endpoint: str, **params) -> list[dict]:
    for attempt in range(4):
        r = httpx.get(f"{_BASE}/{endpoint}", params=params, timeout=_TIMEOUT)
        if r.status_code == 429:
            wait = 2 ** attempt
            logger.debug("openf1_rate_limited", extra={"wait": wait})
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"OpenF1 rate limit not resolved after retries: {endpoint}")


def fetch_race_sessions(year: int, n: int) -> list[dict]:
    today = str(date.today())
    sessions = _get("sessions", year=year, session_name="Race")
    sessions = [s for s in sessions if s.get("date_start", "")[:10] <= today]
    sessions.sort(key=lambda s: s["date_start"], reverse=True)
    return sessions[:n]


def _build_document(session: dict) -> KnowledgeDocument:
    sk = session["session_key"]
    location = session.get("location", "Unknown")
    country = session.get("country_name", "")
    circuit = session.get("circuit_short_name", location)
    raw_date = session.get("date_start", "")
    race_date = raw_date[:10]
    race_year = raw_date[:4]

    # Sequential fetches — OpenF1 rate-limits aggressive parallel bursts
    drivers = _get("drivers",      session_key=sk)
    stints  = _get("stints",       session_key=sk)
    weather = _get("weather",      session_key=sk)
    pits    = _get("pit",          session_key=sk)
    rc      = _get("race_control", session_key=sk)

    driver_map = {d["driver_number"]: d.get("name_acronym", str(d["driver_number"])) for d in drivers}

    # ── Weather ────────────────────────────────────────────────────────
    if weather:
        tt = [w["track_temperature"] for w in weather if w.get("track_temperature") is not None]
        at = [w["air_temperature"]   for w in weather if w.get("air_temperature")   is not None]
        rain = any(w.get("rainfall", 0) and w["rainfall"] > 0 for w in weather)
        humidity = [w["humidity"] for w in weather if w.get("humidity") is not None]
        weather_text = (
            f"Track temp: {min(tt):.0f}–{max(tt):.0f}°C"
            + (f" | Air: {sum(at)/len(at):.0f}°C" if at else "")
            + (f" | Humidity: {sum(humidity)/len(humidity):.0f}%" if humidity else "")
            + f" | Rain: {'Yes' if rain else 'No'}"
        )
        wet_race = rain
    else:
        weather_text = "No weather data available."
        wet_race = False

    # ── Tire strategy ──────────────────────────────────────────────────
    by_driver: dict[int, list[dict]] = {}
    for s in stints:
        by_driver.setdefault(s["driver_number"], []).append(s)

    strategy_lines = []
    for drv_num, drv_stints in sorted(by_driver.items()):
        drv_stints.sort(key=lambda x: x["stint_number"])
        acr = driver_map.get(drv_num, str(drv_num))
        parts = []
        for s in drv_stints:
            c = s.get("compound", "?")
            l0 = s.get("lap_start", "?")
            l1 = s.get("lap_end", "?")
            age = s.get("tyre_age_at_start", 0) or 0
            age_tag = f" (used {age}L)" if age > 0 else ""
            parts.append(f"{c} L{l0}–{l1}{age_tag}")
        stops = max(0, len(drv_stints) - 1)
        strategy_lines.append(f"  {acr}: {' → '.join(parts)} | {stops} stop{'s' if stops != 1 else ''}")

    # ── Pit stops ──────────────────────────────────────────────────────
    valid_pits = [p for p in pits if p.get("pit_duration") and 1.5 < p["pit_duration"] < 120]
    if valid_pits:
        durs = [p["pit_duration"] for p in valid_pits]
        fastest_pit = min(durs)
        avg_pit     = sum(durs) / len(durs)
        pit_text = f"{len(valid_pits)} stops | Fastest: {fastest_pit:.2f}s | Average: {avg_pit:.2f}s"
        # Unusually long pit = likely wet-weather tyre change complexity
        if avg_pit > 25:
            pit_text += " (extended — likely wet/safety-car conditions)"
    else:
        pit_text = "No valid pit stop data."

    # ── Race control events ────────────────────────────────────────────
    notable_flags = {"SAFETY_CAR", "VIRTUAL_SAFETY_CAR", "RED_FLAG"}
    rc_events = [
        e for e in rc
        if e.get("flag") in notable_flags
        or "SAFETY CAR" in str(e.get("message", "")).upper()
        or "RED FLAG" in str(e.get("message", "")).upper()
    ]
    if rc_events:
        rc_lines = [
            f"  L{e.get('lap_number','?')}: {e.get('flag','')} — {e.get('message','')[:120]}"
            for e in rc_events[:8]
        ]
    else:
        rc_lines = ["  No safety car or red flag events."]

    # ── Compounds used ─────────────────────────────────────────────────
    compounds_used = sorted({s.get("compound", "?") for s in stints if s.get("compound")})

    # ── Assemble document ──────────────────────────────────────────────
    title = f"{race_year} {location} Grand Prix"

    sections = [
        f"# {title}",
        f"Date: {race_date} | Circuit: {circuit} | {location}, {country}",
        f"Session key: {sk} | Wet race: {'Yes' if wet_race else 'No'}",
        f"Compounds used: {', '.join(compounds_used)}",
        "",
        "## Weather Conditions",
        weather_text,
        "",
        "## Tire Strategy by Driver",
    ]
    sections += strategy_lines or ["  No stint data available."]
    sections += [
        "",
        "## Pit Stop Summary",
        pit_text,
        "",
        "## Race Control Events",
    ]
    sections += rc_lines

    return KnowledgeDocument(
        source_id=f"openf1_{sk}",
        title=title,
        text="\n".join(sections),
        metadata={
            "track_id": circuit.lower().replace(" ", "_"),
            "year": race_year,
            "source": "openf1",
            "session_key": str(sk),
        },
    )


def ingest(retriever, *, years: list[int] | None = None, n_per_year: int = 8) -> list[str]:
    if years is None:
        current = date.today().year
        years = [current, current - 1]

    all_sessions: list[dict] = []
    for yr in years:
        try:
            sessions = fetch_race_sessions(yr, n_per_year)
            all_sessions.extend(sessions)
            logger.info("openf1_sessions_fetched", extra={"year": yr, "count": len(sessions)})
        except Exception as exc:
            logger.warning("openf1_fetch_failed", extra={"year": yr, "error": str(exc)})

    docs: list[KnowledgeDocument] = []
    ingested: list[str] = []
    errors: list[str] = []

    def process(session: dict):
        loc = session.get("location", "?")
        try:
            doc = _build_document(session)
            return doc, None
        except Exception as exc:
            return None, f"{loc}: {exc}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(process, s): s for s in all_sessions}
        for fut in as_completed(futures):
            doc, err = fut.result()
            if doc:
                docs.append(doc)
                ingested.append(doc.title)
            elif err:
                errors.append(err)

    if docs:
        retriever.add_documents(docs)

    for err in errors:
        logger.warning("openf1_session_skipped", extra={"error": err})

    return ingested
