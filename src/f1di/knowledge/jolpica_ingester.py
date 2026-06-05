from __future__ import annotations

import logging
import time
from datetime import date

import httpx

from f1di.knowledge.track_ids import canonical as canonical_track_id
from f1di.rag.store import KnowledgeDocument

logger = logging.getLogger(__name__)

_BASE = "https://api.jolpi.ca/ergast/f1"
_TIMEOUT = 20.0


def _get(path: str, **params) -> dict:
    for attempt in range(4):
        r = httpx.get(f"{_BASE}/{path}", params=params, timeout=_TIMEOUT)
        if r.status_code == 429:
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Jolpica rate limit not resolved after retries: {path}")


def _build_documents(year: int, n: int) -> list[KnowledgeDocument]:
    data = _get(f"{year}/results.json", limit=100)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    # Most recent n races only
    races = races[-n:] if len(races) > n else races

    docs: list[KnowledgeDocument] = []
    for race in races:
        rnd = race.get("round", "?")
        race_name = race.get("raceName", f"Round {rnd}")
        race_date = race.get("date", "")
        circuit = race.get("Circuit", {})
        circuit_id = canonical_track_id(circuit.get("circuitId", "unknown"))
        circuit_name = circuit.get("circuitName", circuit_id)
        location = circuit.get("Location", {})
        locality = location.get("locality", "")
        country = location.get("country", "")

        results = race.get("Results", [])

        # ── Finishing order ─────────────────────────────────────────────
        finish_lines: list[str] = []
        dnf_lines: list[str] = []
        fastest_lap_text = ""
        for r in results:
            pos = r.get("position", "?")
            drv = r.get("Driver", {})
            code = drv.get("code", drv.get("driverId", "?").upper()[:3])
            team = r.get("Constructor", {}).get("name", "?")
            status = r.get("status", "?")
            laps = r.get("laps", "?")
            points = r.get("points", "0")

            time_info = r.get("Time", {})
            gap = time_info.get("time", "") if time_info else ""

            fl = r.get("FastestLap", {})
            if fl and fl.get("rank") == "1":
                fl_time = fl.get("Time", {}).get("time", "?")
                fl_lap = fl.get("lap", "?")
                fastest_lap_text = f"{code} — {fl_time} (L{fl_lap})"

            if status == "Finished" or "Lap" in status:
                finish_lines.append(
                    f"  P{pos} {code} ({team}) | {laps} laps | +{gap} | {points} pts"
                )
            else:
                dnf_lines.append(f"  P{pos} {code} ({team}) — {status}")

        title = f"{year} {race_name}"
        sections = [
            f"# {title}",
            f"Circuit: {circuit_name} | {locality}, {country} | Date: {race_date}",
            "",
            "## Race Result",
        ]
        sections += finish_lines[:10] or ["  No result data."]
        if dnf_lines:
            sections += ["", "## DNFs / Retirements"]
            sections += dnf_lines
        if fastest_lap_text:
            sections += ["", "## Fastest Lap", f"  {fastest_lap_text}"]

        docs.append(KnowledgeDocument(
            source_id=f"jolpica_{year}_{rnd}",
            title=title,
            text="\n".join(sections),
            metadata={
                "track_id": circuit_id,
                "year": str(year),
                "source": "jolpica",
                "round": str(rnd),
            },
        ))

    return docs


def ingest(retriever, *, years: list[int] | None = None, n_per_year: int = 8) -> list[str]:
    if years is None:
        current = date.today().year
        years = [current, current - 1]

    all_docs: list[KnowledgeDocument] = []
    ingested: list[str] = []

    for yr in years:
        try:
            docs = _build_documents(yr, n_per_year)
            all_docs.extend(docs)
            ingested.extend(d.title for d in docs)
            logger.info("jolpica_races_fetched", extra={"year": yr, "count": len(docs)})
        except Exception as exc:
            logger.warning("jolpica_fetch_failed", extra={"year": yr, "error": str(exc)})

    if all_docs:
        retriever.add_documents(all_docs)

    return ingested
