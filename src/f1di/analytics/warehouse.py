from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

from f1di.config.settings import settings

_SAFE_SQL_RE = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|ATTACH|DETACH|COPY|EXPORT)\b",
    re.IGNORECASE,
)

SCHEMA_DESCRIPTION = textwrap.dedent("""
    Table: insights
    ---------------
    insight_id     TEXT   -- UUID for the inference call
    session_id     TEXT   -- race session identifier
    driver_id      TEXT   -- three-letter driver code (e.g. VER, HAM)
    track_id       TEXT   -- canonical circuit name (e.g. silverstone, monza)
    lap            INT    -- lap number when insight was generated
    compound       TEXT   -- tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
    risk           TEXT   -- INFO | WATCH | WARNING | CRITICAL
    confidence     REAL   -- calibrated confidence [0, 1]
    uncertainty    REAL   -- 1 - confidence
    policy         TEXT   -- SHOW | ENGINEER_ONLY | SUPPRESS
    audience       TEXT   -- DRIVER | ENGINEER | STRATEGY | REPLAY
    recommendation TEXT
    latency_ms     REAL
    created_at     TEXT   -- ISO timestamp

    Table: feedback
    ---------------
    insight_id     TEXT
    rating         INT    -- 1–5 stars
    correct        BOOL
    comment        TEXT
    submitted_by   TEXT
    created_at     TEXT

    Table: telemetry
    ----------------
    session_id     TEXT
    driver_id      TEXT
    track_id       TEXT
    lap            INT
    timestamp_ms   INT
    speed_kph      REAL
    throttle_pct   REAL
    brake_pressure REAL
    compound       TEXT
    stint_lap      INT
    tire_wear_fl   REAL
    tire_wear_fr   REAL
    tire_wear_rl   REAL
    tire_wear_rr   REAL
    grip_estimate  REAL
    battery_soc    REAL
    track_temp_c   REAL
    rain_intensity REAL

    Table: ingestion_runs
    ---------------------
    source         TEXT   -- fastf1 | openf1 | jolpica
    year           INT
    round_num      INT
    track_id       TEXT
    event_name     TEXT
    documents_added INT
    completed_at   TEXT
""").strip()


class TelemetryWarehouse:
    def __init__(self, storage_url: str | None = None) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "Install the analytics extra: pip install 'f1-driver-intelligence[analytics]'"
            ) from exc

        url = storage_url or settings.storage_url
        self._conn = duckdb.connect(":memory:")

        if url.startswith("sqlite:///"):
            sqlite_path = url[len("sqlite:///"):]
            abs_path = str(Path(sqlite_path).resolve())
            self._conn.execute(f"ATTACH '{abs_path}' AS f1di (TYPE sqlite, READ_ONLY true)")
            self._prefix = "f1di."
        elif url.startswith("postgresql"):
            self._conn.execute("INSTALL postgres; LOAD postgres;")
            self._conn.execute(f"ATTACH '{url}' AS f1di (TYPE postgres, READ_ONLY true)")
            self._prefix = "f1di."
        else:
            self._prefix = ""

    def query(self, sql: str) -> list[dict[str, Any]]:
        if _SAFE_SQL_RE.search(sql):
            raise ValueError("Only SELECT queries are permitted.")
        qualified = sql
        for table in ("insights", "feedback", "ingestion_runs", "telemetry"):
            qualified = re.sub(
                rf"\b{table}\b", f"{self._prefix}{table}", qualified, flags=re.IGNORECASE
            )
        rel = self._conn.execute(qualified)
        columns = [d[0] for d in rel.description]
        return [dict(zip(columns, row)) for row in rel.fetchall()]

    def schema_info(self) -> str:
        return SCHEMA_DESCRIPTION

    def sample_queries(self) -> list[dict[str, str]]:
        return [
            {
                "question": "Which driver had the most CRITICAL insights at Silverstone?",
                "sql": "SELECT driver_id, COUNT(*) AS critical_count FROM insights WHERE track_id='silverstone' AND risk='CRITICAL' GROUP BY driver_id ORDER BY critical_count DESC LIMIT 5",
            },
            {
                "question": "Average confidence by risk level",
                "sql": "SELECT risk, ROUND(AVG(confidence), 4) AS avg_conf, COUNT(*) AS n FROM insights GROUP BY risk ORDER BY avg_conf DESC",
            },
            {
                "question": "Rounds with FastF1 data ingested",
                "sql": "SELECT year, COUNT(*) AS rounds FROM ingestion_runs WHERE source='fastf1' GROUP BY year ORDER BY year DESC",
            },
            {
                "question": "Feedback accuracy rate per driver",
                "sql": "SELECT i.driver_id, ROUND(AVG(CAST(f.correct AS INT)), 4) AS accuracy, COUNT(*) AS reviews FROM feedback f JOIN insights i ON f.insight_id=i.insight_id WHERE f.correct IS NOT NULL GROUP BY i.driver_id ORDER BY accuracy DESC",
            },
        ]

    def close(self) -> None:
        self._conn.close()
