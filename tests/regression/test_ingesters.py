"""Lightweight ingester tests — mock network calls, assert KnowledgeDocument shape."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from f1di.knowledge.track_ids import canonical


# ── track_ids ──────────────────────────────────────────────────────────────

def test_canonical_spa():
    assert canonical("Spa-Francorchamps") == "spa"
    assert canonical("spa") == "spa"

def test_canonical_monaco():
    assert canonical("monte_carlo") == "monaco"
    assert canonical("Monaco") == "monaco"

def test_canonical_melbourne():
    assert canonical("albert_park") == "melbourne"
    assert canonical("Melbourne") == "melbourne"

def test_canonical_abu_dhabi():
    assert canonical("yas_marina") == "abu_dhabi"
    assert canonical("Yas Island") == "abu_dhabi"

def test_canonical_singapore():
    assert canonical("marina_bay") == "singapore"
    assert canonical("Marina Bay") == "singapore"

def test_canonical_austin():
    assert canonical("americas") == "austin"
    assert canonical("Austin") == "austin"

def test_canonical_interlagos():
    assert canonical("interlagos") == "interlagos"
    assert canonical("São Paulo") == "interlagos"


# ── jolpica_ingester ───────────────────────────────────────────────────────

def _fake_jolpica_response():
    return {
        "MRData": {
            "RaceTable": {
                "Races": [
                    {
                        "round": "1",
                        "raceName": "Bahrain Grand Prix",
                        "date": "2024-03-02",
                        "Circuit": {
                            "circuitId": "bahrain",
                            "circuitName": "Bahrain International Circuit",
                            "Location": {"locality": "Sakhir", "country": "Bahrain"},
                        },
                        "Results": [
                            {
                                "position": "1",
                                "Driver": {"code": "VER"},
                                "Constructor": {"name": "Red Bull"},
                                "laps": "57",
                                "status": "Finished",
                                "Time": {"time": "1:31:44.742"},
                                "FastestLap": {"rank": "1", "lap": "51", "Time": {"time": "1:32.608"}},
                                "points": "25",
                            },
                            {
                                "position": "2",
                                "Driver": {"code": "SAI"},
                                "Constructor": {"name": "Ferrari"},
                                "laps": "57",
                                "status": "Finished",
                                "Time": {},
                                "FastestLap": {"rank": "2", "lap": "50", "Time": {"time": "1:33.1"}},
                                "points": "18",
                            },
                        ],
                    }
                ]
            }
        }
    }


def test_jolpica_builds_document():
    from f1di.knowledge.jolpica_ingester import _build_documents
    from f1di.rag.store import KnowledgeDocument

    with patch("f1di.knowledge.jolpica_ingester._get", return_value=_fake_jolpica_response()):
        docs = _build_documents(2024, 1)

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, KnowledgeDocument)
    assert doc.source_id == "jolpica_2024_1"
    assert doc.metadata["source"] == "jolpica"
    assert doc.metadata["track_id"] == "bahrain"
    assert "VER" in doc.text
    assert "Bahrain Grand Prix" in doc.title


def test_jolpica_ingest_calls_retriever():
    from f1di.knowledge.jolpica_ingester import ingest

    retriever = MagicMock()
    with patch("f1di.knowledge.jolpica_ingester._get", return_value=_fake_jolpica_response()):
        result = ingest(retriever, years=[2024], n_per_year=1)

    assert retriever.add_documents.called
    assert len(result) == 1
    assert result[0] == "2024 Bahrain Grand Prix"


# ── fastf1_ingester ────────────────────────────────────────────────────────

def _make_fake_session(location="Silverstone", laps_df=None, weather_df=None):
    import pandas as pd

    session = MagicMock()
    session.event = {
        "OfficialEventName": "Formula 1 British Grand Prix 2024",
        "Location": location,
        "Country": "United Kingdom",
        "EventDate": "2024-07-07",
    }

    if laps_df is None:
        laps_df = pd.DataFrame({
            "Driver": ["VER", "VER", "HAM", "HAM"],
            "LapNumber": [1, 2, 1, 2],
            "LapTime": pd.to_timedelta(["0:01:28.234", "0:01:28.4", "0:01:28.9", "0:01:29.1"]),
            "Sector1Time": pd.to_timedelta(["0:00:27.1"] * 4),
            "Sector2Time": pd.to_timedelta(["0:00:31.5"] * 4),
            "Sector3Time": pd.to_timedelta(["0:00:29.6"] * 4),
            "Compound": ["MEDIUM"] * 4,
            "TyreLife": [5, 6, 5, 6],
            "Stint": [1, 1, 1, 1],
            "IsAccurate": [True, True, True, True],
        })

    if weather_df is None:
        weather_df = pd.DataFrame({
            "TrackTemp": [38.0, 39.0],
            "AirTemp": [25.0, 26.0],
            "Rainfall": [False, False],
            "WindSpeed": [12.0, 13.0],
        })

    session.laps = laps_df
    session.weather_data = weather_df
    return session


def test_fastf1_builds_document():
    import sys
    from f1di.knowledge.fastf1_ingester import _build_document
    from f1di.rag.store import KnowledgeDocument

    fastf1_mock = MagicMock()
    fastf1_mock.get_session.return_value = _make_fake_session("Silverstone")
    with patch.dict(sys.modules, {"fastf1": fastf1_mock}), patch("os.makedirs"):
        doc = _build_document(2024, "British Grand Prix", 12)

    assert isinstance(doc, KnowledgeDocument)
    assert doc.source_id == "fastf1_2024_12"
    assert doc.metadata["source"] == "fastf1"
    assert doc.metadata["track_id"] == "silverstone"
    assert "VER" in doc.text
    assert "MEDIUM" in doc.text


def test_fastf1_normalises_spa_track_id():
    import sys
    from f1di.knowledge.fastf1_ingester import _build_document

    fastf1_mock = MagicMock()
    fastf1_mock.get_session.return_value = _make_fake_session("Spa-Francorchamps")
    with patch.dict(sys.modules, {"fastf1": fastf1_mock}), patch("os.makedirs"):
        doc = _build_document(2024, "Belgian Grand Prix", 14)

    assert doc.metadata["track_id"] == "spa"


# ── openf1_live ────────────────────────────────────────────────────────────

_FAKE_OPENF1 = {
    "sessions": [
        {"session_key": 9158, "session_name": "Race", "session_type": "Race",
         "meeting_name": "British Grand Prix", "location": "Silverstone",
         "country_name": "United Kingdom", "circuit_short_name": "Silverstone",
         "year": 2024, "date_start": "2024-07-07T14:00:00+00:00"},
        {"session_key": 9100, "session_name": "Race", "session_type": "Race",
         "meeting_name": "Bahrain Grand Prix", "location": "Sakhir",
         "country_name": "Bahrain", "circuit_short_name": "Bahrain",
         "year": 2024, "date_start": "2024-03-02T15:00:00+00:00"},
    ],
    "drivers": [
        {"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"},
        {"driver_number": 44, "name_acronym": "HAM", "team_name": "Mercedes"},
    ],
    "car_data": [
        {"brake": 0, "date": "2024-07-07T14:01:00+00:00", "driver_number": 1,
         "drs": 8, "n_gear": 8, "rpm": 11000, "session_key": 9158, "speed": 295, "throttle": 98},
        {"brake": 1, "date": "2024-07-07T14:01:04+00:00", "driver_number": 1,
         "drs": 0, "n_gear": 7, "rpm": 9500, "session_key": 9158, "speed": 180, "throttle": 5},
    ],
    "stints": [
        {"compound": "MEDIUM", "driver_number": 1, "lap_start": 1, "lap_end": 30,
         "session_key": 9158, "stint_number": 1, "tyre_age_at_start": 0},
    ],
    "weather": [
        {"air_temperature": 24.0, "date": "2024-07-07T14:00:00+00:00", "humidity": 55.0,
         "rainfall": False, "session_key": 9158, "track_temperature": 38.0,
         "wind_direction": 180, "wind_speed": 3.5},
    ],
    "laps": [
        {"driver_number": 1, "lap_number": 12, "session_key": 9158,
         "date_start": "2024-07-07T14:00:55+00:00", "lap_duration": 88.4},
        {"driver_number": 1, "lap_number": 13, "session_key": 9158,
         "date_start": "2024-07-07T14:02:23+00:00", "lap_duration": 88.2},
    ],
}


def _fake_openf1_get(path, **_params):
    return _FAKE_OPENF1.get(path, [])


def test_openf1_live_builds_window():
    from f1di.domain.schemas import TelemetryWindow
    from f1di.knowledge.openf1_live import build_window

    with patch("f1di.knowledge.openf1_live._get", side_effect=_fake_openf1_get):
        window = build_window(session_key=9158, driver_number=1)

    assert isinstance(window, TelemetryWindow)
    assert window.track_id == "silverstone"
    assert window.driver_id == "1"
    assert len(window.samples) == 2
    assert window.latest.compound.value == "MEDIUM"
    assert window.latest.speed_kph == 180.0


def test_openf1_live_get_sessions_sorted():
    from f1di.knowledge.openf1_live import get_sessions

    with patch("f1di.knowledge.openf1_live._get", return_value=_FAKE_OPENF1["sessions"]):
        sessions = get_sessions(year=2024)

    assert sessions[0]["session_key"] == 9158  # newest first
    assert sessions[1]["session_key"] == 9100


def test_openf1_live_replay_lap():
    from f1di.knowledge.openf1_live import build_window

    with patch("f1di.knowledge.openf1_live._get", side_effect=_fake_openf1_get):
        window = build_window(session_key=9158, driver_number=1, lap_number=12)

    assert window.latest.lap == 12
    assert window.latest.stint_lap == 11  # lap 12 - lap_start 1
