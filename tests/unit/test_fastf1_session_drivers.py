from __future__ import annotations

import pandas as pd

from f1di.knowledge import fastf1_session


class _Session:
    def __init__(self, laps=None, results=None, drivers=None, driver_info=None):
        self.laps = laps
        self.results = results
        self.drivers = drivers or []
        self._driver_info = driver_info or {}

    def get_driver(self, number):
        return self._driver_info[number]


def test_get_drivers_prefers_lap_driver_codes(monkeypatch):
    session = _Session(laps=pd.DataFrame({"Driver": ["VER", "NOR", "VER"]}))
    monkeypatch.setattr(fastf1_session, "_load_session_laps", lambda *args: session)

    assert fastf1_session.get_drivers(2025, 10) == [{"code": "NOR"}, {"code": "VER"}]


def test_get_drivers_falls_back_to_results_metadata(monkeypatch):
    session = _Session(
        laps=pd.DataFrame(),
        results=pd.DataFrame({"Abbreviation": ["RUS", "VER", "RUS"]}),
    )
    monkeypatch.setattr(fastf1_session, "_load_session_laps", lambda *args: session)

    assert fastf1_session.get_drivers(2025, 10) == [{"code": "RUS"}, {"code": "VER"}]


def test_get_drivers_static_grid_is_opt_in(monkeypatch):
    session = _Session(laps=pd.DataFrame(), results=pd.DataFrame())
    monkeypatch.setattr(fastf1_session, "_load_session_laps", lambda *args: session)

    assert fastf1_session.get_drivers(2025, 10) == []

    rows = fastf1_session.get_drivers(2025, 10, allow_fallback=True)
    assert {"code": "BOR", "name": "Gabriel Bortoleto", "source": "fallback_grid"} in rows
    assert {"code": "HAD", "name": "Isack Hadjar", "source": "fallback_grid"} in rows
