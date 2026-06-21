"""Regression tests for the post-race outcome labeler.

FastF1 network access and the database are both mocked — these tests run
fully offline and do not require a FastF1 cache or Postgres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_laps(drivers: list[str], n_laps: int = 50) -> pd.DataFrame:
    """Build a minimal synthetic laps DataFrame matching the FastF1 schema."""
    rows = []
    for drv in drivers:
        for lap in range(1, n_laps + 1):
            rows.append({
                "Driver": drv,
                "LapNumber": lap,
                "LapTime": pd.Timedelta(seconds=90 + np.random.uniform(-2, 2)),
                "Stint": 1 if lap <= 25 else 2,
                "Compound": "MEDIUM" if lap <= 25 else "HARD",
                "TyreLife": (lap % 25) + 1,
            })
    return pd.DataFrame(rows)


def _make_session(drivers: list[str], n_laps: int = 50) -> MagicMock:
    session = MagicMock()
    session.laps = _make_laps(drivers, n_laps)
    session.weather_data = None
    session.event = {"Location": "Silverstone"}
    return session


# ── _extract_incidents ─────────────────────────────────────────────────────


def test_extract_incidents_no_retirements():
    from f1di.data.outcome_labeler import _extract_incidents
    session = _make_session(["VER", "NOR", "HAM"], n_laps=50)
    incidents = _extract_incidents(session)
    retirement_drivers = {i.driver for i in incidents if i.incident_type == "retirement"}
    assert len(retirement_drivers) == 0, "No retirements expected when all drivers finish"


def test_extract_incidents_detects_retirement():
    from f1di.data.outcome_labeler import _extract_incidents

    laps = _make_laps(["VER", "NOR"], n_laps=50)
    # Truncate NOR to lap 20 — clearly retired early
    laps = laps[~((laps["Driver"] == "NOR") & (laps["LapNumber"] > 20))].copy()

    session = MagicMock()
    session.laps = laps
    session.weather_data = None
    session.event = {"Location": "Monaco"}

    incidents = _extract_incidents(session)
    retirement_drivers = {i.driver for i in incidents if i.incident_type == "retirement"}
    assert "NOR" in retirement_drivers


def test_extract_incidents_detects_safety_car():
    from f1di.data.outcome_labeler import _extract_incidents

    laps = _make_laps(["VER", "NOR", "HAM", "RUS", "LEC"], n_laps=50)
    # Introduce a large lap-time spike at lap 30 for ALL drivers → safety car
    spike_idx = laps["LapNumber"] == 30
    laps.loc[spike_idx, "LapTime"] = pd.Timedelta(seconds=130)

    session = MagicMock()
    session.laps = laps
    session.weather_data = None
    session.event = {"Location": "Spa"}

    incidents = _extract_incidents(session)
    sc_incidents = [i for i in incidents if i.incident_type == "safety_car"]
    assert len(sc_incidents) > 0


def test_extract_incidents_falls_back_when_race_control_messages_unloaded():
    from f1di.data.outcome_labeler import _extract_incidents

    class SessionWithUnloadedMessages:
        event = {"Location": "Spa"}

        def __init__(self, laps):
            self.laps = laps

        @property
        def race_control_messages(self):
            raise RuntimeError("The data you are trying to access has not been loaded yet. See `Session.load`")

    laps = _make_laps(["VER", "NOR", "HAM", "RUS", "LEC"], n_laps=50)
    laps.loc[laps["LapNumber"] == 30, "LapTime"] = pd.Timedelta(seconds=130)

    incidents = _extract_incidents(SessionWithUnloadedMessages(laps))

    assert any(i.incident_type == "safety_car" for i in incidents)


def test_extract_incidents_detects_forced_pit():
    from f1di.data.outcome_labeler import _extract_incidents

    drivers = ["VER", "NOR"]
    rows = []
    for drv in drivers:
        for lap in range(1, 51):
            rows.append({
                "Driver": drv,
                "LapNumber": lap,
                "LapTime": pd.Timedelta(seconds=90),
                "Stint": 1,
                "Compound": "MEDIUM",
                # TyreLife only 3 laps before pit — far below 30% of MEDIUM expected (26 laps)
                "TyreLife": lap if lap <= 3 else 3,
            })
    laps = pd.DataFrame(rows)

    session = MagicMock()
    session.laps = laps
    session.weather_data = None
    session.event = {"Location": "Bahrain"}

    incidents = _extract_incidents(session)
    forced_pit_drivers = {i.driver for i in incidents if i.incident_type == "forced_pit"}
    # Both drivers have extremely short stints
    assert len(forced_pit_drivers) > 0


def test_extract_incidents_detects_lockup_proxy():
    from f1di.data.outcome_labeler import _extract_incidents

    rows = []
    # VER: laps 20→21→22 each grow >1.5s over the previous — triggers lockup_proxy
    # The check fires at index i when lt[i]-lt[i-1] > 1.5 AND lt[i-1]-lt[i-2] > 1.5
    # i.e. it needs two back-to-back worsening steps both > 1.5s.
    times = {20: 90.0, 21: 91.7, 22: 93.4}  # each step +1.7s
    for lap in range(1, 51):
        lt = pd.Timedelta(seconds=times.get(lap, 90.0))
        rows.append({"Driver": "VER", "LapNumber": lap, "LapTime": lt,
                     "Stint": 1, "Compound": "SOFT", "TyreLife": lap})
    laps = pd.DataFrame(rows)

    session = MagicMock()
    session.laps = laps
    session.weather_data = None
    session.event = {"Location": "Silverstone"}

    incidents = _extract_incidents(session)
    lockup_drivers = {i.driver for i in incidents if i.incident_type == "lockup_proxy"}
    assert "VER" in lockup_drivers


def test_extract_incidents_empty_laps():
    from f1di.data.outcome_labeler import _extract_incidents

    session = MagicMock()
    session.laps = pd.DataFrame()
    session.weather_data = None
    session.event = {"Location": "Silverstone"}

    incidents = _extract_incidents(session)
    assert incidents == []


# ── label_race dry_run ─────────────────────────────────────────────────────


def test_label_race_db_unavailable_returns_partial_report():
    """When storage modules are unavailable, label_race falls back gracefully."""
    import sys

    session = _make_session(["VER", "NOR"], n_laps=50)
    fastf1_mock = MagicMock()
    fastf1_mock.get_session.return_value = session
    fastf1_mock.Cache.enable_cache = MagicMock()

    track_ids_mock = MagicMock()
    track_ids_mock.canonical.return_value = "silverstone"

    # Replace storage modules with None so from-imports inside label_race fail
    storage_modules = {
        "f1di.storage.database": None,
        "f1di.storage.models": None,
    }

    with (
        patch.dict(sys.modules, {"fastf1": fastf1_mock, **storage_modules}),
        patch.dict(sys.modules, {"f1di.knowledge.track_ids": track_ids_mock}),
    ):
        from importlib import reload
        import f1di.data.outcome_labeler as labeler
        reload(labeler)
        report = labeler.label_race(year=2024, round_num=5, dry_run=True)

    assert report.year == 2024
    assert report.round_num == 5
    assert report.n_insights_examined == 0
    assert isinstance(report.incidents_found, list)


def test_label_race_fastf1_unavailable_returns_empty_report():
    """When fastf1 is not importable, label_race returns a zero-count report."""
    import sys

    saved = sys.modules.get("fastf1")
    sys.modules["fastf1"] = None  # type: ignore[assignment]
    try:
        from importlib import reload
        import f1di.data.outcome_labeler as labeler_mod
        reload(labeler_mod)

        report = labeler_mod.label_race(year=2024, round_num=5)
        assert report.n_insights_examined == 0
        assert report.track_id in ("unknown", "silverstone", "")
    finally:
        if saved is None:
            sys.modules.pop("fastf1", None)
        else:
            sys.modules["fastf1"] = saved


# ── OutcomeReport dataclass ────────────────────────────────────────────────


def test_outcome_report_asdict_is_serializable():
    from f1di.data.outcome_labeler import OutcomeReport
    from dataclasses import asdict
    import json

    report = OutcomeReport(
        year=2024,
        round_num=3,
        track_id="monaco",
        n_insights_examined=10,
        n_labeled_correct=6,
        n_labeled_incorrect=3,
        n_no_match=1,
        incidents_found=[{"driver": "VER", "lap": 22, "type": "retirement", "severity": 0.95}],
    )
    # Must be JSON-serializable (for the API endpoint)
    json.dumps(asdict(report))


def test_outcome_label_endpoint_serializes_numpy_scalars():
    import numpy as np

    from f1di.api.main import label_race_outcomes
    from f1di.data.outcome_labeler import OutcomeReport

    report = OutcomeReport(
        year=2026,
        round_num=np.int64(1),
        track_id="melbourne",
        n_insights_examined=np.int64(2),
        n_labeled_correct=np.int64(1),
        n_labeled_incorrect=np.int64(1),
        n_no_match=np.int64(0),
        incidents_found=[
            {
                "driver": "VER",
                "lap": np.int64(12),
                "type": "safety_car",
                "severity": np.float64(0.7),
            }
        ],
    )

    with patch("f1di.data.outcome_labeler.label_race", return_value=report):
        result = label_race_outcomes(year=2026, round_num=1, dry_run=True)

    assert result["round_num"] == 1
    assert isinstance(result["round_num"], int)
    assert isinstance(result["n_insights_examined"], int)
    assert isinstance(result["incidents_found"][0]["lap"], int)
    assert isinstance(result["incidents_found"][0]["severity"], float)
