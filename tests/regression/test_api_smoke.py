from __future__ import annotations

import json
from pathlib import Path

import pytest

from f1di.domain.schemas import TelemetryWindow


def _fixture_window() -> TelemetryWindow:
    cases = json.loads(Path("data/fixtures/real_replay_eval.json").read_text(encoding="utf-8"))
    return TelemetryWindow.model_validate(cases[0]["window"])


def test_api_insight_and_session_routes_smoke(monkeypatch):
    from f1di.api.main import (
        app_version,
        create_insight,
        get_orchestrator,
        health,
        session_insight,
        session_trace,
    )

    get_orchestrator.cache_clear()

    window = _fixture_window()

    def fake_trace(*, year: int, round_num: int, driver: str, lap_number: int, session_type: str = "R"):
        return [{"dist": 0.0, "speed": 280.0, "throttle": 90.0, "brake": False, "drs": True}]

    def fake_build_window(*, year: int, round_num: int, driver: str, lap_number: int | None = None, session_type: str = "R"):
        return window

    monkeypatch.setattr("f1di.knowledge.fastf1_session.get_lap_trace", fake_trace)
    monkeypatch.setattr("f1di.knowledge.fastf1_session.build_window", fake_build_window)

    assert health() == {"status": "ok"}
    assert app_version()["name"] == "f1-driver-intelligence"

    insight = create_insight(window)
    assert insight.session_id == window.session_id
    assert insight.findings

    trace = session_trace(2024, 1, "VER", 10)
    assert trace[0]["speed"] == 280.0

    replay_insight = session_insight(year=2024, round_num=1, driver="VER", lap_number=10)
    assert replay_insight.session_id == window.session_id


def test_spa_static_fallback_smoke_if_frontend_is_built():
    from f1di.api.main import app

    if not Path("frontend/dist/index.html").exists():
        pytest.skip("frontend dist is not built")

    route_paths = {getattr(route, "path", None) for route in app.routes}
    mounted_paths = {getattr(route, "path", None) for route in app.routes if getattr(route, "name", "") == "assets"}
    assert "/{full_path:path}" in route_paths
    assert "/assets" in mounted_paths
