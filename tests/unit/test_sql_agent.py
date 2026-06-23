from __future__ import annotations

from f1di.analytics.sql_agent import SQLAgent
from f1di.config.settings import settings


class FakeWarehouse:
    def __init__(self) -> None:
        self.last_sql = ""

    def schema_info(self) -> str:
        return "Table: insights"

    def sample_queries(self) -> list[dict[str, str]]:
        return []

    def query(self, sql: str) -> list[dict]:
        self.last_sql = sql
        return [{"driver_id": "VER", "risk": "CRITICAL"}]


def test_sql_agent_handles_recent_critical_signals_without_llm(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "rules")
    warehouse = FakeWarehouse()

    result = SQLAgent(warehouse).answer("List the most recent CRITICAL signals")

    assert result["model"] == "template"
    assert result["error"] is None
    assert result["results"] == [{"driver_id": "VER", "risk": "CRITICAL"}]
    assert "FROM insights WHERE risk='CRITICAL'" in result["sql"]
    assert "ORDER BY created_at DESC" in result["sql"]
    assert "LIMIT 10" in result["sql"]
    assert warehouse.last_sql == result["sql"]


def test_sql_agent_keeps_unmatched_questions_on_llm_error_path(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "rules")

    result = SQLAgent(FakeWarehouse()).answer("Explain the strategic tradeoffs at Monaco")

    assert result["sql"] is None
    assert result["results"] == []
    assert "LLM unavailable" in result["error"]
