"""Text-to-SQL agent backed by the DuckDB telemetry warehouse.

Converts a natural-language question into a SQL query using the locally-running
Ollama server (or any OpenAI-compatible endpoint).  No API keys required.

Set ``F1DI_LLM_BACKEND=openai_compatible`` and start Ollama with a code-capable
model (e.g. ``ollama pull qwen2.5-coder:7b``) to enable automatic SQL generation.
When the LLM is unavailable the agent returns the schema + sample queries so the
analyst can write SQL directly.

Example:
    agent = SQLAgent()
    result = agent.answer("Who had the most CRITICAL tire warnings at Monaco?")
    print(result["sql"])     # generated SELECT ...
    print(result["results"]) # [{"driver_id": "VER", "count": 4}, ...]
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from f1di.analytics.warehouse import TelemetryWarehouse
from f1di.config.settings import settings

logger = logging.getLogger("f1di.analytics.sql_agent")

_SYSTEM_PROMPT = """\
You are an expert SQL analyst for a Formula 1 telemetry platform.
You speak DuckDB SQL dialect (window functions, ASOF JOIN, PIVOT, LIST_AGG supported).

Database schema:
{schema}

Rules:
1. Respond with a single valid SQL SELECT statement only.
2. No markdown fences, no explanation text — just the SQL.
3. Limit to 100 rows unless the question explicitly asks for more.
4. Never use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE, ATTACH, or DETACH.
"""

_USER_TEMPLATE = "Question: {question}"


def _extract_sql(text: str) -> str:
    clean = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip().rstrip("`").strip()
    return clean.split(";")[0].strip() + ";"


def _limit_from_question(question: str, default: int = 10) -> int:
    match = re.search(r"\b(?:top|first|latest|recent|last|list|show)?\s*(\d{1,3})\b", question)
    if not match:
        return default
    return min(100, max(1, int(match.group(1))))


def _fallback_sql(question: str) -> tuple[str | None, str | None]:
    """Return SQL for common analytics questions that should not require an LLM."""
    q = re.sub(r"\s+", " ", question.strip().lower())
    limit = _limit_from_question(q)

    risk_match = re.search(r"\b(info|watch|warning|critical)s?\b", q)
    risk = risk_match.group(1).upper() if risk_match else None
    mentions_insights = any(term in q for term in ("insight", "signal", "alert", "warning"))

    if (
        risk
        and mentions_insights
        and any(term in q for term in ("recent", "latest", "last", "list"))
    ):
        return (
            "SELECT insight_id, session_id, driver_id, track_id, lap, risk, "
            "ROUND(confidence, 4) AS confidence, policy, recommendation, created_at "
            f"FROM insights WHERE risk='{risk}' "
            f"ORDER BY created_at DESC LIMIT {limit};",
            "template",
        )

    if risk and mentions_insights and "per driver" in q:
        where = f"WHERE risk='{risk}'"
        if "week" in q:
            where += " AND created_at >= CAST(CURRENT_DATE - INTERVAL 7 DAY AS TEXT)"
        return (
            "SELECT driver_id, COUNT(*) AS insight_count "
            f"FROM insights {where} "
            "GROUP BY driver_id ORDER BY insight_count DESC "
            f"LIMIT {limit};",
            "template",
        )

    if "average" in q and "confidence" in q and "risk" in q:
        return (
            "SELECT risk, ROUND(AVG(confidence), 4) AS avg_confidence, COUNT(*) AS n "
            "FROM insights GROUP BY risk ORDER BY avg_confidence DESC LIMIT 100;",
            "template",
        )

    if "track" in q and "confidence" in q and ("highest" in q or "average" in q):
        return (
            "SELECT track_id, ROUND(AVG(confidence), 4) AS avg_confidence, COUNT(*) AS n "
            "FROM insights GROUP BY track_id ORDER BY avg_confidence DESC "
            f"LIMIT {limit};",
            "template",
        )

    if "tire wear" in q and "compound" in q:
        return (
            "SELECT compound, "
            "ROUND(AVG(tire_wear_fl), 4) AS avg_wear_fl, "
            "ROUND(AVG(tire_wear_fr), 4) AS avg_wear_fr, "
            "ROUND(AVG(tire_wear_rl), 4) AS avg_wear_rl, "
            "ROUND(AVG(tire_wear_rr), 4) AS avg_wear_rr, "
            "COUNT(*) AS samples "
            "FROM telemetry GROUP BY compound ORDER BY samples DESC LIMIT 100;",
            "template",
        )

    return None, None


class SQLAgent:
    def __init__(self, warehouse: TelemetryWarehouse | None = None) -> None:
        self.warehouse = warehouse or TelemetryWarehouse()

    def answer(self, question: str) -> dict[str, Any]:
        start = time.perf_counter()
        schema = self.warehouse.schema_info()
        sql: str | None = None
        results: list[dict] = []
        error: str | None = None
        model: str | None

        sql, model = _fallback_sql(question)

        if sql is None and settings.llm_backend in {"openai_compatible", "anthropic"}:
            try:
                sql = self._generate_sql(question, schema)
                model = (
                    settings.llm_open_source_model
                    if settings.llm_backend == "openai_compatible"
                    else settings.llm_advice_model
                )
            except Exception as exc:
                logger.warning("SQL generation failed: %s", exc)

        if sql is None:
            return {
                "question": question,
                "sql": None,
                "results": [],
                "schema": schema,
                "sample_queries": self.warehouse.sample_queries(),
                "error": (
                    "LLM unavailable. Set F1DI_LLM_BACKEND=openai_compatible and start Ollama "
                    "(e.g. `ollama run qwen2.5-coder:7b`) to enable automatic SQL generation."
                ),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        try:
            results = self.warehouse.query(sql)
        except Exception as exc:
            error = str(exc)
            logger.warning("SQL execution failed for %r: %s", sql, exc)

        return {
            "question": question,
            "sql": sql,
            "model": model,
            "results": results,
            "row_count": len(results),
            "error": error,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    def _generate_sql(self, question: str, schema: str) -> str:
        system = _SYSTEM_PROMPT.format(schema=schema)
        user = _USER_TEMPLATE.format(question=question)

        if settings.llm_backend == "openai_compatible":
            return self._ollama_sql(system, user)
        if settings.llm_backend == "anthropic":
            return self._anthropic_sql(system, user)
        raise RuntimeError("No LLM backend configured")

    def _ollama_sql(self, system: str, user: str) -> str:
        import httpx

        payload = {
            "model": settings.llm_open_source_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {settings.llm_api_key or 'ollama'}"}
        resp = httpx.post(
            f"{settings.llm_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        return _extract_sql(resp.json()["choices"][0]["message"]["content"])

    def _anthropic_sql(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.llm_advice_model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _extract_sql(msg.content[0].text)
