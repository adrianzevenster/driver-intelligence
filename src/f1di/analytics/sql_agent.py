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


class SQLAgent:
    def __init__(self, warehouse: TelemetryWarehouse | None = None) -> None:
        self.warehouse = warehouse or TelemetryWarehouse()

    def answer(self, question: str) -> dict[str, Any]:
        start = time.perf_counter()
        schema = self.warehouse.schema_info()
        sql: str | None = None
        results: list[dict] = []
        error: str | None = None

        if settings.llm_backend in {"openai_compatible", "anthropic"}:
            try:
                sql = self._generate_sql(question, schema)
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
            "model": settings.llm_open_source_model if settings.llm_backend == "openai_compatible" else settings.llm_advice_model,
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
