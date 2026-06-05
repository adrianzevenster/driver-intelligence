from __future__ import annotations

import json
import logging
import time

from f1di.domain.schemas import AgentFinding, InsightAudience, RiskLevel

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior Formula 1 race engineer providing real-time guidance during a race.
Your output is a single actionable recommendation based on telemetry analysis from specialist agents.
Be direct, specific, and reference the data. Never hedge unless confidence is genuinely low.

For DRIVER audience: one imperative sentence the driver acts on immediately (radio-style).
For ENGINEER or STRATEGY audience: two sentences — the key metric driving concern, then the recommended action.
Never fabricate metrics not present in the context. Never speculate beyond the data provided.

Always respond with valid JSON matching exactly: {"recommendation": "<text>"}"""


def generate_recommendation(
    *,
    risk: RiskLevel,
    findings: list[AgentFinding],
    audience: InsightAudience,
    calibrated_confidence: float,
    evidence_snippets: list[str],
    compound: str,
    stint_lap: int,
) -> str | None:
    from f1di.config.settings import settings

    active = "\n".join(
        f"- {f.agent} [{f.risk.value}] {f.summary} (conf {f.confidence:.0%})"
        for f in findings
        if f.risk != RiskLevel.INFO
    ) or "- All agents nominal."

    user_content = (
        f"Risk: {risk.value} | Confidence: {calibrated_confidence:.0%} | Audience: {audience.value}\n"
        f"Compound: {compound}, Stint lap: {stint_lap}\n\n"
        f"Agent findings:\n{active}\n\n"
        f"Top evidence:\n" + "\n".join(f"[{e}]" for e in evidence_snippets[:3] or ["None."])
    )

    if settings.llm_backend == "anthropic":
        return _call_anthropic(user_content, settings)
    if settings.llm_backend == "openai_compatible":
        return _call_openai_compatible(user_content, settings)
    return None


def _call_anthropic(user_content: str, settings) -> str | None:
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed")
        return None

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.llm_timeout_ms / 1000,
    )
    try:
        start = time.perf_counter()
        response = client.messages.create(
            model=settings.llm_advice_model,
            max_tokens=256,
            output_config={
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"recommendation": {"type": "string"}},
                        "required": ["recommendation"],
                        "additionalProperties": False,
                    },
                },
            },
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        _log_latency(start, settings.llm_timeout_ms)
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text:
            return json.loads(text).get("recommendation")
    except Exception as exc:
        logger.warning("anthropic_advisor_failed", extra={"error": str(exc)})
    return None


def _call_openai_compatible(user_content: str, settings) -> str | None:
    try:
        import httpx
    except ImportError:
        logger.warning("httpx package not installed")
        return None

    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    payload = {
        "model": settings.llm_open_source_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }

    try:
        start = time.perf_counter()
        r = httpx.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=settings.llm_timeout_ms / 1000,
        )
        r.raise_for_status()
        _log_latency(start, settings.llm_timeout_ms)
        content = r.json()["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(content).get("recommendation", content)
        except json.JSONDecodeError:
            return content or None
    except Exception as exc:
        logger.warning("openai_compatible_advisor_failed", extra={"error": str(exc)})
    return None


def _log_latency(start: float, timeout_ms: float) -> None:
    ms = (time.perf_counter() - start) * 1000
    logger.debug("llm_advice_latency", extra={"latency_ms": round(ms, 1)})
    if ms > timeout_ms:
        logger.warning("llm_advice_timeout", extra={"latency_ms": round(ms, 1)})
