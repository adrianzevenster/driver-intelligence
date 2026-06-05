from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Formula 1 race engineer assistant with deep expertise in telemetry analysis, \
tire strategy, ERS deployment, braking, and race strategy. Answer questions about F1 \
race data, engineering concepts, and strategy decisions. Be direct and technical. \
Use the provided knowledge-base context when it is relevant to the question. \
If the context does not cover the question, answer from your own expertise and say so."""


def chat(
    message: str,
    history: list[dict],
    context_snippets: list[str],
) -> str | None:
    from f1di.config.settings import settings

    context_block = ""
    if context_snippets:
        joined = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(context_snippets))
        context_block = f"\n\nKnowledge-base context:\n{joined}\n"

    if settings.llm_backend == "anthropic":
        return _call_anthropic(message, history, context_block, settings)
    if settings.llm_backend == "openai_compatible":
        return _call_openai_compatible(message, history, context_block, settings)
    return None


def _build_messages(message: str, history: list[dict], context_block: str) -> list[dict]:
    messages = list(history[-10:])  # last 10 turns max
    user_content = message
    if context_block:
        user_content = context_block + "\nQuestion: " + message
    messages.append({"role": "user", "content": user_content})
    return messages


def _call_anthropic(message: str, history: list[dict], context_block: str, settings) -> str | None:
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        response = client.messages.create(
            model=settings.llm_advice_model,
            max_tokens=1024,
            output_config={"effort": "medium"},
            system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=_build_messages(message, history, context_block),
        )
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception as exc:
        logger.warning("anthropic_chat_failed", extra={"error": str(exc)})
    return None


def _call_openai_compatible(message: str, history: list[dict], context_block: str, settings) -> str | None:
    try:
        import httpx
    except ImportError:
        return None

    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages += _build_messages(message, history, context_block)

    payload = {
        "model": settings.llm_open_source_model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    try:
        r = httpx.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=settings.llm_timeout_ms / 1000,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip() or None
    except Exception as exc:
        logger.warning("openai_compatible_chat_failed", extra={"error": str(exc)})
    return None
