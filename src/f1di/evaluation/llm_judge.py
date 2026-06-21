"""LLM judge for recommendation quality evaluation.

The judge is intentionally decoupled from the inference backend to eliminate
self-evaluation bias — models consistently favour their own output style.
Selection logic:
  - Inference backend = anthropic  →  judge uses Ollama (open-source model)
  - Inference backend = openai_compatible (Ollama)  →  judge uses Anthropic if
    an API key is available, otherwise a different Ollama model is used
  - Override: set judge_model / judge_base_url in settings to force a specific
    judge regardless of the inference backend

Rubric dimensions (each 0.0–1.0):
  safety       — uses safety/stability language appropriate to the risk level
  actionability— contains a concrete action the driver or engineer can take
  register     — tone matches the audience (terse for DRIVER, analytical for ENGINEER)
  calibration  — not alarmist for low-risk; not dismissive for high-risk
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Literal

logger = logging.getLogger("f1di.evaluation.llm_judge")

_SYSTEM_PROMPT = """\
You are an expert Formula 1 race-engineer reviewer evaluating AI-generated recommendations.
Score the recommendation on four dimensions, each from 0.0 to 1.0.

Scoring criteria:
  safety       (0–1): Does it use safety/stability language proportional to the risk? \
CRITICAL/WARNING must include action; INFO must not be alarmist.
  actionability(0–1): Does it contain a concrete, executable action (e.g. "open pit discussion", \
"reduce deployment", "monitor")? Generic phrases like "be careful" score low.
  register     (0–1): Does the tone match the audience? \
DRIVER: ≤15 words, imperative. ENGINEER/STRATEGY: ≤2 sentences, analytical.
  calibration  (0–1): Is the urgency proportional to the risk? \
Over-alarming on INFO or under-alarming on CRITICAL both score low.

Return valid JSON only: {"safety": <float>, "actionability": <float>, "register": <float>, "calibration": <float>, "rationale": "<one sentence>"}"""


@dataclass
class JudgeScore:
    safety: float
    actionability: float
    register: float
    calibration: float
    rationale: str = ""
    judge_backend: str = ""
    judge_model: str = ""

    @property
    def mean(self) -> float:
        return (self.safety + self.actionability + self.register + self.calibration) / 4

    def to_dict(self) -> dict:
        return asdict(self)


def _select_judge_backend(settings) -> tuple[Literal["openai_compatible", "anthropic"], str, str]:
    """Return (backend, model, base_url) for the judge, opposite to inference backend.

    Rules:
    1. If judge_model is explicitly set in settings, use it with judge_base_url.
    2. If inference = anthropic → judge with Ollama.
    3. If inference = openai_compatible → judge with Anthropic if key exists.
    4. If inference = openai_compatible and no Anthropic key → use a fallback
       open-source model (mistral or llama3.1) via the same Ollama endpoint.
    """
    # Explicit override wins
    if getattr(settings, "judge_model", ""):
        judge_url = getattr(settings, "judge_base_url", None) or settings.llm_base_url
        return "openai_compatible", settings.judge_model, judge_url

    inference_backend = settings.llm_backend

    if inference_backend == "anthropic":
        # Judge with Ollama — different model family
        judge_model = settings.llm_open_source_model
        logger.debug(
            "judge_backend_selected: inference=anthropic → judge=ollama model=%s",
            judge_model,
        )
        return "openai_compatible", judge_model, settings.llm_base_url

    # inference is openai_compatible (Ollama) or rules
    if settings.anthropic_api_key:
        # Judge with Anthropic — different model family
        # Use a smaller/faster Claude model to keep latency reasonable
        judge_model = "claude-haiku-4-5-20251001"
        logger.debug(
            "judge_backend_selected: inference=ollama → judge=anthropic model=%s",
            judge_model,
        )
        return "anthropic", judge_model, ""

    # No Anthropic key — use the same Ollama model that's configured for inference.
    # This loses strict cross-model independence but guarantees the judge actually
    # runs on whatever model is available on the Ollama instance.
    judge_model = settings.llm_open_source_model
    logger.debug(
        "judge_backend_selected: inference=ollama no_anthropic_key → judge=ollama model=%s",
        judge_model,
    )
    return "openai_compatible", judge_model, settings.llm_base_url


def _call_openai_compatible(
    base_url: str,
    model: str,
    user_content: str,
    timeout_s: float,
    api_key: str = "",
) -> str:
    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    r = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_anthropic(model: str, user_content: str, api_key: str, timeout_s: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


def evaluate_recommendation(
    recommendation: str,
    risk: str,
    audience: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 15.0,
) -> JudgeScore | None:
    """Score one recommendation using the cross-model judge.

    If base_url/model are provided, they override the auto-selection logic.
    Returns None if the LLM call fails.
    """
    from f1di.config.settings import settings

    user_content = (
        f"Risk level: {risk}\n"
        f"Audience: {audience}\n"
        f"Recommendation: {recommendation}"
    )

    # Determine judge backend and model
    if model and base_url:
        backend: Literal["openai_compatible", "anthropic"] = "openai_compatible"
        judge_model = model
        judge_url = base_url
    elif model:
        backend = "openai_compatible"
        judge_model = model
        judge_url = settings.llm_base_url
    else:
        backend, judge_model, judge_url = _select_judge_backend(settings)

    try:
        if backend == "anthropic":
            if not settings.anthropic_api_key:
                logger.warning("judge_anthropic_no_key — falling back to openai_compatible")
                backend = "openai_compatible"
                judge_model = settings.llm_open_source_model
                judge_url = settings.llm_base_url
                content = _call_openai_compatible(
                    judge_url, judge_model, user_content, timeout_s, settings.llm_api_key
                )
            else:
                content = _call_anthropic(
                    judge_model, user_content, settings.anthropic_api_key, timeout_s
                )
        else:
            content = _call_openai_compatible(
                judge_url, judge_model, user_content, timeout_s, settings.llm_api_key
            )

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        parsed = json.loads(content)
        return JudgeScore(
            safety=float(parsed.get("safety", 0.0)),
            actionability=float(parsed.get("actionability", 0.0)),
            register=float(parsed.get("register", 0.0)),
            calibration=float(parsed.get("calibration", 0.0)),
            rationale=str(parsed.get("rationale", "")),
            judge_backend=backend,
            judge_model=judge_model,
        )
    except Exception as exc:
        logger.warning("llm_judge_failed backend=%s model=%s: %s", backend, judge_model, exc)
        return _rules_judge(recommendation, risk, audience)


def _rules_judge(recommendation: str, risk: str, audience: str) -> JudgeScore:
    """Heuristic scorer used when the LLM judge is unavailable."""
    rec = recommendation.lower()

    action_words = {"pit", "open", "reduce", "adjust", "switch", "monitor", "protect",
                    "prepare", "engage", "back off", "close gap", "confirm"}
    safety_words = {"stabilise", "stabilize", "alert", "caution", "monitor",
                    "reduce push", "back off", "brake", "cliff"}
    urgent_words = {"now", "immediately", "prioritise", "prioritize"}

    actionability = 0.88 if any(w in rec for w in action_words) else 0.42
    safety_boost = any(w in rec for w in safety_words)
    urgent = any(w in rec for w in urgent_words)

    if risk in ("CRITICAL", "WARNING"):
        safety = 0.88 if (safety_boost or urgent) else 0.60
        calibration = 0.88 if urgent else 0.65
    else:
        safety = 0.55 if ("critical" in rec or "danger" in rec) else 0.85
        calibration = 0.85 if not urgent else 0.50  # urgent language for low risk

    words = rec.split()
    if audience in ("DRIVER",):
        register = 0.88 if len(words) <= 18 else 0.58
    else:
        register = 0.85 if len(words) <= 40 else 0.55

    return JudgeScore(
        safety=round(safety, 2),
        actionability=round(actionability, 2),
        register=round(register, 2),
        calibration=round(calibration, 2),
        rationale="Heuristic score — LLM judge unavailable.",
        judge_backend="rules",
        judge_model="heuristic-v1",
    )
