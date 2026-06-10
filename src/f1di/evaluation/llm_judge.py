"""LLM judge for recommendation quality evaluation.

Scores each recommendation on four rubric dimensions using the configured
Ollama (or any OpenAI-compatible) backend. Intended for offline eval runs
against the replay fixture set — not invoked in the hot path.

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

    @property
    def mean(self) -> float:
        return (self.safety + self.actionability + self.register + self.calibration) / 4

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_recommendation(
    recommendation: str,
    risk: str,
    audience: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 15.0,
) -> JudgeScore | None:
    """Score one recommendation. Returns None if the LLM call fails."""
    from f1di.config.settings import settings

    _base_url = base_url or settings.llm_base_url
    _model = model or settings.llm_open_source_model

    user_content = (
        f"Risk level: {risk}\n"
        f"Audience: {audience}\n"
        f"Recommendation: {recommendation}"
    )

    try:
        import httpx

        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        payload = {
            "model": _model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        r = httpx.post(
            f"{_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_s,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()

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
        )
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return None
