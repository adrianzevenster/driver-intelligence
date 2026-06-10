from __future__ import annotations

import logging
import textwrap

import httpx

from f1di.config.settings import settings
from f1di.domain.schemas import DriverInsight, RiskLevel

logger = logging.getLogger("f1di.delivery")

_RISK_EMOJI = {
    RiskLevel.INFO: "ℹ️",
    RiskLevel.WATCH: "👀",
    RiskLevel.WARNING: "⚠️",
    RiskLevel.CRITICAL: "🚨",
}

_RISK_ORDER = [RiskLevel.INFO, RiskLevel.WATCH, RiskLevel.WARNING, RiskLevel.CRITICAL]


def _format_message(insight: DriverInsight) -> str:
    emoji = _RISK_EMOJI.get(insight.risk, "🔔")
    rec = textwrap.shorten(insight.recommendation, width=280, placeholder="…")
    agents_fired = [f.agent for f in insight.findings if f.risk not in {RiskLevel.INFO}]
    return (
        f"{emoji} *F1DI Alert* — {insight.risk.value}\n"
        f"Driver: `{insight.driver_id}` | Session: `{insight.session_id}`\n"
        f"Confidence: {insight.confidence:.0%} | Policy: {insight.policy}\n"
        f"Agents: {', '.join(agents_fired) or '—'}\n\n"
        f"{rec}"
    )


class PushNotifier:
    def __init__(self) -> None:
        self._min_risk = RiskLevel(getattr(settings, "notify_min_risk", "WARNING"))

    def should_notify(self, insight: DriverInsight) -> bool:
        return _RISK_ORDER.index(insight.risk) >= _RISK_ORDER.index(self._min_risk)

    def notify(self, insight: DriverInsight) -> None:
        if not self.should_notify(insight):
            return
        message = _format_message(insight)
        if getattr(settings, "telegram_bot_token", "") and getattr(settings, "telegram_chat_id", ""):
            self._telegram(message)
        if getattr(settings, "slack_webhook_url", ""):
            self._slack(message)

    def _telegram(self, message: str) -> bool:
        try:
            resp = httpx.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={"chat_id": settings.telegram_chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=5.0,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)
            return False

    def _slack(self, message: str) -> bool:
        try:
            resp = httpx.post(
                settings.slack_webhook_url,
                json={"text": message},
                timeout=5.0,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Slack notification failed: %s", exc)
            return False


_notifier: PushNotifier | None = None


def get_notifier() -> PushNotifier:
    global _notifier
    if _notifier is None:
        _notifier = PushNotifier()
    return _notifier


def notify_if_configured(insight: DriverInsight) -> None:
    has_telegram = bool(getattr(settings, "telegram_bot_token", ""))
    has_slack = bool(getattr(settings, "slack_webhook_url", ""))
    if has_telegram or has_slack:
        get_notifier().notify(insight)
