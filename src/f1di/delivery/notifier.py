from __future__ import annotations

import logging
import smtplib
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

_RISK_COLORS = {
    RiskLevel.INFO: "#64748b",
    RiskLevel.WATCH: "#2563eb",
    RiskLevel.WARNING: "#d97706",
    RiskLevel.CRITICAL: "#dc2626",
}

_RISK_ORDER = [RiskLevel.INFO, RiskLevel.WATCH, RiskLevel.WARNING, RiskLevel.CRITICAL]

# Runtime-mutable recipient list — seeded from settings, editable via API.
_runtime_recipients: list[str] | None = None


def get_recipients() -> list[str]:
    global _runtime_recipients
    if _runtime_recipients is None:
        _runtime_recipients = [
            r.strip() for r in settings.email_recipients.split(",") if r.strip()
        ]
    return _runtime_recipients


def set_recipients(recipients: list[str]) -> None:
    global _runtime_recipients
    _runtime_recipients = [r.strip() for r in recipients if r.strip()]


def _format_message(insight: DriverInsight) -> str:
    return _format_plain(insight)


def _format_plain(insight: DriverInsight) -> str:
    emoji = _RISK_EMOJI.get(insight.risk, "🔔")
    rec = textwrap.shorten(insight.recommendation, width=280, placeholder="…")
    agents_fired = [f.agent for f in insight.findings if f.risk not in {RiskLevel.INFO}]
    return (
        f"{emoji} F1DI Alert — {insight.risk.value}\n"
        f"Driver: {insight.driver_id} | Session: {insight.session_id}\n"
        f"Confidence: {insight.confidence:.0%} | Policy: {insight.policy}\n"
        f"Agents: {', '.join(agents_fired) or '—'}\n\n"
        f"{rec}"
    )


def _format_html(insight: DriverInsight) -> str:
    color = _RISK_COLORS.get(insight.risk, "#64748b")
    emoji = _RISK_EMOJI.get(insight.risk, "🔔")
    rec = textwrap.shorten(insight.recommendation, width=400, placeholder="…")
    agents_fired = [f.agent for f in insight.findings if f.risk not in {RiskLevel.INFO}]
    rows = "".join(
        f"<tr><td style='padding:4px 8px;color:#94a3b8;'>{f.agent}</td>"
        f"<td style='padding:4px 8px;'>{f.summary}</td></tr>"
        for f in insight.findings
    )
    return f"""
<div style="font-family:monospace;background:#0f172a;color:#e2e8f0;padding:24px;border-radius:8px;max-width:600px;">
  <div style="border-left:4px solid {color};padding-left:12px;margin-bottom:16px;">
    <h2 style="margin:0;color:{color};">{emoji} F1DI — {insight.risk.value}</h2>
    <p style="margin:4px 0;color:#94a3b8;font-size:12px;">
      Driver: <strong style="color:#e2e8f0;">{insight.driver_id}</strong> &nbsp;|&nbsp;
      Session: <strong style="color:#e2e8f0;">{insight.session_id}</strong>
    </p>
    <p style="margin:4px 0;color:#94a3b8;font-size:12px;">
      Confidence: <strong style="color:#e2e8f0;">{insight.confidence:.0%}</strong> &nbsp;|&nbsp;
      Policy: <strong style="color:#e2e8f0;">{insight.policy}</strong> &nbsp;|&nbsp;
      Agents: <strong style="color:#e2e8f0;">{', '.join(agents_fired) or '—'}</strong>
    </p>
  </div>
  <p style="color:#e2e8f0;margin-bottom:16px;">{rec}</p>
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <thead><tr>
      <th style="text-align:left;padding:4px 8px;color:#64748b;">Agent</th>
      <th style="text-align:left;padding:4px 8px;color:#64748b;">Finding</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""


class PushNotifier:
    def __init__(self) -> None:
        try:
            self._min_risk = RiskLevel(getattr(settings, "notify_min_risk", "WARNING"))
        except (ValueError, TypeError):
            self._min_risk = RiskLevel.WARNING

    def set_min_risk(self, risk: str) -> None:
        self._min_risk = RiskLevel(risk)

    def get_min_risk(self) -> str:
        return self._min_risk.value

    def should_notify(self, insight: DriverInsight) -> bool:
        return _RISK_ORDER.index(insight.risk) >= _RISK_ORDER.index(self._min_risk)

    def notify(self, insight: DriverInsight) -> dict:
        if not self.should_notify(insight):
            return {"skipped": True, "reason": f"risk {insight.risk.value} below threshold {self._min_risk.value}"}
        plain = _format_plain(insight)
        html = _format_html(insight)
        subject = f"[F1DI] {insight.risk.value} — {insight.driver_id} / {insight.session_id}"
        results: dict[str, bool] = {}
        if getattr(settings, "smtp_username", "") and getattr(settings, "smtp_password", ""):
            results["email"] = self._email(subject, plain, html)
        if getattr(settings, "telegram_bot_token", "") and getattr(settings, "telegram_chat_id", ""):
            results["telegram"] = self._telegram(plain)
        if getattr(settings, "slack_webhook_url", ""):
            results["slack"] = self._slack(plain)
        return results

    def send_test(self, to: list[str] | None = None) -> dict:
        subject = "[F1DI] Test notification — delivery check"
        plain = "This is a test notification from F1 Driver Intelligence.\nIf you received this, delivery is configured correctly."
        html = """
<div style="font-family:monospace;background:#0f172a;color:#e2e8f0;padding:24px;border-radius:8px;">
  <h2 style="color:#22c55e;">F1DI — Test Notification</h2>
  <p>Delivery is configured correctly. You will receive alerts for qualifying risk events.</p>
</div>
"""
        recipients = to or get_recipients()
        if getattr(settings, "smtp_username", "") and getattr(settings, "smtp_password", ""):
            return {"email": self._email(subject, plain, html, override_recipients=recipients)}
        return {"email": False, "reason": "SMTP not configured"}

    def _email(
        self,
        subject: str,
        plain: str,
        html: str,
        override_recipients: list[str] | None = None,
    ) -> bool:
        recipients = override_recipients or get_recipients()
        if not recipients:
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = settings.smtp_username
            msg["To"] = ", ".join(recipients)
            msg.attach(MIMEText(plain, "plain"))
            msg.attach(MIMEText(html, "html"))
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
                smtp.starttls()
                smtp.login(settings.smtp_username, settings.smtp_password)
                smtp.sendmail(settings.smtp_username, recipients, msg.as_string())
            return True
        except Exception as exc:
            logger.warning("Email notification failed: %s", exc)
            return False

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
    has_email = bool(getattr(settings, "smtp_username", "") and getattr(settings, "smtp_password", ""))
    has_telegram = bool(getattr(settings, "telegram_bot_token", ""))
    has_slack = bool(getattr(settings, "slack_webhook_url", ""))
    if has_email or has_telegram or has_slack:
        get_notifier().notify(insight)


def send_system_alert(subject: str, plain: str) -> None:
    """Send a plain-text system-level alert (not tied to a specific insight).

    Silently no-ops when no delivery channel is configured.
    """
    has_email    = bool(getattr(settings, "smtp_username", "") and getattr(settings, "smtp_password", ""))
    has_telegram = bool(getattr(settings, "telegram_bot_token", ""))
    has_slack    = bool(getattr(settings, "slack_webhook_url", ""))
    if not (has_email or has_telegram or has_slack):
        return
    notifier = get_notifier()
    if has_email:
        notifier._email(subject, plain, f"<pre style='font-family:monospace'>{plain}</pre>")
    if has_telegram:
        notifier._telegram(f"*{subject}*\n{plain}")
    if has_slack:
        notifier._slack(f"*{subject}*\n{plain}")
