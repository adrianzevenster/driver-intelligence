from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch


from f1di.delivery.notifier import PushNotifier, _format_message, notify_if_configured
from f1di.domain.schemas import (
    AgentFinding, DriverInsight, InsightAudience, RiskLevel,
)


def _insight(risk: RiskLevel = RiskLevel.WARNING) -> DriverInsight:
    finding = AgentFinding(
        agent="telemetry",
        risk=risk,
        confidence=0.75,
        summary="Tire wear high",
        features={},
        evidence=[],
    )
    return DriverInsight(
        insight_id=str(uuid.uuid4()),
        session_id="test_session",
        driver_id="VER",
        risk=risk,
        confidence=0.75,
        uncertainty=0.25,
        policy="SHOW",
        audience=InsightAudience.DRIVER,
        recommendation="Consider pitting within 3 laps.",
        findings=[finding],
        evidence=[],
        supporting_factors=[],
        latency_ms=12.0,
    )


class TestShouldNotify:
    def test_warning_above_threshold(self):
        n = PushNotifier()
        assert n.should_notify(_insight(RiskLevel.WARNING))

    def test_critical_above_threshold(self):
        n = PushNotifier()
        assert n.should_notify(_insight(RiskLevel.CRITICAL))

    def test_info_below_threshold(self):
        n = PushNotifier()
        assert not n.should_notify(_insight(RiskLevel.INFO))

    def test_watch_below_warning_threshold(self):
        n = PushNotifier()
        assert not n.should_notify(_insight(RiskLevel.WATCH))


class TestFormatMessage:
    def test_contains_driver(self):
        msg = _format_message(_insight())
        assert "VER" in msg

    def test_contains_risk(self):
        msg = _format_message(_insight(RiskLevel.CRITICAL))
        assert "CRITICAL" in msg

    def test_contains_recommendation(self):
        msg = _format_message(_insight())
        assert "pitting" in msg


class TestNotifyIfConfigured:
    def test_no_op_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr("f1di.delivery.notifier.settings", MagicMock(
            telegram_bot_token="", slack_webhook_url=""
        ))
        notify_if_configured(_insight())  # should not raise

    def test_calls_telegram_when_configured(self, monkeypatch):
        mock_settings = MagicMock(
            telegram_bot_token="tok123",
            telegram_chat_id="-100999",
            slack_webhook_url="",
            notify_min_risk="WARNING",
        )
        monkeypatch.setattr("f1di.delivery.notifier.settings", mock_settings)

        sent = []

        def fake_post(url, **kwargs):
            sent.append(url)
            r = MagicMock()
            r.raise_for_status = lambda: None
            return r

        with patch("httpx.post", side_effect=fake_post):
            notify_if_configured(_insight(RiskLevel.WARNING))

        assert len(sent) == 1
        assert "telegram" in sent[0]
