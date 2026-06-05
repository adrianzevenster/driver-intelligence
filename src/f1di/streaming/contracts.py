from __future__ import annotations

import json
from f1di.domain.schemas import DriverInsight, TelemetryWindow


def encode_window(window: TelemetryWindow) -> bytes:
    return json.dumps(window.model_dump(mode="json")).encode("utf-8")


def decode_window(payload: bytes) -> TelemetryWindow:
    return TelemetryWindow.model_validate(json.loads(payload.decode("utf-8")))


def encode_insight(insight: DriverInsight) -> bytes:
    return json.dumps(insight.model_dump(mode="json")).encode("utf-8")
