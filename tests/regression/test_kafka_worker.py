from __future__ import annotations

import json

from f1di.domain.schemas import DriverInsight, TelemetryWindow
from f1di.inference.fusion import InferenceOrchestrator
from f1di.rag.store import HybridMemoryRetriever
from f1di.regression.real_replay import build_window
from f1di.streaming.contracts import decode_window, encode_insight, encode_window
from f1di.streaming.kafka_worker import process_payload


def _make_window() -> TelemetryWindow:
    return build_window(
        {
            "case_id": "kafka_test_nominal",
            "profile": "nominal",
            "compound": "MEDIUM",
            "lap": 10,
            "stint_lap": 6,
            "track_id": "silverstone",
            "driver_id": "VER",
        }
    )


def test_encode_decode_window_roundtrip():
    window = _make_window()
    roundtripped = decode_window(encode_window(window))
    assert roundtripped.session_id == window.session_id
    assert roundtripped.driver_id == window.driver_id
    assert roundtripped.track_id == window.track_id
    assert len(roundtripped.samples) == len(window.samples)


def test_encode_decode_preserves_sample_values():
    window = _make_window()
    rt = decode_window(encode_window(window))
    original = window.samples[0]
    restored = rt.samples[0]
    assert restored.speed_kph == original.speed_kph
    assert restored.compound == original.compound
    assert restored.tire_wear_fl == original.tire_wear_fl


def test_encode_insight_is_valid_json():
    window = _make_window()
    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    insight = orchestrator.analyze(window)
    payload = encode_insight(insight)
    data = json.loads(payload.decode("utf-8"))
    assert "insight_id" in data
    assert "risk" in data
    assert "confidence" in data


def test_process_payload_roundtrip():
    window = _make_window()
    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    result_bytes = process_payload(encode_window(window), orchestrator)
    data = json.loads(result_bytes.decode("utf-8"))
    insight = DriverInsight.model_validate(data)
    assert insight.driver_id == window.driver_id
    assert insight.session_id == window.session_id
    assert insight.findings


def test_process_payload_critical_scenario_raises_risk():
    window = build_window(
        {
            "case_id": "kafka_test_critical",
            "profile": "brake_lockup",
            "compound": "HARD",
            "lap": 37,
            "stint_lap": 22,
            "track_id": "singapore",
            "driver_id": "HAM",
        }
    )
    orchestrator = InferenceOrchestrator(retriever=HybridMemoryRetriever())
    result_bytes = process_payload(encode_window(window), orchestrator)
    insight = DriverInsight.model_validate(json.loads(result_bytes.decode("utf-8")))
    assert insight.risk.value in {"WARNING", "CRITICAL"}
