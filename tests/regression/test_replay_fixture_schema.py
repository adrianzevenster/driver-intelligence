from __future__ import annotations

import json
from pathlib import Path

from f1di.domain.schemas import TelemetryWindow


REQUIRED_SOURCE_FIELDS = {
    "type",
    "series",
    "event",
    "session",
    "labeling_date",
    "labeler",
    "provenance_note",
}


def test_real_replay_fixture_schema_and_metadata():
    cases = json.loads(Path("data/fixtures/real_replay_eval.json").read_text(encoding="utf-8"))

    assert cases
    for case in cases:
        assert case["case_id"]
        assert case["class"]
        assert REQUIRED_SOURCE_FIELDS.issubset(case["source"])
        assert case["label"]["rationale"]
        assert case["label"]["outcome"]
        assert case.get("expected_min_risk") or case.get("expected_max_risk")
        assert case["expected_sources"]

        if case.get("expected_min_risk"):
            assert case.get("expected_agents")
            if case.get("expected_policy"):
                assert case["expected_policy"] in {"SHOW", "ENGINEER_ONLY", "SUPPRESS"}

        window = TelemetryWindow.model_validate(case["window"])
        assert window.samples
        assert window.session_id
        assert window.driver_id
        assert window.track_id

