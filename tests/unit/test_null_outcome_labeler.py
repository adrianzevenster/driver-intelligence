"""Tests for label_quiet_stints null-outcome flywheel."""
from __future__ import annotations

import datetime
import uuid


def _make_session_id(year: int, round_num: int) -> str:
    return f"fastf1_{year}_{round_num}"


def test_label_quiet_stints_no_insights():
    """label_quiet_stints returns 0 when no insights exist for the round."""
    from f1di.data.outcome_labeler import label_quiet_stints
    result = label_quiet_stints(9999, 99)
    assert result == 0


def test_label_quiet_stints_skips_recent():
    """Insights created <2h ago are not labeled (race may not have finished)."""
    from f1di.storage.database import db_session
    from f1di.storage.models import InsightRecord

    iid = str(uuid.uuid4())
    with db_session() as session:
        ins = InsightRecord(
            insight_id=iid,
            session_id=_make_session_id(9999, 98),
            driver_id="VER",
            track_id="silverstone",
            risk="INFO",
            policy="INFO",
            confidence=0.4,
            uncertainty=0.6,
            audience="DRIVER",
            recommendation="All nominal.",
            latency_ms=50.0,
            created_at=datetime.datetime.utcnow(),  # too recent
        )
        session.add(ins)
        session.commit()

    from f1di.data.outcome_labeler import label_quiet_stints
    result = label_quiet_stints(9999, 98)
    assert result == 0


def test_label_quiet_stints_labels_old_info():
    """Old INFO insights with no feedback get labeled as correct."""
    from f1di.storage.database import db_session
    from f1di.storage.models import FeedbackRecord, InsightRecord

    iid = str(uuid.uuid4())
    with db_session() as session:
        ins = InsightRecord(
            insight_id=iid,
            session_id=_make_session_id(9999, 97),
            driver_id="HAM",
            track_id="silverstone",
            risk="INFO",
            policy="INFO",
            confidence=0.35,
            uncertainty=0.65,
            audience="DRIVER",
            recommendation="No action needed.",
            latency_ms=45.0,
            created_at=datetime.datetime.utcnow() - datetime.timedelta(hours=5),
        )
        session.add(ins)
        session.commit()

    from f1di.data.outcome_labeler import label_quiet_stints
    result = label_quiet_stints(9999, 97)
    assert result >= 1

    with db_session() as session:
        fb = session.query(FeedbackRecord).filter_by(insight_id=iid).first()
    assert fb is not None
    assert fb.correct is True
    assert fb.submitted_by == "null_outcome"


def test_label_quiet_stints_skips_suppress():
    """SUPPRESS insights are excluded — they are system-internal and not feedback candidates."""
    from f1di.storage.database import db_session
    from f1di.storage.models import FeedbackRecord, InsightRecord

    iid = str(uuid.uuid4())
    with db_session() as session:
        ins = InsightRecord(
            insight_id=iid,
            session_id=_make_session_id(9999, 95),
            driver_id="NOR",
            track_id="monza",
            risk="INFO",
            policy="SUPPRESS",
            confidence=0.3,
            uncertainty=0.7,
            audience="DRIVER",
            recommendation="Suppressed.",
            latency_ms=40.0,
            created_at=datetime.datetime.utcnow() - datetime.timedelta(hours=5),
        )
        session.add(ins)
        session.commit()

    from f1di.data.outcome_labeler import label_quiet_stints
    result = label_quiet_stints(9999, 95)
    assert result == 0  # SUPPRESS must be skipped

    with db_session() as session:
        fb = session.query(FeedbackRecord).filter_by(insight_id=iid).first()
    assert fb is None


def test_label_quiet_stints_no_double_label():
    """Insights that already have feedback are not double-labeled."""
    from f1di.storage.database import db_session
    from f1di.storage.models import FeedbackRecord, InsightRecord

    iid = str(uuid.uuid4())
    with db_session() as session:
        ins = InsightRecord(
            insight_id=iid,
            session_id=_make_session_id(9999, 96),
            driver_id="LEC",
            track_id="monaco",
            risk="LOW",
            policy="INFO",
            confidence=0.3,
            uncertainty=0.7,
            audience="DRIVER",
            recommendation="Monitor.",
            latency_ms=40.0,
            created_at=datetime.datetime.utcnow() - datetime.timedelta(hours=6),
        )
        session.add(ins)
        fb = FeedbackRecord(insight_id=iid, rating=5, correct=True, submitted_by="test")
        session.add(fb)
        session.commit()

    from f1di.data.outcome_labeler import label_quiet_stints
    result = label_quiet_stints(9999, 96)
    assert result == 0
