from __future__ import annotations

from pathlib import Path

from f1di.confidence.calibration import compute_raw_score
from f1di.domain.schemas import AgentFinding, RetrievedEvidence, RiskLevel
from f1di.rag.store import HybridMemoryRetriever, load_markdown_knowledge


def _retriever() -> HybridMemoryRetriever:
    retriever = HybridMemoryRetriever()
    retriever.add_documents(load_markdown_knowledge(Path("data/knowledge")))
    return retriever


def test_static_knowledge_retrieval_hits_expected_circuit_documents():
    retriever = _retriever()

    cases = [
        ("front-left thermal load through maggotts becketts chapel", "silverstone_track"),
        ("monaco rain intermediate tyres safety car casino tunnel exit", "monaco_weather"),
        ("spa ers deployment kemmel eau rouge la source soc", "spa_ers"),
        ("singapore undercut sector 2 esplanade pit entry fatigue", "singapore_track"),
    ]

    for query, expected_source_id in cases:
        results = retriever.search(query, top_k=3)
        assert results, f"No retrieval results for query: {query}"
        assert expected_source_id in {r.source_id for r in results}, (
            f"Expected {expected_source_id} in top-3 for query {query!r}; "
            f"got {[r.source_id for r in results]}"
        )


def test_static_knowledge_retrieval_filter_limits_track_scope():
    retriever = _retriever()

    results = retriever.search(
        "ERS deployment harvest zone battery state",
        top_k=5,
        filters={"track_id": "spa"},
    )

    assert results
    assert all(r.metadata["track_id"] == "spa" for r in results)


def test_raw_calibration_features_include_agreement_and_max_risk():
    evidence = [
        RetrievedEvidence(
            source_id="test_source",
            title="Test evidence",
            text="Grounded evidence",
            score=0.8,
        )
    ]
    findings = [
        AgentFinding(
            agent="telemetry",
            risk=RiskLevel.WARNING,
            summary="warning",
            confidence=0.7,
            evidence=evidence,
        ),
        AgentFinding(
            agent="tire_strategy",
            risk=RiskLevel.CRITICAL,
            summary="critical",
            confidence=0.8,
            evidence=evidence,
        ),
    ]

    raw, features = compute_raw_score(findings)

    assert 0.0 <= raw <= 1.0
    assert features["agent_agreement"] > 0.0
    assert features["risk_max"] == 0.90
    assert features["evidence_strength"] == 0.8

