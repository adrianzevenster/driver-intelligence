from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("F1DI_INTEGRATION") != "1",
    reason="integration gates are opt-in; set F1DI_INTEGRATION=1",
)


def test_qdrant_retriever_integration_smoke():
    from f1di.rag.qdrant_backend import QdrantHybridRetriever
    from f1di.rag.store import KnowledgeDocument

    retriever = QdrantHybridRetriever(
        url=os.environ.get("F1DI_QDRANT_URL", "http://localhost:6333"),
        collection=os.environ.get("F1DI_QDRANT_COLLECTION", "f1di_integration_test"),
        model_name=os.environ.get("F1DI_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    retriever.add_documents([
        KnowledgeDocument(
            source_id="integration_spa_ers",
            title="Spa ERS Integration",
            text="Kemmel deployment and La Source harvest require SOC management.",
            metadata={"track_id": "spa", "source": "knowledge"},
        )
    ])

    results = retriever.search("spa kemmel source soc deployment", filters={"track_id": "spa"})

    assert results
    assert results[0].source_id == "integration_spa_ers"


def test_fastf1_capture_integration_smoke():
    from f1di.knowledge.fastf1_session import build_window

    window = build_window(year=2024, round_num=12, driver="VER", lap_number=10, n_samples=4, window_laps=1)

    assert window.samples
    assert window.track_id
    assert window.driver_id == "VER"

