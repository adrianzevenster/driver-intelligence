from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from f1di.domain.schemas import AgentFinding, RetrievedEvidence, TelemetryWindow
from f1di.features.extractor import RaceFeatures
from f1di.rag.store import HybridMemoryRetriever


def multi_source_evidence(
    retriever,
    track_id: str,
    knowledge_query: str,
    fastf1_query: str,
    jolpica_query: str,
    top_k: int = 4,
) -> list[RetrievedEvidence]:
    """Gather and deduplicate evidence from circuit knowledge, FastF1, and Jolpica in parallel."""
    searches = [
        (knowledge_query, {"track_id": track_id}, 2),
        (fastf1_query,    {"source": "fastf1", "track_id": track_id}, 2),
        (jolpica_query,   {"source": "jolpica", "track_id": track_id}, 1),
    ]

    results: list[list[RetrievedEvidence]] = [[] for _ in searches]

    def _search(idx: int, query: str, filters: dict, k: int) -> tuple[int, list]:
        return idx, retriever.search(query, top_k=k, filters=filters)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_search, i, q, f, k): i for i, (q, f, k) in enumerate(searches)}
        for fut in as_completed(futures):
            idx, hits = fut.result()
            results[idx] = hits

    seen: set[str] = set()
    evidence: list[RetrievedEvidence] = []
    for batch in results:
        for e in batch:
            if e.source_id not in seen:
                seen.add(e.source_id)
                evidence.append(e)
    return evidence[:top_k]


class RaceAgent(ABC):
    name: str

    @abstractmethod
    def analyze(
        self,
        window: TelemetryWindow,
        features: RaceFeatures,
        retriever: HybridMemoryRetriever,
    ) -> AgentFinding:
        raise NotImplementedError
