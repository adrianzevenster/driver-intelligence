from __future__ import annotations

import logging
from typing import Any, Iterable

from f1di.domain.schemas import RetrievedEvidence
from f1di.rag.store import HybridMemoryRetriever, KnowledgeDocument

logger = logging.getLogger("f1di.rag.tiered")

_HOT_SEASONS = 2
_MIN_HOT_RESULTS = 2


def _is_recent(doc: KnowledgeDocument, hot_years: set[int]) -> bool:
    year_str = doc.metadata.get("year", "")
    if year_str:
        try:
            return int(year_str) in hot_years
        except ValueError:
            pass
    for part in doc.source_id.split("_"):
        try:
            y = int(part)
            if 2010 <= y <= 2030:
                return y in hot_years
        except ValueError:
            continue
    # Circuit track/reference docs have no year — always keep hot.
    source = doc.metadata.get("source", "")
    if source in {"knowledge", "fastf1"} and "track" in doc.source_id:
        return True
    return False


class TieredRetriever:
    """Hot (recent N seasons) + cold (full history) retrieval with automatic tier routing."""

    def __init__(
        self,
        hot_seasons: int = _HOT_SEASONS,
        min_hot_results: int = _MIN_HOT_RESULTS,
        encoder: Any | None = None,
    ) -> None:
        from datetime import date
        current_year = date.today().year
        self.hot_years = set(range(current_year - hot_seasons + 1, current_year + 1))
        self.min_hot_results = min_hot_results
        self._hot = HybridMemoryRetriever(encoder=encoder)
        self._cold = HybridMemoryRetriever(encoder=encoder)

    @property
    def documents(self) -> list[KnowledgeDocument]:
        return self._hot.documents + self._cold.documents

    def add_documents(self, docs: Iterable[KnowledgeDocument]) -> None:
        hot_batch: list[KnowledgeDocument] = []
        cold_batch: list[KnowledgeDocument] = []
        for doc in docs:
            (hot_batch if _is_recent(doc, self.hot_years) else cold_batch).append(doc)
        if hot_batch:
            self._hot.add_documents(hot_batch)
        if cold_batch:
            self._cold.add_documents(cold_batch)

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for k, v in self._hot.source_counts().items():
            counts[f"hot:{k}"] = v
        for k, v in self._cold.source_counts().items():
            counts[f"cold:{k}"] = v
        return counts

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedEvidence]:
        hot_results = self._hot.search(query, top_k=top_k, filters=filters)
        if len(hot_results) >= self.min_hot_results:
            return [
                RetrievedEvidence(
                    source_id=r.source_id, title=r.title, text=r.text,
                    score=min(1.0, r.score * 1.15),
                    metadata={**r.metadata, "tier": "hot"},
                )
                for r in hot_results
            ]

        cold_results = self._cold.search(query, top_k=top_k, filters=filters)
        seen: set[str] = {r.source_id for r in hot_results}
        merged = list(hot_results)
        for r in cold_results:
            if r.source_id not in seen:
                merged.append(RetrievedEvidence(
                    source_id=r.source_id, title=r.title, text=r.text,
                    score=r.score * 0.85,
                    metadata={**r.metadata, "tier": "cold"},
                ))
                seen.add(r.source_id)

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    @property
    def hot_document_count(self) -> int:
        return len(self._hot.documents)

    @property
    def cold_document_count(self) -> int:
        return len(self._cold.documents)
