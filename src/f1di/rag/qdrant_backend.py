from __future__ import annotations

import hashlib
from typing import Any, Iterable

from f1di.domain.schemas import RetrievedEvidence


class QdrantHybridRetriever:
    def __init__(
        self,
        url: str,
        collection: str,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self.url = url.rstrip("/")
        self.client = QdrantClient(url=url)
        self.collection = collection
        from f1di.config.settings import settings as _s
        self._encoder = SentenceTransformer(model_name, local_files_only=_s.embedding_offline)
        self._vector_size: int = self._encoder.get_embedding_dimension()
        self._ensure_collection()

    @property
    def documents(self) -> list[Any]:
        try:
            info = self.client.get_collection(self.collection)
            count = info.points_count or 0
        except Exception:
            count = 0
        return [None] * count

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=250,
                offset=offset,
                with_payload=["meta_source"],
                with_vectors=False,
            )
            for point in results:
                src = point.payload.get("meta_source", "unknown")
                counts[src] = counts.get(src, 0) + 1
            if next_offset is None:
                break
            offset = next_offset
        return counts

    def add_documents(self, docs: Iterable) -> None:
        from qdrant_client.models import PointStruct

        doc_list = list(docs)
        if not doc_list:
            return

        texts = [d.title + " " + d.text[:500] for d in doc_list]
        embeddings = self._encoder.encode(texts, normalize_embeddings=True)

        points = [
            PointStruct(
                id=self._stable_id(d.source_id),
                vector=embeddings[i].tolist(),
                payload={
                    "source_id": d.source_id,
                    "title": d.title,
                    "text": d.text,
                    **{f"meta_{k}": v for k, v in d.metadata.items()},
                },
            )
            for i, d in enumerate(doc_list)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedEvidence]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not query.strip():
            return []

        query_emb = self._encoder.encode([query], normalize_embeddings=True)[0]

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=f"meta_{k}", match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_emb.tolist(),
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            results = response.points
        except Exception:
            try:
                results = self._rest_search(
                    query_emb.tolist(),
                    top_k,
                    filters,
                )
            except Exception:
                return []

        return [
            RetrievedEvidence(
                source_id=r.payload["source_id"],
                title=r.payload["title"],
                text=r.payload["text"][:900],
                score=round(r.score, 6),
                metadata={k[5:]: v for k, v in r.payload.items() if k.startswith("meta_")},
            )
            for r in results
        ]

    def _rest_search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, str] | None,
    ):
        import httpx
        from types import SimpleNamespace

        qdrant_filter = None
        if filters:
            qdrant_filter = {
                "must": [
                    {"key": f"meta_{k}", "match": {"value": v}}
                    for k, v in filters.items()
                ]
            }

        payload: dict[str, Any] = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
        if qdrant_filter:
            payload["filter"] = qdrant_filter

        response = httpx.post(
            f"{self.url}/collections/{self.collection}/points/search",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return [
            SimpleNamespace(score=item["score"], payload=item["payload"])
            for item in response.json().get("result", [])
        ]

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    @staticmethod
    def _stable_id(source_id: str) -> int:
        return int(hashlib.md5(source_id.encode()).hexdigest(), 16) % (2**63)
