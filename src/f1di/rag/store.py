from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from f1di.domain.schemas import RetrievedEvidence

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class KnowledgeDocument:
    source_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class HybridMemoryRetriever:
    """Hybrid BM25-like + optional dense retriever for local development and regression tests.

    Production replacement: Qdrant/pgvector dense+sparse retrieval with reranking.
    Pass any object with an `.encode(texts: list[str]) -> np.ndarray` interface as
    `encoder` to enable dense retrieval blended with the sparse signal.
    """

    def __init__(self, encoder: Any | None = None) -> None:
        self.documents: list[KnowledgeDocument] = []
        self.doc_terms: list[set[str]] = []
        self.df: dict[str, int] = {}
        self._encoder = encoder
        self._doc_embeddings: list[Any] = []

    def add_documents(self, docs: Iterable[KnowledgeDocument]) -> None:
        new_docs = list(docs)
        for doc in new_docs:
            terms = set(tokenize(doc.title + " " + doc.text))
            self.documents.append(doc)
            self.doc_terms.append(terms)
            for term in terms:
                self.df[term] = self.df.get(term, 0) + 1
        if self._encoder is not None and new_docs:
            texts = [d.title + " " + d.text[:500] for d in self.documents]
            embs = self._encoder.encode(texts, normalize_embeddings=True)
            self._doc_embeddings = [embs[i] for i in range(len(self.documents))]

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in self.documents:
            src = doc.metadata.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def search(self, query: str, *, top_k: int = 5, filters: dict[str, str] | None = None) -> list[RetrievedEvidence]:
        q_terms = tokenize(query)
        if not q_terms:
            return []

        q_emb = None
        if self._encoder is not None and self._doc_embeddings:
            import numpy as np
            q_emb = self._encoder.encode([query], normalize_embeddings=True)[0]

        scored: list[tuple[float, int, KnowledgeDocument]] = []
        n = max(len(self.documents), 1)
        for idx, (doc, terms) in enumerate(zip(self.documents, self.doc_terms)):
            if filters and any(str(doc.metadata.get(k)) != str(v) for k, v in filters.items()):
                continue
            text_tokens = tokenize(doc.text)
            tf = {t: text_tokens.count(t) for t in set(q_terms)}
            bm25_like = sum((tf[t] * math.log((n + 1) / (self.df.get(t, 0) + 1))) for t in q_terms)
            jaccard = len(set(q_terms) & terms) / max(len(set(q_terms) | terms), 1)
            sparse_score = (0.7 * bm25_like) + (0.3 * jaccard)
            if q_emb is not None:
                import numpy as np
                cosine = float(np.dot(q_emb, self._doc_embeddings[idx]))
                score = 0.6 * sparse_score + 0.4 * max(0.0, cosine)
            else:
                score = sparse_score
            if score > 0:
                scored.append((score, idx, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return []

        max_score = scored[0][0]
        normalizer = max_score if max_score > 0 else 1.0

        return [
            RetrievedEvidence(
                source_id=doc.source_id,
                title=doc.title,
                text=doc.text[:900],
                score=round(score / normalizer, 6),
                metadata=doc.metadata,
            )
            for score, _idx, doc in scored[:top_k]
        ]


def load_markdown_knowledge(path: Path) -> list[KnowledgeDocument]:
    docs: list[KnowledgeDocument] = []
    for file in sorted(path.glob("*.md")):
        text = file.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ") if text.splitlines() else file.stem
        track_id = file.stem.split("_")[0] if "_" in file.stem else file.stem
        metadata = {"track_id": track_id, "source": "knowledge"}
        docs.append(KnowledgeDocument(source_id=file.stem, title=title, text=text, metadata=metadata))
    return docs
