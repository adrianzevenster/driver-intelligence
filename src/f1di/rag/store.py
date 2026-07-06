from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from f1di.domain.schemas import RetrievedEvidence

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

# BM25 hyperparameters: k1 controls TF saturation, b controls length normalisation.
_BM25_K1 = 1.5
_BM25_B  = 0.75

# Multiplicative score prior by source_id prefix.  Curated circuit/topic docs
# (silverstone_track, monaco_ers, etc.) receive 1.0.  Raw fastf1 session docs
# share vocabulary with every strategy query and consistently outrank curated
# knowledge on BM25 alone — dampen them.  Circuit guides duplicate specific
# track docs and cause ties; apply a mild penalty so the specific doc wins.
_SOURCE_PRIORS: dict[str, float] = {
    "fastf1_":       0.35,
    "openf1_":       0.35,
    "uploaded_text_": 0.05,
    "circuit_guides_": 0.65,
}

# Corner / landmark → canonical circuit prefix in source_id.
# Used to detect which circuit a query is about and boost matching docs.
_CIRCUIT_ALIASES: dict[str, str] = {
    # Silverstone
    "silverstone": "silverstone", "maggotts": "silverstone",
    "becketts": "silverstone", "copse": "silverstone",
    # Monaco
    "monaco": "monaco", "casino": "monaco", "tunnel": "monaco",
    # Spa-Francorchamps
    "spa": "spa", "raidillon": "spa", "kemmel": "spa", "blanchard": "spa",
    # Singapore
    "singapore": "singapore", "esplanade": "singapore",
    # Bahrain
    "bahrain": "bahrain",
    # Monza
    "monza": "monza", "parabolica": "monza", "lesmo": "monza", "ascari": "monza",
    # Abu Dhabi
    "dhabi": "abu_dhabi", "yas": "abu_dhabi",
    # Suzuka
    "suzuka": "suzuka", "130r": "suzuka", "spoon": "suzuka",
    "degner": "suzuka", "casio": "suzuka",
    # Budapest
    "budapest": "budapest", "hungaroring": "budapest",
    # Zandvoort
    "zandvoort": "zandvoort",
    # Baku
    "baku": "baku",
    # Interlagos
    "interlagos": "interlagos",
    # Mexico City
    "mexico": "mexico_city",
    # Austin / COTA
    "austin": "austin", "cota": "austin",
    # Jeddah
    "jeddah": "jeddah",
    # Miami
    "miami": "miami",
    # Spielberg / Austria
    "spielberg": "spielberg",
    # Barcelona
    "barcelona": "barcelona",
}
_CIRCUIT_BOOST = 2.0   # applied when query names the circuit and doc matches

# Topic keyword signals → expected source_id suffix.  Helps break ties when
# multiple docs from the same circuit are in the candidate set.
_TOPIC_SUFFIX_SIGNALS: dict[str, set[str]] = {
    "_ers":     {"ers", "battery", "soc", "harvest", "regeneration", "deployment", "electric"},
    "_weather": {"rain", "wet", "intermediate", "crosswind", "weather", "humidity", "tropical", "pluie", "gusts"},
    "_track":   {"apex", "hairpin", "chicane", "kerb", "sector", "corner", "lockup", "camber",
                 "undulation", "banked", "braking", "throttle", "traction", "degradation", "wear"},
}
_TOPIC_BOOST = 1.3


def _source_prior(source_id: str) -> float:
    for prefix, weight in _SOURCE_PRIORS.items():
        if source_id.startswith(prefix):
            return weight
    return 1.0


def _structural_boost(source_id: str, q_term_set: set[str]) -> float:
    """Multiplier combining circuit-name detection and topic-suffix matching."""
    boost = 1.0

    # Circuit boost: if query mentions a circuit's landmarks, its docs win.
    for term in q_term_set:
        circuit = _CIRCUIT_ALIASES.get(term)
        if circuit and source_id.startswith(circuit):
            boost *= _CIRCUIT_BOOST
            break

    # Topic boost: secondary signal for doc type within a circuit.
    for suffix, signals in _TOPIC_SUFFIX_SIGNALS.items():
        if q_term_set & signals and source_id.endswith(suffix):
            boost *= _TOPIC_BOOST
            break

    return boost


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class KnowledgeDocument:
    source_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class HybridMemoryRetriever:
    """Hybrid BM25 + optional dense retriever for local development and regression tests.

    Production replacement: Qdrant/pgvector dense+sparse retrieval with reranking.
    Pass any object with an `.encode(texts: list[str]) -> np.ndarray` interface as
    `encoder` to enable dense retrieval blended with the sparse signal.
    """

    def __init__(self, encoder: Any | None = None) -> None:
        self.documents: list[KnowledgeDocument] = []
        self.doc_terms: list[set[str]] = []
        self._tf_dicts: list[dict[str, int]] = []   # per-doc term frequencies
        self._doc_lengths: list[int] = []           # token counts for BM25 length norm
        self._avg_doc_len: float = 1.0
        self.df: dict[str, int] = {}
        self._encoder = encoder
        self._doc_embeddings: list[Any] = []
        # Tracks the highest raw BM25 score ever seen so scores are normalised
        # against a corpus-level maximum rather than per-query.  This preserves
        # cross-query discrimination: a query that finds a strong match scores
        # near 1.0 while a query whose best match is weak scores proportionally lower.
        self._global_max_score: float = 1.0

    def add_documents(self, docs: Iterable[KnowledgeDocument]) -> None:
        new_docs = list(docs)
        for doc in new_docs:
            title_tokens = tokenize(doc.title)
            body_tokens  = tokenize(doc.text)
            # Build combined TF: title terms get 3x weight to boost title matches.
            all_tokens   = title_tokens * 3 + body_tokens
            tf: dict[str, int] = {}
            for t in all_tokens:
                tf[t] = tf.get(t, 0) + 1
            terms = set(all_tokens)
            self.documents.append(doc)
            self.doc_terms.append(terms)
            self._tf_dicts.append(tf)
            self._doc_lengths.append(len(all_tokens))
            for term in terms:
                self.df[term] = self.df.get(term, 0) + 1
        # Recompute average doc length after each batch.
        if self._doc_lengths:
            self._avg_doc_len = sum(self._doc_lengths) / len(self._doc_lengths)
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

        q_term_set = set(q_terms)

        q_emb = None
        if self._encoder is not None and self._doc_embeddings:
            import numpy as np
            q_emb = self._encoder.encode([query], normalize_embeddings=True)[0]

        scored: list[tuple[float, int, KnowledgeDocument]] = []
        n = max(len(self.documents), 1)
        avgdl = max(self._avg_doc_len, 1.0)

        for idx, (doc, terms, tf, dl) in enumerate(
            zip(self.documents, self.doc_terms, self._tf_dicts, self._doc_lengths)
        ):
            if filters and any(str(doc.metadata.get(k)) != str(v) for k, v in filters.items()):
                continue

            # Proper BM25 with k1 + b length normalisation.
            bm25 = 0.0
            for t in q_terms:
                tf_t = tf.get(t, 0)
                if tf_t == 0:
                    continue
                idf = math.log((n + 1) / (self.df.get(t, 0) + 1))
                numerator   = tf_t * (_BM25_K1 + 1)
                denominator = tf_t + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
                bm25 += idf * numerator / denominator

            jaccard = len(q_term_set & terms) / max(len(q_term_set | terms), 1)
            sparse_score = (0.7 * bm25) + (0.3 * jaccard)

            if q_emb is not None:
                import numpy as np
                cosine = float(np.dot(q_emb, self._doc_embeddings[idx]))
                score = 0.35 * sparse_score + 0.65 * max(0.0, cosine)
            else:
                score = sparse_score

            score *= _source_prior(doc.source_id)
            score *= _structural_boost(doc.source_id, q_term_set)

            if score > 0:
                scored.append((score, idx, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return []

        # Update the running corpus-level maximum, then normalise against it.
        self._global_max_score = max(self._global_max_score, scored[0][0])
        normalizer = self._global_max_score

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
    for file in sorted(path.rglob("*.md")):
        text = file.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ") if text.splitlines() else file.stem
        rel = file.relative_to(path)
        subdir = rel.parts[0] if len(rel.parts) > 1 else None
        parts = file.stem.split("_")
        track_id = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
        source = subdir if subdir else "knowledge"
        metadata = {"track_id": track_id, "source": source}
        source_id = str(rel.with_suffix("")).replace("/", "_").replace("\\", "_")
        docs.append(KnowledgeDocument(source_id=source_id, title=title, text=text, metadata=metadata))
    return docs


def save_document_as_markdown(doc: KnowledgeDocument, base_path: Path) -> Path:
    subdir = doc.metadata.get("source", "knowledge")
    if subdir in ("knowledge", "unknown"):
        subdir = "fastf1"
    out_dir = base_path / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{doc.source_id}.md"
    fname.write_text(doc.text, encoding="utf-8")
    return fname
