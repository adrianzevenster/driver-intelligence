from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def make_retriever():
    from f1di.config.settings import settings
    from f1di.rag.store import HybridMemoryRetriever

    if settings.vector_backend == "qdrant":
        try:
            from f1di.rag.qdrant_backend import QdrantHybridRetriever
            retriever = QdrantHybridRetriever(
                url=settings.qdrant_url,
                collection=settings.qdrant_collection,
                model_name=settings.embedding_model,
            )
            return retriever
        except ImportError:
            logger.warning(
                "qdrant-client or sentence-transformers not installed; "
                "falling back to in-memory retriever. "
                "Install with: pip install 'f1-driver-intelligence[rag]'"
            )
        except Exception as exc:
            logger.warning(
                "Qdrant unavailable at %s (%s: %s); falling back to in-memory retriever. "
                "Set F1DI_VECTOR_BACKEND=memory to suppress this warning.",
                settings.qdrant_url, type(exc).__name__, exc,
            )

    if settings.vector_backend == "tiered":
        try:
            from f1di.rag.tiered_retriever import TieredRetriever
            return TieredRetriever(
                hot_seasons=settings.tiered_hot_seasons,
                min_hot_results=settings.tiered_min_hot_results,
            )
        except ImportError:
            logger.warning("Tiered retriever dependencies not installed; falling back to in-memory retriever.")

    return HybridMemoryRetriever()
