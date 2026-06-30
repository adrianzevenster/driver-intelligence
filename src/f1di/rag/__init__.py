from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _make_memory_retriever(settings=None):
    from f1di.rag.store import HybridMemoryRetriever
    model_name = (settings.embedding_model if settings else None) or "all-MiniLM-L6-v2"
    offline = bool(settings.embedding_offline) if settings and hasattr(settings, "embedding_offline") else False
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name, local_files_only=offline)
        logger.info("HybridMemoryRetriever: dense encoder loaded (%s)", model_name)
        return HybridMemoryRetriever(encoder=encoder)
    except Exception as exc:
        logger.debug("Dense encoder unavailable (%s), using sparse-only retriever", exc)
        return HybridMemoryRetriever()


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

    return _make_memory_retriever(settings)
