from __future__ import annotations


def make_retriever():
    from f1di.config.settings import settings
    from f1di.rag.store import HybridMemoryRetriever

    if settings.vector_backend == "qdrant":
        from f1di.rag.qdrant_backend import QdrantHybridRetriever
        return QdrantHybridRetriever(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            model_name=settings.embedding_model,
        )

    if settings.vector_backend == "tiered":
        from f1di.rag.tiered_retriever import TieredRetriever
        return TieredRetriever(
            hot_seasons=settings.tiered_hot_seasons,
            min_hot_results=settings.tiered_min_hot_results,
        )

    return HybridMemoryRetriever()
