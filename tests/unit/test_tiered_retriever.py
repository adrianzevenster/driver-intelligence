from __future__ import annotations


from f1di.rag.store import KnowledgeDocument
from f1di.rag.tiered_retriever import TieredRetriever, _is_recent


def _doc(source_id: str, year: str | None = None, text: str = "f1 tire data") -> KnowledgeDocument:
    meta = {"year": year} if year else {}
    return KnowledgeDocument(source_id=source_id, title=source_id, text=text, metadata=meta)


class TestIsRecent:
    def test_year_metadata_hot(self):
        assert _is_recent(_doc("x", year="2025"), {2024, 2025})

    def test_year_metadata_cold(self):
        assert not _is_recent(_doc("x", year="2019"), {2024, 2025})

    def test_source_id_year_token(self):
        # year embedded in source_id, no metadata
        assert not _is_recent(_doc("race_2020_r5"), {2024, 2025})
        assert _is_recent(_doc("race_2024_r5"), {2024, 2025})

    def test_no_year_info_defaults_cold(self):
        assert not _is_recent(_doc("generic_doc"), {2024, 2025})


class TestTieredRetriever:
    def _make_retriever(self) -> TieredRetriever:
        r = TieredRetriever(hot_seasons=2, min_hot_results=2)
        # Override hot_years for determinism
        r.hot_years = {2024, 2025}
        return r

    def test_split_hot_cold(self):
        r = self._make_retriever()
        r.add_documents([
            _doc("hot_2024", year="2024", text="soft tire silverstone 2024 wear data"),
            _doc("cold_2019", year="2019", text="soft tire silverstone 2019 wear data"),
            _doc("hot_2025", year="2025", text="medium tire bahrain 2025 wear data"),
        ])
        assert r.hot_document_count == 2
        assert r.cold_document_count == 1

    def test_hot_boost_applied(self):
        r = self._make_retriever()
        r.add_documents([
            _doc("h1", year="2024", text="tire wear degradation cliff soft compound"),
            _doc("h2", year="2024", text="tire wear degradation cliff medium compound"),
            _doc("h3", year="2025", text="tire wear degradation cliff hard compound"),
        ])
        results = r.search("tire wear degradation cliff", top_k=3)
        assert all(hasattr(res, "score") for res in results)
        assert all(res.metadata.get("tier") == "hot" for res in results)
        # Scores are boosted — all ≤ 1.0 after clamping
        assert all(res.score <= 1.0 for res in results)

    def test_fallback_to_cold_when_hot_insufficient(self):
        r = self._make_retriever()
        r.add_documents([
            _doc("c1", year="2019", text="tire wear strategy rain wet conditions"),
            _doc("c2", year="2020", text="tire wear degradation soft compound"),
        ])
        results = r.search("tire wear", top_k=2)
        assert len(results) > 0
        assert all(res.metadata.get("tier") == "cold" for res in results)

    def test_total_documents(self):
        r = self._make_retriever()
        r.add_documents([_doc(f"d{i}", year=str(2020 + i)) for i in range(5)])
        assert len(r.documents) == 5

    def test_source_counts_prefixed(self):
        r = self._make_retriever()
        r.add_documents([_doc("h", year="2025"), _doc("c", year="2019")])
        counts = r.source_counts()
        assert any(k.startswith("hot:") for k in counts)
        assert any(k.startswith("cold:") for k in counts)
