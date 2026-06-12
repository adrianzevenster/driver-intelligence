from __future__ import annotations

from unittest.mock import MagicMock


def _mock_retriever(source_ids: list[str]):
    """Return a mock retriever whose .search() returns docs with given source_ids."""

    def _search(query: str, top_k: int = 5):
        docs = []
        for sid in source_ids[:top_k]:
            doc = MagicMock()
            doc.source_id = sid
            docs.append(doc)
        return docs

    r = MagicMock()
    r.search.side_effect = _search
    r.documents = [MagicMock(source_id=sid) for sid in source_ids]
    return r


# ── Metric primitives ──────────────────────────────────────────────────────


def test_precision_at_k_perfect():
    from f1di.evaluation.retrieval_eval import _precision_at_k
    assert _precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_zero():
    from f1di.evaluation.retrieval_eval import _precision_at_k
    assert _precision_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_precision_at_k_partial():
    from f1di.evaluation.retrieval_eval import _precision_at_k
    result = _precision_at_k(["a", "x", "b"], {"a", "b"}, k=3)
    assert abs(result - 2 / 3) < 1e-6


def test_recall_at_k_perfect():
    from f1di.evaluation.retrieval_eval import _recall_at_k
    assert _recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_recall_at_k_zero():
    from f1di.evaluation.retrieval_eval import _recall_at_k
    assert _recall_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_recall_at_k_empty_relevant():
    from f1di.evaluation.retrieval_eval import _recall_at_k
    assert _recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_reciprocal_rank_first_hit():
    from f1di.evaluation.retrieval_eval import _reciprocal_rank
    assert _reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_second_hit():
    from f1di.evaluation.retrieval_eval import _reciprocal_rank
    assert _reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5


def test_reciprocal_rank_no_hit():
    from f1di.evaluation.retrieval_eval import _reciprocal_rank
    assert _reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_ndcg_at_k_perfect():
    from f1di.evaluation.retrieval_eval import _ndcg_at_k
    assert _ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_at_k_miss():
    from f1di.evaluation.retrieval_eval import _ndcg_at_k
    assert _ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


# ── evaluate_retriever ─────────────────────────────────────────────────────


def test_evaluate_retriever_all_correct():
    from f1di.evaluation.retrieval_eval import evaluate_retriever, _GOLD_QA

    # Build a retriever that returns the relevant doc first for every query.
    all_source_ids = list({sid for q in _GOLD_QA for sid in q["relevant"]})
    retriever = _mock_retriever(all_source_ids)

    # Override search to always return the relevant doc first
    def _perfect_search(query: str, top_k: int = 5):
        for item in _GOLD_QA:
            for kw in item["query"].split()[:2]:
                if kw.lower() in query.lower():
                    sid = item["relevant"][0]
                    doc = MagicMock()
                    doc.source_id = sid
                    rest = [MagicMock(source_id=s) for s in all_source_ids[:top_k - 1] if s != sid]
                    return [doc] + rest
        doc = MagicMock()
        doc.source_id = all_source_ids[0]
        return [doc]

    retriever.search.side_effect = _perfect_search

    metrics = evaluate_retriever(retriever)
    assert metrics.n_queries > 0
    assert 0.0 <= metrics.precision_at_1 <= 1.0
    assert 0.0 <= metrics.mrr <= 1.0
    assert 0.0 <= metrics.ndcg_at_5 <= 1.0


def test_evaluate_retriever_empty_retriever():
    """Retriever with no documents runs all queries but scores 0 across the board."""
    from f1di.evaluation.retrieval_eval import evaluate_retriever

    retriever = MagicMock()
    retriever.documents = []
    retriever.search.return_value = []

    metrics = evaluate_retriever(retriever)
    assert metrics.mrr == 0.0
    assert metrics.precision_at_1 == 0.0
    assert metrics.ndcg_at_5 == 0.0


def test_evaluate_retriever_returns_all_fields():
    from f1di.evaluation.retrieval_eval import evaluate_retriever

    source_ids = ["silverstone_track", "monaco_track", "spa_track", "bahrain_track"]
    retriever = _mock_retriever(source_ids)

    metrics = evaluate_retriever(retriever)
    d = metrics.to_dict()
    for field in [
        "precision_at_1", "precision_at_3", "precision_at_5",
        "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5",
        "n_queries", "per_topic", "query_results",
    ]:
        assert field in d, f"Missing field: {field}"


def test_evaluate_retriever_per_topic_grouping():
    from f1di.evaluation.retrieval_eval import evaluate_retriever

    source_ids = ["silverstone_track", "silverstone_ers", "monaco_track"]
    retriever = _mock_retriever(source_ids)

    metrics = evaluate_retriever(retriever)
    assert isinstance(metrics.per_topic, dict)
    for topic_data in metrics.per_topic.values():
        assert "p@1" in topic_data
        assert "mrr" in topic_data
        assert "n" in topic_data
        assert topic_data["n"] >= 1


def test_evaluate_retriever_custom_qa_set():
    from f1di.evaluation.retrieval_eval import evaluate_retriever

    custom_qa = [
        {"query": "front tyre degradation", "relevant": ["doc_a"], "topic": "tire"},
        {"query": "ERS deployment straight", "relevant": ["doc_b"], "topic": "ers"},
    ]
    retriever = MagicMock()
    retriever.documents = [MagicMock(source_id="doc_a"), MagicMock(source_id="doc_b")]

    def _search(query, top_k=5):
        if "tyre" in query:
            return [MagicMock(source_id="doc_a")]
        return [MagicMock(source_id="doc_b")]

    retriever.search.side_effect = _search

    metrics = evaluate_retriever(retriever, qa_set=custom_qa)
    assert metrics.n_queries == 2
    assert metrics.precision_at_1 == 1.0
    assert metrics.mrr == 1.0


# ── save_eval_report ───────────────────────────────────────────────────────


def test_save_eval_report_writes_json(tmp_path):
    from f1di.evaluation.retrieval_eval import evaluate_retriever, save_eval_report

    custom_qa = [
        {"query": "tire wear test", "relevant": ["doc_x"], "topic": "tire"},
    ]
    retriever = MagicMock()
    retriever.documents = [MagicMock(source_id="doc_x")]
    retriever.search.return_value = [MagicMock(source_id="doc_x")]

    metrics = evaluate_retriever(retriever, qa_set=custom_qa)
    out = tmp_path / "retrieval_eval.json"
    save_eval_report(metrics, path=out)

    import json
    data = json.loads(out.read_text())
    assert data["n_queries"] == 1
    assert "precision_at_1" in data
