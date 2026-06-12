"""RAGAS-inspired retrieval quality evaluation.

Measures precision@k, recall@k, MRR, and NDCG@k against a gold-standard
QA set with hand-annotated relevant document source_ids.
"""
from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("f1di.evaluation.retrieval")

# Gold-standard QA set — each entry maps a natural-language query to the
# source_ids of documents that contain the definitive answer.
# Source IDs match the filenames under data/knowledge/ without extension.
_GOLD_QA: list[dict[str, Any]] = [
    # Silverstone
    {"query": "front-left thermal load maggotts becketts chapel high-speed tire wear", "relevant": ["silverstone_track"], "topic": "tire"},
    {"query": "silverstone ERS deployment vale wing sector 3 harvest zone battery soc", "relevant": ["silverstone_ers"], "topic": "ers"},
    {"query": "silverstone rain crosswind weather intermediate compound crossover", "relevant": ["silverstone_weather"], "topic": "weather"},
    # Monaco
    {"query": "monaco rain intermediate tyres safety car casino tunnel exit wet", "relevant": ["monaco_weather"], "topic": "weather"},
    {"query": "monaco sector 1 hairpin blind apex braking temperature lockup", "relevant": ["monaco_track"], "topic": "braking"},
    {"query": "monaco ERS deployment low speed harvest regeneration soc lap", "relevant": ["monaco_ers"], "topic": "ers"},
    # Spa
    {"query": "spa ERS deployment kemmel raidillon eau rouge drs battery straight", "relevant": ["spa_ers"], "topic": "ers"},
    {"query": "spa rain safety car weather mixed conditions wet compound pluie", "relevant": ["spa_weather"], "topic": "weather"},
    {"query": "spa tire wear compound soft sector 2 high speed thermal blanchard", "relevant": ["spa_track"], "topic": "tire"},
    # Singapore
    {"query": "singapore undercut overcut pit window sector 2 esplanade strategy", "relevant": ["singapore_track"], "topic": "strategy"},
    {"query": "singapore ERS harvest slow corners regeneration battery soc heat", "relevant": ["singapore_ers"], "topic": "ers"},
    {"query": "singapore rain weather humidity tropical safety car night race", "relevant": ["singapore_weather"], "topic": "weather"},
    # Bahrain
    {"query": "bahrain tire cliff degradation C3 C4 sector 2 high energy load soft", "relevant": ["bahrain_track"], "topic": "tire"},
    {"query": "bahrain battery ERS straight DRS deployment sector 1 3 harvest", "relevant": ["bahrain_ers"], "topic": "ers"},
    {"query": "bahrain rain crosswind desert weather sand gusts", "relevant": ["bahrain_weather"], "topic": "weather"},
    # Monza
    {"query": "monza slipstream DRS tow drag reduction straight speed lesmo", "relevant": ["monza_track"], "topic": "strategy"},
    {"query": "monza ERS deployment parabolica chicane ascari battery energy", "relevant": ["monza_ers"], "topic": "ers"},
    # Abu Dhabi
    {"query": "abu dhabi tire strategy medium hard compound degradation low wear yas", "relevant": ["abu_dhabi_track"], "topic": "tire"},
    {"query": "abu dhabi ERS deployment straight marina battery soc sector 1", "relevant": ["abu_dhabi_ers"], "topic": "ers"},
    # Suzuka
    {"query": "suzuka sector 2 130R spoon high speed tire thermal load esses", "relevant": ["suzuka_track"], "topic": "tire"},
    {"query": "suzuka ERS deployment degner casio triangle harvest battery", "relevant": ["suzuka_ers"], "topic": "ers"},
    # Budapest / Hungary
    {"query": "budapest hungaroring high downforce slow corners tire wear medium heat", "relevant": ["budapest_track"], "topic": "tire"},
    # Zandvoort
    {"query": "zandvoort banked turn tire loading camber wind North Sea crosswind", "relevant": ["zandvoort_track"], "topic": "weather"},
    # Baku
    {"query": "baku walls street circuit braking zone turn 8 safety car lock up", "relevant": ["baku_track"], "topic": "braking"},
    # Interlagos
    {"query": "interlagos sao paulo rain safety car weather wet compound tropical", "relevant": ["interlagos_weather"], "topic": "weather"},
    # Mexico City
    {"query": "mexico city altitude thin air engine temperature brake cooling high downforce", "relevant": ["mexico_city_track"], "topic": "braking"},
    # Austin
    {"query": "austin cota sector 1 undulations tire wear rear degradation kerb", "relevant": ["austin_track"], "topic": "tire"},
    # Jeddah
    {"query": "jeddah wall proximity street Saudi arabia night DRS kerb barriers turn", "relevant": ["jeddah_track"], "topic": "strategy"},
    # Miami
    {"query": "miami pit lane undercut compound strategy tire life wall proximity", "relevant": ["miami_track"], "topic": "strategy"},
    # Spielberg / Austria
    {"query": "spielberg red bull ring short lap compound cliff tire degradation sector", "relevant": ["spielberg_track"], "topic": "tire"},
    # Barcelona
    {"query": "barcelona compound strategy medium hard high degradation long stint", "relevant": ["barcelona_track"], "topic": "tire"},
]


@dataclass
class QueryResult:
    query: str
    relevant: list[str]
    retrieved: list[str]
    p_at_1: float
    p_at_3: float
    p_at_5: float
    r_at_3: float
    r_at_5: float
    mrr: float
    ndcg_at_5: float
    topic: str


@dataclass
class RetrievalMetrics:
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    n_queries: int
    per_topic: dict[str, dict[str, float]]
    query_results: list[dict]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        return (
            f"P@1={self.precision_at_1:.3f}  P@3={self.precision_at_3:.3f}  "
            f"R@3={self.recall_at_3:.3f}  MRR={self.mrr:.3f}  "
            f"NDCG@5={self.ndcg_at_5:.3f}  n={self.n_queries}"
        )


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for r in retrieved[:k] if r in relevant) / k


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant)


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_retriever(
    retriever,
    qa_set: list[dict] | None = None,
    top_k: int = 5,
) -> RetrievalMetrics:
    """Run gold QA set against retriever; return aggregate + per-topic metrics."""
    qa = qa_set or _GOLD_QA

    available_ids: set[str] = set()
    if hasattr(retriever, "documents"):
        available_ids = {doc.source_id for doc in retriever.documents}

    p1_list, p3_list, p5_list = [], [], []
    r3_list, r5_list, mrr_list, ndcg5_list = [], [], [], []
    topic_rows: dict[str, list[dict]] = {}
    query_results: list[dict] = []

    for item in qa:
        relevant_all = set(item["relevant"])
        # Skip queries whose relevant docs aren't loaded into the retriever
        relevant = relevant_all & available_ids if available_ids else relevant_all
        if not relevant:
            continue

        results = retriever.search(item["query"], top_k=top_k)
        retrieved = [r.source_id for r in results]

        p1 = _precision_at_k(retrieved, relevant, 1)
        p3 = _precision_at_k(retrieved, relevant, 3)
        p5 = _precision_at_k(retrieved, relevant, 5)
        r3 = _recall_at_k(retrieved, relevant, 3)
        r5 = _recall_at_k(retrieved, relevant, 5)
        mrr = _reciprocal_rank(retrieved, relevant)
        ndcg5 = _ndcg_at_k(retrieved, relevant, 5)

        p1_list.append(p1); p3_list.append(p3); p5_list.append(p5)
        r3_list.append(r3); r5_list.append(r5)
        mrr_list.append(mrr); ndcg5_list.append(ndcg5)

        topic = item.get("topic", "other")
        row = {"p1": p1, "p3": p3, "r3": r3, "mrr": mrr, "ndcg5": ndcg5}
        topic_rows.setdefault(topic, []).append(row)
        query_results.append({
            "query": item["query"],
            "relevant": list(relevant),
            "retrieved_top3": retrieved[:3],
            "p@1": p1, "mrr": mrr, "ndcg@5": ndcg5,
        })

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    per_topic = {
        topic: {
            "p@1": _mean([r["p1"] for r in rows]),
            "p@3": _mean([r["p3"] for r in rows]),
            "r@3": _mean([r["r3"] for r in rows]),
            "mrr": _mean([r["mrr"] for r in rows]),
            "ndcg@5": _mean([r["ndcg5"] for r in rows]),
            "n": len(rows),
        }
        for topic, rows in topic_rows.items()
    }

    return RetrievalMetrics(
        precision_at_1=_mean(p1_list),
        precision_at_3=_mean(p3_list),
        precision_at_5=_mean(p5_list),
        recall_at_3=_mean(r3_list),
        recall_at_5=_mean(r5_list),
        mrr=_mean(mrr_list),
        ndcg_at_5=_mean(ndcg5_list),
        n_queries=len(p1_list),
        per_topic=per_topic,
        query_results=query_results,
    )


def save_eval_report(
    metrics: RetrievalMetrics,
    path: Path = Path("data/calibration/retrieval_eval.json"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_dict(), indent=2))
    logger.info("retrieval_eval_saved  %s  %s", path, metrics.summary())
