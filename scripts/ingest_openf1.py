#!/usr/bin/env python3
"""Fetch recent race sessions from OpenF1 and index them into Qdrant.

Usage:
    python scripts/ingest_openf1.py                  # last 8 races from current + previous year
    python scripts/ingest_openf1.py --years 2025     # 2025 only
    python scripts/ingest_openf1.py --years 2024 2025 --n 15
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f1di.rag import make_retriever
from f1di.knowledge.openf1_ingester import ingest

parser = argparse.ArgumentParser(description="Ingest OpenF1 race data into the knowledge base.")
parser.add_argument("--years", nargs="+", type=int, default=None, help="Calendar years to ingest (default: current + previous)")
parser.add_argument("--n", type=int, default=8, help="Max races per year (default: 8)")
args = parser.parse_args()

print("Building retriever…")
retriever = make_retriever()

print(f"Fetching OpenF1 race sessions (years={args.years or 'current+prev'}, n={args.n})…")
ingested = ingest(retriever, years=args.years, n_per_year=args.n)

print(f"\nIndexed {len(ingested)} sessions:")
for title in sorted(ingested):
    print(f"  ✓ {title}")
print(f"\nKnowledge base now has {len(retriever.documents)} documents.")
