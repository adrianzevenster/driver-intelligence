"""Ingest FastF1 race/qualifying sessions and persist them as markdown files.

Saved under data/knowledge/fastf1/ so they are loaded at every API startup
without needing a live network call. Re-running is idempotent: existing files
are overwritten with fresh content if the session data has changed.

Usage:
    uv run python scripts/ingest_fastf1_to_disk.py
    uv run python scripts/ingest_fastf1_to_disk.py --years 2023 2024 --n 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from f1di.knowledge.fastf1_ingester import _build_document, _build_qualifying_document, _CACHE_DIR
from f1di.rag.store import save_document_as_markdown

_KNOWLEDGE_DIR = Path("data/knowledge")


def main(years: list[int], n_per_year: int, include_qualifying: bool) -> None:
    import fastf1
    import warnings
    import os
    from datetime import date

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fastf1.Cache.enable_cache(_CACHE_DIR)

    saved: list[str] = []
    failed: list[str] = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            past = schedule[schedule["EventDate"].astype(str) <= str(date.today())]
            events = list(past.tail(n_per_year).iterrows())
            print(f"\n{year}: {len(events)} events")
        except Exception as e:
            print(f"  schedule failed {year}: {e}")
            continue

        for _, row in events:
            rnd = int(row["RoundNumber"])
            name = str(row["EventName"])
            print(f"  {year} R{rnd} {name} ...", end="", flush=True)

            try:
                doc = _build_document(year, name, rnd)
                save_document_as_markdown(doc, _KNOWLEDGE_DIR)
                saved.append(doc.title)
                print(" race ✓", end="")
            except Exception as e:
                failed.append(f"{year} {name} race: {e}")
                print(f" race ✗({e})", end="")

            if include_qualifying:
                try:
                    qdoc = _build_qualifying_document(year, name, rnd)
                    if qdoc:
                        save_document_as_markdown(qdoc, _KNOWLEDGE_DIR)
                        saved.append(qdoc.title)
                        print(" quali ✓", end="")
                except Exception as e:
                    failed.append(f"{year} {name} quali: {e}")
                    print(f" quali ✗({e})", end="")

            print()

    print(f"\n{'='*60}")
    print(f"Saved: {len(saved)} documents → {_KNOWLEDGE_DIR / 'fastf1'}/")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  {f}")
    print("\nRestart the API server to load the new knowledge into memory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--n", type=int, default=10, help="Events per year (most recent)")
    parser.add_argument("--no-qualifying", action="store_true")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="Ingest cold-tier history (2018–2022, 5 events/year) for the tiered retriever",
    )
    args = parser.parse_args()

    if args.cold:
        years = args.years or list(range(2018, 2023))
        n = args.n if args.years else 5
    else:
        years = args.years or [2023, 2024]
        n = args.n

    main(years, n, not args.no_qualifying)
