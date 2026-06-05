from __future__ import annotations

import argparse
from pathlib import Path
from f1di.regression.runner import run_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/scenarios/synthetic_race.jsonl")
    parser.add_argument("--output", default="data/scenarios/regression_report.json")
    args = parser.parse_args()
    print(run_replay(Path(args.input), Path(args.output)))


if __name__ == "__main__":
    main()
