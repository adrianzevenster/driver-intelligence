from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from f1di.domain.schemas import TelemetryWindow


def read_windows(path: Path) -> Iterable[TelemetryWindow]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield TelemetryWindow.model_validate(json.loads(line))
