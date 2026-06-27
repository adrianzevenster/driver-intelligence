from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CircuitThresholds:
    wear_critical: float = 0.78
    wear_warning: float = 0.66
    brake_temp_critical_c: float = 910.0
    fl_degradation_pressure_critical: float = 0.72
    fl_degradation_pressure_warning: float = 0.60
    rain_warning: float = 0.35
    battery_soc_warning: float = 0.35
    crosswind_watch: float = 12.0
    # Pit-lane time loss (entry deceleration + pit lane traversal + box stop + exit
    # acceleration). Measured per-circuit from FastF1 PitInTime → PitOutTime;
    # global default 22.0s used when no circuit-specific value is calibrated.
    pit_loss_s: float = 22.0


_DEFAULTS = CircuitThresholds()
_REGISTRY: dict[str, CircuitThresholds] = {}
_LOADED = False
from f1di.agents.classifier_utils import _CALIBRATION_DIR
_PATH = _CALIBRATION_DIR / "thresholds.json"


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    if not _PATH.exists():
        return
    try:
        data = json.loads(_PATH.read_text())
        for track_id, vals in data.items():
            fields = {k: v for k, v in vals.items() if k in CircuitThresholds.__dataclass_fields__}
            _REGISTRY[track_id] = CircuitThresholds(**fields)
    except Exception:
        pass


def get(track_id: str) -> CircuitThresholds:
    _load()
    return _REGISTRY.get(track_id, _REGISTRY.get("default", _DEFAULTS))


def save(registry: dict[str, CircuitThresholds], path: Path = _PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: asdict(v) for k, v in registry.items()}, indent=2))
