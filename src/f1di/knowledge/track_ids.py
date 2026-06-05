from __future__ import annotations

# Canonical track IDs used across all sources and the frontend.
# Keys = any lowercase variant (from FastF1 Location or Jolpica circuitId).
# Values = canonical ID that must match data/knowledge/{id}_*.md filenames and frontend TRACKS list.

_MAP: dict[str, str] = {
    # Silverstone
    "silverstone": "silverstone",
    # Monza
    "monza": "monza",
    # Spa
    "spa": "spa",
    "spa-francorchamps": "spa",
    "spa francorchamps": "spa",
    # Monaco
    "monaco": "monaco",
    "monte_carlo": "monaco",
    "monte-carlo": "monaco",
    "monte carlo": "monaco",
    # Suzuka
    "suzuka": "suzuka",
    # Bahrain
    "bahrain": "bahrain",
    "sakhir": "bahrain",
    # Melbourne
    "melbourne": "melbourne",
    "albert park": "melbourne",
    "albert_park": "melbourne",
    # Barcelona
    "barcelona": "barcelona",
    "catalunya": "barcelona",
    # Baku
    "baku": "baku",
    # Singapore
    "singapore": "singapore",
    "marina bay": "singapore",
    "marina_bay": "singapore",
    # Austin / COTA
    "austin": "austin",
    "americas": "austin",
    # Interlagos / São Paulo
    "interlagos": "interlagos",
    "são paulo": "interlagos",
    "sao paulo": "interlagos",
    # Abu Dhabi
    "abu dhabi": "abu_dhabi",
    "abu_dhabi": "abu_dhabi",
    "yas marina": "abu_dhabi",
    "yas_marina": "abu_dhabi",
    "yas island": "abu_dhabi",
    # Jeddah
    "jeddah": "jeddah",
    # Shanghai
    "shanghai": "shanghai",
    # Miami
    "miami": "miami",
    # Imola
    "imola": "imola",
    # Montreal
    "montreal": "montreal",
    "montréal": "montreal",
    "villeneuve": "montreal",
    # Red Bull Ring / Spielberg
    "spielberg": "spielberg",
    "red_bull_ring": "spielberg",
    "red bull ring": "spielberg",
    # Hungaroring / Budapest
    "budapest": "budapest",
    "hungaroring": "budapest",
    # Zandvoort
    "zandvoort": "zandvoort",
    # Mexico City
    "mexico city": "mexico_city",
    "mexico_city": "mexico_city",
    "rodriguez": "mexico_city",
    # Las Vegas
    "las vegas": "las_vegas",
    "las_vegas": "las_vegas",
    "vegas": "las_vegas",
    # Qatar / Lusail
    "lusail": "lusail",
    "losail": "lusail",
}


def canonical(raw: str) -> str:
    """Return the canonical track ID for any source-specific name/id."""
    key = raw.lower().strip()
    if key in _MAP:
        return _MAP[key]
    key2 = key.replace("_", " ").replace("-", " ")
    if key2 in _MAP:
        return _MAP[key2]
    return key.replace(" ", "_").replace("-", "_")
