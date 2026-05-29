"""Material presets for the band-analysis tabs (pure data, no Qt)."""
from __future__ import annotations


PRESETS: dict[str, dict] = {
    "Custom": {},
    "BaNi2P2": {"a": 4.143, "lattice": "square", "omega_max_meV": 25.0},
    "Bi2212":  {"a": 5.40,  "lattice": "square", "omega_max_meV": 80.0},
    "FeSe":    {"a": 3.77,  "lattice": "square", "omega_max_meV": 15.0},
    "Cu(111)": {"a": 2.56,  "lattice": "hex",    "omega_max_meV": 5.0},
}
