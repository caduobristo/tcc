"""Canonical local dataset locations; TCC_DATA_ROOT may point to another disk."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = {
    "mono_cont": "Mono_Continuous",
    "mono_disc": "Mono_Discrete",
    "poly_cont": "Poly_Continuous",
    "poly_disc": "Poly_Discrete",
}


def data_root() -> Path:
    return Path(os.environ.get("TCC_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()


def guitar_fx_root(representation: str = "raw") -> Path:
    """Return a location, without implying that its dataset has been downloaded."""
    if representation == "raw":
        return data_root() / "raw" / "guitar_fx_dist"
    if representation in {"baseline", "mel16", "mel32"}:
        return data_root() / "processed" / "guitar_fx_dist" / representation
    raise ValueError(f"Unknown GUITAR-FX representation: {representation}")


def baseline_scenario_root(scenario: str) -> Path:
    name = SCENARIOS.get(scenario, scenario)
    if name not in SCENARIOS.values():
        raise ValueError(f"Unknown GUITAR-FX scenario: {scenario}")
    path = guitar_fx_root("baseline") / name
    if not (path / "Features").is_dir():
        raise FileNotFoundError(
            f"Original baseline features are missing at {path / 'Features'}. "
            "The local mel16/mel32 archives are different representations (198 x 128), "
            "not replacements for the original baseline. See data/README.md."
        )
    return path
