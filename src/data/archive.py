"""Read GUITAR-FX processed arrays directly from local ZIP archives."""

import io
import csv
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from src.data.paths import SCENARIOS, data_root


class GuitarFxArchive:
    """Context manager for one scenario/variant; it never extracts or rewrites data."""

    def __init__(self, scenario: str, variant: str, root: Path | None = None):
        self.scenario = SCENARIOS.get(scenario, scenario)
        if self.scenario not in SCENARIOS.values() or variant not in {"mel16", "mel32"}:
            raise ValueError("Use a known GUITAR-FX scenario and mel16 or mel32.")
        self.variant = variant
        self.root = Path(root) if root is not None else data_root()
        short_name = {v: k for k, v in SCENARIOS.items()}[self.scenario]
        candidates = sorted((self.root / "archives/guitar_fx_dist").glob(f"{short_name}_{variant}-*.zip"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected exactly one archive for {self.scenario}/{variant}: {candidates}")
        self.path = candidates[0]
        self.zip = None
        self._settings = {}

    def __enter__(self):
        self.zip = ZipFile(self.path)
        return self

    def __exit__(self, *_):
        self.zip.close()
        self.zip = None
        self._settings.clear()

    def iter_samples(self, exclude_effects=("MT2",), include_nofx=False):
        if self.zip is None:
            raise RuntimeError("Use GuitarFxArchive inside a with statement.")
        for member in self.zip.infolist():
            parts = member.filename.split("/")
            if member.is_dir() or len(parts) != 3 or parts[0] != self.scenario:
                continue
            effect, filename = parts[1:]
            if effect in exclude_effects or (effect.startswith("_NoFX") and not include_nofx):
                continue
            if filename.endswith(".npy"):
                yield effect, filename[:-4]

    def class_counts(self, **kwargs):
        return dict(Counter(effect for effect, _ in self.iter_samples(**kwargs)))

    def read_feature(self, effect: str, filename: str) -> np.ndarray:
        if self.zip is None:
            raise RuntimeError("Use GuitarFxArchive inside a with statement.")
        for component in (effect, filename):
            if not component or "/" in component or "\\" in component or component in {".", ".."}:
                raise ValueError("Effect and filename must be simple names.")
        stem = filename[:-4] if filename.endswith(".npy") else filename
        content = self.zip.read(f"{self.scenario}/{effect}/{stem}.npy")
        array = np.load(io.BytesIO(content), allow_pickle=False)
        if array.shape != (198, 128) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Unexpected feature representation: {effect}/{stem}")
        return array

    def read_settings(self, effect: str, filename: str) -> dict:
        """Return parameters from the derived CSV, rejecting unvalidated metadata."""
        if not effect or "/" in effect or "\\" in effect or effect in {".", ".."}:
            raise ValueError("Effect must be a simple name.")
        if effect not in self._settings:
            path = self.root / "metadata/guitar_fx_dist/validated" / self.scenario / effect / "proc_settings.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Validated metadata missing: {path}. Run scripts/audit_datasets.py audit.")
            with path.open(encoding="utf-8", newline="") as stream:
                self._settings[effect] = {Path(row["filename"]).stem: row for row in csv.DictReader(stream)}
        stem = filename[:-4] if filename.endswith((".npy", ".wav")) else filename
        return dict(self._settings[effect][stem])
