"""Download the two official checkpoints, recording local SHA-256; no training."""
from pathlib import Path
import hashlib
import json
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.transfer import WEIGHTS_ROOT, WEIGHTS, load_frozen_encoder

SOURCES = {
    "passt": {"source": "https://github.com/kkoutini/PaSST/releases/tag/v0.0.1-audioset",
              "url": "https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt"},
    "htsat": {"source": "https://github.com/RetroCirce/HTS-Audio-Transformer",
              "url": "https://drive.google.com/file/d/1OK8a5XuMVLyeVKF117L8pfxeZYdfSDZv/view"}}


def prepare():
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, source in SOURCES.items():
        target = WEIGHTS_ROOT / WEIGHTS[name]
        if not target.exists():
            if name == "passt":
                partial = target.with_suffix(target.suffix + ".partial")
                urllib.request.urlretrieve(source["url"], partial)
                partial.rename(target)
            else:
                import gdown
                result = gdown.download(id="1OK8a5XuMVLyeVKF117L8pfxeZYdfSDZv", output=str(target), resume=True)
                if result is None:
                    raise OSError("Official HTS-AT checkpoint download failed")
        # Require complete, exactly matching model weights; never substitute random weights.
        encoder = load_frozen_encoder(name)
        del encoder
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        records[name] = {**source, "filename": target.name, "bytes": target.stat().st_size,
                         "sha256": digest.hexdigest(), "strict_state_dict_validated": True}
    (WEIGHTS_ROOT / "sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    prepare()
