"""Rutas, versiones y constantes del pipeline de clustering folklórico.

Una VERSIÓN es solo una duración de clip (v1 = 30 s, v2 = 60 s, ...). Todo lo
demás (features, PCA, algoritmos, métricas) se aplica igual a cada versión.
Las versiones se guardan en data/versiones.json y se pueden ampliar con
`python src/main.py --all --duracion 120`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUSICA_DIR = ROOT / "musica"
MUSICA_PRUEBA_DIR = ROOT / "musica_prueba"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
BITACORA_DIR = ROOT / "bitacora"
VERSIONES_JSON = DATA_DIR / "versiones.json"

SAMPLE_RATE = 22050
MIN_DURATION_SEC = 20.0
# Un clip cuenta como "completo" si dura al menos este % de la versión.
MIN_FULL_RATIO = 0.95
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

N_MFCC = 13
RANDOM_STATE = 42

VERSIONES_BASE = {"v1": 30, "v2": 60}


def load_versions() -> dict[str, int]:
    """Devuelve {'v1': 30, 'v2': 60, ...} leyendo data/versiones.json."""
    if VERSIONES_JSON.exists():
        try:
            raw = json.loads(VERSIONES_JSON.read_text(encoding="utf-8"))
            versions = {str(k): int(v) for k, v in raw.items()}
            if versions:
                return versions
        except (ValueError, OSError):
            pass
    return dict(VERSIONES_BASE)


def clip_seconds() -> tuple[int, ...]:
    return tuple(load_versions().values())


def version_name(seconds: int) -> str:
    for name, sec in load_versions().items():
        if sec == seconds:
            return name
    return f"{seconds}s"


def register_version(seconds: int) -> str:
    """Añade una versión nueva (si no existe) y devuelve su nombre."""
    versions = load_versions()
    for name, sec in versions.items():
        if sec == int(seconds):
            return name
    name = f"v{len(versions) + 1}"
    versions[name] = int(seconds)
    VERSIONES_JSON.parent.mkdir(parents=True, exist_ok=True)
    VERSIONES_JSON.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    return name


def source_dir(split: str = "train") -> Path:
    return MUSICA_PRUEBA_DIR if split == "prueba" else MUSICA_DIR


def clips_dir(seconds: int, split: str = "train") -> Path:
    prefix = "prueba_clips" if split == "prueba" else "clips"
    return DATA_DIR / f"{prefix}_{seconds}s"


def features_csv(seconds: int, split: str = "train") -> Path:
    prefix = "prueba_features" if split == "prueba" else "features"
    return DATA_DIR / f"{prefix}_{seconds}s.csv"


def results_dir(seconds: int) -> Path:
    return RESULTS_DIR / f"{seconds}s"


def models_dir(seconds: int) -> Path:
    return MODELS_DIR / f"{seconds}s"


def ensure_dirs() -> None:
    for folder in (DATA_DIR, RESULTS_DIR, MODELS_DIR, BITACORA_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    for seconds in clip_seconds():
        clips_dir(seconds).mkdir(parents=True, exist_ok=True)
        results_dir(seconds).mkdir(parents=True, exist_ok=True)
        models_dir(seconds).mkdir(parents=True, exist_ok=True)
