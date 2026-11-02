"""Recorta canciones de musica/ (y musica_prueba/) en clips WAV, uno por versión."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    AUDIO_EXTS,
    MIN_DURATION_SEC,
    MIN_FULL_RATIO,
    SAMPLE_RATE,
    clip_seconds,
    clips_dir,
    ensure_dirs,
    source_dir,
)


def iter_source_tracks(split: str = "train") -> list[tuple[Path, str]]:
    root = source_dir(split)
    if not root.exists():
        if split == "train":
            raise FileNotFoundError(f"No existe la carpeta de música: {root}")
        return []

    tracks: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = path.relative_to(root)
        genre = rel.parts[0] if len(rel.parts) > 1 else "sin_genero"
        tracks.append((path, genre))
    return tracks


def _safe_stem(path: Path) -> str:
    return path.stem


def centered_slice(y: np.ndarray, sr: int, seconds: int) -> np.ndarray:
    n_target = int(seconds * sr)
    if len(y) <= n_target:
        return y
    start = max(0, (len(y) - n_target) // 2)
    return y[start : start + n_target]


def load_and_trim(path: Path) -> tuple[np.ndarray, int]:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y, sr


def process_track(
    src: Path,
    genre: str,
    force: bool = False,
    split: str = "train",
    versions: tuple[int, ...] | None = None,
) -> dict:
    rel_name = f"{genre}/{src.name}"
    versions = versions or clip_seconds()
    needed = []
    for seconds in versions:
        out_dir = clips_dir(seconds, split) / genre
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_safe_stem(src)}.wav"
        if force or not out_path.exists():
            needed.append(seconds)

    if not needed:
        return {"archivo": rel_name, "estado": "omitido", "detalle": "clips ya existen"}

    try:
        y, sr = load_and_trim(src)
    except Exception as exc:  # noqa: BLE001
        return {
            "archivo": rel_name,
            "estado": "error",
            "detalle": f"no se pudo cargar ({exc}). ¿ffmpeg instalado?",
        }

    duration = len(y) / sr if sr else 0.0
    if duration < MIN_DURATION_SEC:
        return {
            "archivo": rel_name,
            "estado": "saltado",
            "detalle": f"duración {duration:.1f}s < {MIN_DURATION_SEC}s",
        }

    hechos, omitidos = [], []
    for seconds in needed:
        # Cada versión solo usa canciones que alcanzan su duración (clips completos).
        if duration < seconds * MIN_FULL_RATIO:
            omitidos.append(seconds)
            continue
        out_path = clips_dir(seconds, split) / genre / f"{_safe_stem(src)}.wav"
        sf.write(str(out_path), centered_slice(y, sr, seconds), sr)
        hechos.append(seconds)

    detalle = f"duración {duration:.1f}s; clips {hechos}"
    if omitidos:
        detalle += f"; muy corta para {omitidos}s"
    return {"archivo": rel_name, "estado": "ok" if hechos else "saltado", "detalle": detalle}


def run_clips(force: bool = False, split: str = "train", seconds: int | None = None) -> list[dict]:
    ensure_dirs()
    tracks = iter_source_tracks(split)
    if not tracks:
        print(f"No se encontraron audios en {source_dir(split)}")
        return []

    versions = (seconds,) if seconds else clip_seconds()
    etiqueta = "de prueba " if split == "prueba" else ""
    print(f"Recortando {len(tracks)} pista(s) {etiqueta}a {list(versions)} s ...")
    results = []
    for src, genre in tracks:
        info = process_track(src, genre, force=force, split=split, versions=versions)
        results.append(info)
        print(f"  [{info['estado']}] {info['archivo']} — {info['detalle']}")
    return results


if __name__ == "__main__":
    run_clips()
