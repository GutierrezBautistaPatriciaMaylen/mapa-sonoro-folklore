"""Familias de rasgos del audio (timbre, armonía, ritmo, energía) y selección de columnas."""

from __future__ import annotations

import pandas as pd

from config import EXCLUIR_RASGOS

META_COLS = ["archivo", "genero", "duracion_clip", "duracion_real", "ruta"]

FAMILIAS: dict[str, tuple[str, ...]] = {
    "Timbre": ("mfcc", "bandwidth", "centroid", "rolloff", "zcr", "contrast"),
    "Armonía": ("chroma",),
    "Ritmo": ("tempo_bpm", "beat_rate", "beat_interval", "onset"),
    "Energía": ("rms",),
}
# Rasgos que dependen del volumen de la grabación más que del género.
VOLUMEN = ("rms", "mfcc_mean_00", "mfcc_std_00")


def familia(col: str) -> str:
    for nombre, prefijos in FAMILIAS.items():
        if any(col == p or col.startswith(p + "_") for p in prefijos):
            return nombre
    return "Otros"


def columnas_rasgo(df: pd.DataFrame, excluir: tuple[str, ...] | None = None) -> list[str]:
    """Columnas de rasgos (sin metadatos), quitando los prefijos de config.EXCLUIR_RASGOS."""
    excluir = EXCLUIR_RASGOS if excluir is None else excluir
    return [c for c in df.columns if c not in META_COLS and not any(c == e or c.startswith(e) for e in excluir)]


def columnas_de(df: pd.DataFrame, familias: list[str], sin_volumen: bool = False) -> list[str]:
    cols = [c for c in columnas_rasgo(df, excluir=()) if familia(c) in familias]
    if sin_volumen:
        cols = [c for c in cols if not any(c == v or c.startswith(v) for v in VOLUMEN)]
    return cols
