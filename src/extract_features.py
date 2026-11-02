"""Extrae un vector de features por clip y guarda CSV."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    N_MFCC,
    SAMPLE_RATE,
    clip_seconds,
    clips_dir,
    ensure_dirs,
    features_csv,
)


def _mean_std(feat: np.ndarray, prefix: str) -> dict[str, float]:
    feat = np.atleast_2d(feat)
    out = {}
    means = np.mean(feat, axis=1)
    stds = np.std(feat, axis=1)
    for i, (m, s) in enumerate(zip(means, stds)):
        out[f"{prefix}_mean_{i:02d}"] = float(m)
        out[f"{prefix}_std_{i:02d}"] = float(s)
    return out


def _mean_only(feat: np.ndarray, prefix: str) -> dict[str, float]:
    feat = np.atleast_2d(feat)
    return {f"{prefix}_mean_{i:02d}": float(v) for i, v in enumerate(np.mean(feat, axis=1))}


def extract_vector(y: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    feats: dict[str, float] = {}
    hop = 512

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop)
    feats.update(_mean_std(mfcc, "mfcc"))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    feats.update(_mean_only(chroma, "chroma"))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    feats.update(_mean_only(contrast, "contrast"))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)
    feats.update(_mean_std(centroid, "centroid"))
    feats.update(_mean_std(bandwidth, "bandwidth"))
    feats.update(_mean_std(rolloff, "rolloff"))
    feats.update(_mean_std(zcr, "zcr"))
    feats.update(_mean_std(rms, "rms"))

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    feats["onset_mean"] = float(np.mean(onset))
    feats["onset_std"] = float(np.std(onset))

    try:
        tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=hop)
        feats["tempo_bpm"] = float(np.atleast_1d(tempo)[0])
    except Exception:  # noqa: BLE001
        tempo = librosa.beat.tempo(onset_envelope=onset, sr=sr, hop_length=hop)
        feats["tempo_bpm"] = float(np.atleast_1d(tempo)[0])

    # Pulso: intervalos entre beats y beats por segundo (comparable entre 30 s, 60 s, ...).
    duracion_s = max(len(y) / sr, 1e-6)
    feats["beat_interval_mean"] = 0.0
    feats["beat_interval_std"] = 0.0
    feats["beat_rate"] = 0.0
    try:
        _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
        if len(beats) > 0:
            feats["beat_rate"] = float(len(beats)) / duracion_s
        if len(beats) > 1:
            intervals = np.diff(librosa.frames_to_time(beats, sr=sr, hop_length=hop))
            feats["beat_interval_mean"] = float(np.mean(intervals))
            feats["beat_interval_std"] = float(np.std(intervals))
    except Exception:  # noqa: BLE001
        pass

    return feats


def extract_from_file(path: Path) -> tuple[dict[str, float], float]:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return extract_vector(y, sr), len(y) / sr


def run_features(seconds: int | None = None, split: str = "train") -> dict[int, pd.DataFrame]:
    ensure_dirs()
    targets = [seconds] if seconds else list(clip_seconds())
    tables: dict[int, pd.DataFrame] = {}

    for sec in targets:
        root = clips_dir(sec, split)
        wavs = sorted(root.rglob("*.wav"))
        if not wavs:
            print(f"No hay clips en {root}. Corre primero --clips.")
            continue

        rows = []
        print(f"Extrayendo features de {len(wavs)} clip(s) de {sec}s ...")
        for wav in wavs:
            rel = wav.relative_to(root)
            genre = rel.parts[0] if len(rel.parts) > 1 else "sin_genero"
            try:
                vec, dur = extract_from_file(wav)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {rel}: {exc}")
                continue
            row = {
                "archivo": wav.stem,
                "genero": genre,
                "duracion_clip": sec,
                "duracion_real": round(dur, 2),
                "ruta": str(rel).replace("\\", "/"),
            }
            row.update(vec)
            rows.append(row)
            print(f"  [ok] {rel}")

        if not rows:
            continue
        df = pd.DataFrame(rows)
        meta = ["archivo", "genero", "duracion_clip", "duracion_real", "ruta"]
        feat_cols = [c for c in df.columns if c not in meta]
        df = df[meta + sorted(feat_cols)]
        out = features_csv(sec, split)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Guardado {out} ({len(df)} filas, {len(feat_cols)} features)")
        tables[sec] = df

    return tables


if __name__ == "__main__":
    run_features()
