"""Guarda los modelos entrenados por versión y predice canciones nuevas.

Cada versión (duración de clip) tiene un paquete en models/<seg>s/bundle.joblib con:
scaler, PCA, PCA 2D, los 6 modelos y, para cada uno, el género dominante de cada cluster.

Uso por consola:
  python src/predict_new.py ruta/cancion.mp3 --duracion 30
  python src/predict_new.py ruta/cancion.mp3 --duracion 60 --genero Caporales
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import MIN_FULL_RATIO, SAMPLE_RATE, clip_seconds, models_dir

MODELOS = ["KMeans", "Jerarquico", "GMM", "DBSCAN", "OPTICS", "HDBSCAN"]
SIN_GRUPO = "ruido (sin grupo)"


def cargar_features(seconds: int, split: str = "train", verbose: bool = False) -> pd.DataFrame:
    """Lee features_<seg>s.csv con las mismas reglas que el entrenamiento (clips completos, beat_rate)."""
    from config import features_csv

    path = features_csv(seconds, split)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre --features primero.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Compatibilidad: beat_count dependía de la duración; ahora se usa beats por segundo.
    if "beat_count" in df.columns:
        if "beat_rate" not in df.columns:
            dur = df["duracion_real"] if "duracion_real" in df.columns else df["duracion_clip"]
            df["beat_rate"] = df["beat_count"] / dur.clip(lower=1)
        df = df.drop(columns=["beat_count"])
    # Cada versión solo usa clips completos (una canción corta no puede ser un "clip de 60 s").
    if "duracion_real" in df.columns:
        completos = df["duracion_real"] >= seconds * MIN_FULL_RATIO
        if verbose and (~completos).any():
            print(f"  {int((~completos).sum())} clip(s) más cortos que {seconds}s se excluyen de esta versión")
        df = df[completos].reset_index(drop=True)
    return df


def bundle_path(seconds: int) -> Path:
    return models_dir(seconds) / "bundle.joblib"


def _cluster_genre(labels: np.ndarray, genres: np.ndarray) -> dict[int, tuple[str, float]]:
    """Género dominante de cada cluster y su pureza (fracción de ese género)."""
    out: dict[int, tuple[str, float]] = {}
    for lab in sorted(set(labels.tolist())):
        if lab == -1:
            out[-1] = (SIN_GRUPO, 0.0)
            continue
        members = genres[labels == lab]
        genre, count = Counter(members.tolist()).most_common(1)[0]
        out[int(lab)] = (genre, count / len(members))
    return out


def guardar_bundle(
    seconds: int,
    df: pd.DataFrame,
    feat_cols: list[str],
    scaler,
    pca_full,
    info: dict,
    X: np.ndarray,
    X_den: np.ndarray,
    genres: np.ndarray,
    fitted: dict[str, dict],
) -> Path:
    from sklearn.neighbors import NearestNeighbors

    models: dict[str, dict] = {}
    for name, payload in fitted.items():
        labels = np.asarray(payload["labels"])
        entry: dict = {
            "labels": labels,
            "cluster_genre": _cluster_genre(labels, genres),
            "model": None,
            "centroids": None,
            "threshold": None,
        }
        if name in ("KMeans", "GMM"):
            entry.update(kind="predict", space="cluster", model=payload["model"])
        elif name == "Jerarquico":
            cents = {int(l): X[labels == l].mean(axis=0) for l in set(labels.tolist())}
            entry.update(kind="centroide", space="cluster", centroids=cents)
        else:
            valid = labels != -1
            base = X_den[valid] if valid.sum() > 2 else X_den
            nn = NearestNeighbors(n_neighbors=2).fit(base)
            dist = nn.kneighbors(base)[0][:, 1]
            # Un punto nuevo más lejos que el 95 % de las distancias vecinas se considera ruido.
            entry.update(kind="vecino", space="den", threshold=float(np.percentile(dist, 95)))
        models[name] = entry

    bundle = {
        "seconds": seconds,
        "sklearn": sklearn.__version__,
        "feature_cols": list(feat_cols),
        "scaler": scaler,
        "pca": pca_full,
        "pca2": info["pca2"],
        "n_keep": int(info["n_keep"]),
        "n_den": int(info["n_den"]),
        "X_cluster": X,
        "X_den": X_den,
        "genres": np.asarray(genres),
        "archivos": df["archivo"].tolist(),
        "rutas": df["ruta"].tolist(),
        "models": models,
    }
    path = bundle_path(seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def cargar_bundle(seconds: int) -> dict:
    path = bundle_path(seconds)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre python src/main.py --cluster primero.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def transformar(bundle: dict, feats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features crudas -> (espacio de cluster, espacio de densidad, coordenadas 2D)."""
    row = np.array([[float(feats.get(c, 0.0)) for c in bundle["feature_cols"]]], dtype=float)
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    xs = bundle["scaler"].transform(row)
    proj = bundle["pca"].transform(xs)
    return proj[:, : bundle["n_keep"]], proj[:, : bundle["n_den"]], bundle["pca2"].transform(xs)[0]


def asignar(bundle: dict, name: str, xc: np.ndarray, xd: np.ndarray) -> tuple[int, str, float]:
    m = bundle["models"][name]
    if m["kind"] == "predict":
        lab = int(m["model"].predict(xc)[0])
    elif m["kind"] == "centroide":
        labs = list(m["centroids"])
        dists = [np.linalg.norm(xc[0] - m["centroids"][l]) for l in labs]
        lab = int(labs[int(np.argmin(dists))])
    else:
        train = bundle["X_den"]
        d = np.linalg.norm(train - xd[0], axis=1)
        order = np.argsort(d)
        lab = int(m["labels"][order[0]])
        # Si el vecino más cercano es ruido o está muy lejos, la canción queda sin grupo.
        if d[order[0]] > m["threshold"]:
            lab = -1
    genre, purity = m["cluster_genre"].get(lab, (SIN_GRUPO, 0.0))
    return lab, genre, float(purity)


def vecinos(bundle: dict, xc: np.ndarray, k: int = 5) -> pd.DataFrame:
    d = np.linalg.norm(bundle["X_cluster"] - xc[0], axis=1)
    idx = np.argsort(d)[:k]
    return pd.DataFrame(
        {
            "archivo": [bundle["archivos"][i] for i in idx],
            "genero": [bundle["genres"][i] for i in idx],
            "ruta": [bundle["rutas"][i] for i in idx],
            "distancia": d[idx],
        }
    )


def predecir(bundle: dict, feats: dict) -> dict:
    """Predice con los 6 modelos de una versión. Devuelve tabla, 2D y vecinos."""
    xc, xd, xy = transformar(bundle, feats)
    rows = []
    for name in MODELOS:
        if name not in bundle["models"]:
            continue
        lab, genre, purity = asignar(bundle, name, xc, xd)
        rows.append({"modelo": name, "cluster": lab, "genero_predicho": genre, "pureza": purity})
    return {"tabla": pd.DataFrame(rows), "xy": xy, "vecinos": vecinos(bundle, xc)}


def predecir_tabla(bundle: dict, feats_df: pd.DataFrame) -> pd.DataFrame:
    """Predice cada fila de un CSV de features (canciones de prueba)."""
    out = []
    for _, row in feats_df.iterrows():
        res = predecir(bundle, row.to_dict())
        rec = {"archivo": row["archivo"], "ruta": row["ruta"], "genero_real": row["genero"]}
        for _, r in res["tabla"].iterrows():
            rec[r["modelo"]] = r["genero_predicho"]
            rec[f"{r['modelo']}_cluster"] = r["cluster"]
        out.append(rec)
    return pd.DataFrame(out)


def aciertos(tabla: pd.DataFrame) -> pd.DataFrame:
    """% de canciones de prueba cuyo género real coincide con el género dominante del cluster."""
    rows = []
    for name in MODELOS:
        if name in tabla.columns:
            ok = (tabla[name] == tabla["genero_real"]).mean() * 100
            sin = (tabla[name] == SIN_GRUPO).mean() * 100
            rows.append({"modelo": name, "aciertos_pct": float(ok), "sin_grupo_pct": float(sin)})
    return pd.DataFrame(rows)


def features_de_audio(path: Path, seconds: int) -> tuple[dict, float]:
    """Corta el centro del audio (igual que en el entrenamiento) y extrae sus features."""
    import librosa

    from clip_audio import centered_slice
    from extract_features import extract_vector

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    dur = len(y) / sr
    if dur < seconds * MIN_FULL_RATIO:
        raise ValueError(
            f"La canción dura {dur:.0f} s y esta versión necesita al menos {seconds * MIN_FULL_RATIO:.0f} s."
        )
    clip = centered_slice(y, sr, seconds)
    feats = extract_vector(clip, sr)
    return feats, len(clip) / sr


def main() -> None:
    p = argparse.ArgumentParser(description="Predice la agrupación de una canción nueva")
    p.add_argument("audio", type=Path)
    p.add_argument("--duracion", type=int, choices=list(clip_seconds()), default=clip_seconds()[0])
    p.add_argument("--genero", default=None, help="Género real (opcional) para ver si acierta")
    args = p.parse_args()

    bundle = cargar_bundle(args.duracion)
    feats, _ = features_de_audio(args.audio, args.duracion)
    res = predecir(bundle, feats)
    tabla = res["tabla"]
    if args.genero:
        tabla["acierta"] = tabla["genero_predicho"] == args.genero
    print(tabla.to_string(index=False))
    print("\nCanciones más parecidas:")
    print(res["vecinos"].to_string(index=False))


if __name__ == "__main__":
    main()
