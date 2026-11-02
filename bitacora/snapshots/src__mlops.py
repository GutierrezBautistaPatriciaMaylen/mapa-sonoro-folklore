"""Registro de corridas (MLOps ligero): qué se entrenó, con qué datos, con qué resultado.

Cada vez que se entrena una versión se guarda una fila en models/registro_mlops.csv y una carpeta
models/<seg>s/runs/<run_id>/ con el resumen y el ranking de esa corrida. La corrida más reciente de
cada versión queda en etapa 'produccion' y las anteriores pasan a 'archivado'.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bitacora
from config import MODELS_DIR, models_dir, version_name

REGISTRO_CSV = MODELS_DIR / "registro_mlops.csv"
COLUMNAS = [
    "run_id", "fecha", "hora", "responsable", "version", "segundos", "n_canciones", "n_generos",
    "n_features", "mejor_modelo", "params", "silhouette", "calinski_harabasz", "davies_bouldin",
    "ari", "nmi", "pct_ruido", "ari_kmeans_k_generos", "hash_datos", "sklearn", "python",
    "duracion_seg", "motivo", "etapa", "metricas_modelos",
]
NUMERICAS = ["segundos", "n_canciones", "n_generos", "n_features", "silhouette", "calinski_harabasz",
             "davies_bouldin", "ari", "nmi", "pct_ruido", "ari_kmeans_k_generos", "duracion_seg"]


def hash_archivo(path: Path) -> str:
    """Huella corta del CSV de features: si cambian los datos, cambia la huella."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def leer_registro() -> pd.DataFrame:
    if not REGISTRO_CSV.exists():
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.read_csv(REGISTRO_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    for col in COLUMNAS:
        if col not in df.columns:
            df[col] = ""
    for col in NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[COLUMNAS]


def registrar_corrida(
    seconds: int,
    ranked: pd.DataFrame,
    n_canciones: int,
    n_generos: int,
    n_features: int,
    ari_k_generos: float,
    hash_datos: str,
    duracion_seg: float,
) -> str:
    """Guarda la corrida en el registro y en la bitácora. Devuelve el run_id."""
    import sklearn

    ahora = datetime.now()
    version = version_name(seconds)
    run_id = f"{version}-{ahora:%Y%m%d-%H%M%S}"
    top = ranked.iloc[0]
    metricas = {
        str(r["modelo"]): {
            k: _num(r[k]) for k in ("silhouette", "calinski_harabasz", "davies_bouldin", "ari", "nmi", "pct_ruido")
        } | {"valido": bool(r["valido"]), "params": str(r["params"])}
        for _, r in ranked.iterrows()
    }
    motivo = os.environ.get("MOTIVO_CORRIDA", "").strip() or "Corrida del pipeline"
    fila = {
        "run_id": run_id,
        "fecha": f"{ahora:%Y-%m-%d}",
        "hora": f"{ahora:%H:%M:%S}",
        "responsable": bitacora.responsable_actual(),
        "version": version,
        "segundos": seconds,
        "n_canciones": n_canciones,
        "n_generos": n_generos,
        "n_features": n_features,
        "mejor_modelo": str(top["modelo"]),
        "params": str(top["params"]),
        "silhouette": _num(top["silhouette"]),
        "calinski_harabasz": _num(top["calinski_harabasz"]),
        "davies_bouldin": _num(top["davies_bouldin"]),
        "ari": _num(top["ari"]),
        "nmi": _num(top["nmi"]),
        "pct_ruido": _num(top["pct_ruido"]),
        "ari_kmeans_k_generos": _num(ari_k_generos),
        "hash_datos": hash_datos,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "duracion_seg": round(duracion_seg, 1),
        "motivo": motivo,
        "etapa": "produccion",
        "metricas_modelos": json.dumps(metricas, ensure_ascii=False),
    }
    reg = leer_registro()
    reg.loc[(reg["version"] == version) & (reg["etapa"] == "produccion"), "etapa"] = "archivado"
    reg = pd.concat([reg, pd.DataFrame([fila])], ignore_index=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    reg[COLUMNAS].to_csv(REGISTRO_CSV, index=False, encoding="utf-8-sig")

    run_dir = models_dir(seconds) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(run_dir / "ranking.csv", index=False)
    (run_dir / "resumen.json").write_text(json.dumps(fila, ensure_ascii=False, indent=2), encoding="utf-8")

    bitacora.agregar(
        "corrida",
        f"{motivo}. Mejor: {fila['mejor_modelo']} ({fila['params']}); ARI={fila['ari']}, silueta={fila['silhouette']}",
        seccion="MLOps",
        version=version,
    )
    return run_id


def con_deltas(reg: pd.DataFrame) -> pd.DataFrame:
    """Agrega el cambio de cada métrica respecto a la corrida anterior de la misma versión."""
    if reg.empty:
        return reg.assign(d_ari=np.nan, d_silhouette=np.nan, d_davies_bouldin=np.nan, veredicto="")
    reg = reg.sort_values(["version", "fecha", "hora"]).reset_index(drop=True)
    for col in ("ari", "silhouette", "davies_bouldin"):
        reg[f"d_{col}"] = reg.groupby("version")[col].diff()
    primera = reg["d_ari"].isna()
    # Se considera mejora si sube el ARI o, con ARI igual, si sube la silueta.
    mejora = (reg["d_ari"] > 1e-4) | ((reg["d_ari"].abs() <= 1e-4) & (reg["d_silhouette"] > 1e-4))
    peor = (reg["d_ari"] < -1e-4) | ((reg["d_ari"].abs() <= 1e-4) & (reg["d_silhouette"] < -1e-4))
    reg["veredicto"] = np.select([primera, mejora, peor], ["primera corrida", "mejora", "empeora"], "sin cambio")
    return reg


def metricas_por_modelo(reg: pd.DataFrame) -> pd.DataFrame:
    """Expande metricas_modelos a formato largo: una fila por corrida y modelo."""
    filas = []
    for _, r in reg.iterrows():
        try:
            datos = json.loads(r["metricas_modelos"])
        except (TypeError, ValueError):
            continue
        for modelo, m in datos.items():
            filas.append({"run_id": r["run_id"], "version": r["version"], "fecha": r["fecha"], "hora": r["hora"],
                          "modelo": modelo, **{k: v for k, v in m.items() if k != "params"}})
    return pd.DataFrame(filas)
