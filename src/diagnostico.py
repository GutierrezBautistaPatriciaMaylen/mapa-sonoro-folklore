"""Diagnósticos de calidad del agrupamiento: ¿hay estructura real y en qué se apoya?

- Hopkins y comparación con el azar (rasgos mezclados): ¿los datos tienen tendencia a agruparse?
- Familias de rasgos: ¿separa mejor el timbre, el ritmo, la armonía o la energía?
- Estabilidad: ¿cambian los grupos si se quita el 20 % de las canciones?
- Efecto artista: ¿los grupos se parecen más al artista que al género?

Uso:  python src/main.py --diagnostico        (o: python src/diagnostico.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATA_DIR, RANDOM_STATE, clip_seconds, results_dir  # noqa: E402
from predict_new import cargar_features  # noqa: E402
from rasgos import FAMILIAS, columnas_de, columnas_rasgo  # noqa: E402


def _reducir(X: np.ndarray, var: float = 0.90) -> np.ndarray:
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    p = PCA(random_state=RANDOM_STATE).fit(Xs)
    k = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), var) + 1)
    return p.transform(Xs)[:, : max(2, min(k, Xs.shape[1]))]


def hopkins(X: np.ndarray, seed: int = RANDOM_STATE, repeticiones: int = 20) -> float:
    """Estadístico de Hopkins: ~0.5 = datos al azar, cerca de 1 = tendencia clara a formar grupos."""
    rng = np.random.default_rng(seed)
    n = len(X)
    m = max(5, int(0.1 * n))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    lo, hi = X.min(axis=0), X.max(axis=0)
    vals = []
    for _ in range(repeticiones):
        idx = rng.choice(n, m, replace=False)
        w = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
        u = nn.kneighbors(rng.uniform(lo, hi, size=(m, X.shape[1])), n_neighbors=1)[0][:, 0]
        vals.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(vals))


def _kmeans(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)


def comparar_con_azar(X_raw: np.ndarray, genres: np.ndarray, repeticiones: int = 100) -> dict:
    """Compara KMeans(k = nº de géneros) con lo que sale al mezclar cada rasgo por separado."""
    k = len(set(genres))
    obs = _kmeans(_reducir(X_raw), k)
    sil_o, ari_o = float(silhouette_score(_reducir(X_raw), obs)), float(adjusted_rand_score(genres, obs))
    rng = np.random.default_rng(RANDOM_STATE)
    sil_n, ari_n = [], []
    for _ in range(repeticiones):
        Xn = np.column_stack([rng.permutation(X_raw[:, j]) for j in range(X_raw.shape[1])])
        Z = _reducir(Xn)
        lab = _kmeans(Z, k)
        sil_n.append(float(silhouette_score(Z, lab)))
        ari_n.append(float(adjusted_rand_score(genres, lab)))
    p = lambda o, nulo: float((np.sum(np.array(nulo) >= o) + 1) / (len(nulo) + 1))  # noqa: E731
    return {"k": k, "silueta_obs": sil_o, "ari_obs": ari_o, "silueta_nulo": sil_n, "ari_nulo": ari_n,
            "p_silueta": p(sil_o, sil_n), "p_ari": p(ari_o, ari_n)}


def ablacion_familias(df: pd.DataFrame, genres: np.ndarray) -> pd.DataFrame:
    k = len(set(genres))
    todas = list(FAMILIAS)
    grupos = {"Solo timbre": columnas_de(df, ["Timbre"]), "Solo armonía": columnas_de(df, ["Armonía"]),
              "Solo ritmo": columnas_de(df, ["Ritmo"]), "Solo energía": columnas_de(df, ["Energía"]),
              "Timbre + armonía": columnas_de(df, ["Timbre", "Armonía"]),
              "Todo sin energía": columnas_de(df, ["Timbre", "Armonía", "Ritmo"]),
              "Todo sin volumen": columnas_de(df, todas, sin_volumen=True),
              "Todos los rasgos": columnas_de(df, todas)}
    filas = []
    for nombre, cols in grupos.items():
        if len(cols) < 2:
            continue
        Z = _reducir(df[cols].to_numpy(float))
        for alg, lab in (("KMeans", _kmeans(Z, k)), ("Jerarquico", AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z))):
            h, c, v = homogeneity_completeness_v_measure(genres, lab)
            filas.append({"conjunto": nombre, "n_rasgos": len(cols), "algoritmo": alg, "ari": adjusted_rand_score(genres, lab),
                          "nmi": normalized_mutual_info_score(genres, lab), "v_measure": v, "silueta": silhouette_score(Z, lab)})
    return pd.DataFrame(filas)


def estabilidad(X_raw: np.ndarray, k: int, repeticiones: int = 30, fraccion: float = 0.8) -> dict:
    """ARI entre agrupaciones hechas con submuestras distintas (sobre las canciones que comparten)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X_raw)
    sols = []
    for _ in range(repeticiones):
        idx = np.sort(rng.choice(n, int(fraccion * n), replace=False))
        sols.append((idx, _kmeans(_reducir(X_raw[idx]), k)))
    aris = []
    for i in range(0, len(sols) - 1, 2):
        (ia, la), (ib, lb) = sols[i], sols[i + 1]
        comun, pa, pb = np.intersect1d(ia, ib, return_indices=True)
        aris.append(adjusted_rand_score(la[pa], lb[pb]))
    return {"k": k, "media": float(np.mean(aris)), "desv": float(np.std(aris)), "valores": [float(a) for a in aris]}


def efecto_artista(asig: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ARI de cada modelo contra el género y contra el artista; y artistas por género."""
    m = asig.merge(meta[["ruta", "artista"]], on="ruta", how="left")
    m["artista"] = m["artista"].fillna("").str.strip()
    conocido = m[m["artista"] != ""]
    por_genero = conocido.groupby("genero").agg(canciones=("ruta", "count"), artistas=("artista", "nunique")).reset_index()
    por_genero["canciones_por_artista"] = (por_genero["canciones"] / por_genero["artistas"]).round(2)
    filas = []
    for col in [c for c in ("KMeans", "Jerarquico", "GMM", "DBSCAN", "OPTICS", "HDBSCAN") if c in m.columns]:
        filas.append({"modelo": col, "ari_genero": adjusted_rand_score(conocido["genero"], conocido[col]),
                      "ari_artista": adjusted_rand_score(conocido["artista"], conocido[col]), "canciones": len(conocido)})
    return pd.DataFrame(filas), por_genero


def diagnosticar(seconds: int, repeticiones: int = 100) -> None:
    out = results_dir(seconds)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== Diagnósticos {seconds}s =====")
    df = cargar_features(seconds, "train")
    cols = columnas_rasgo(df)
    X_raw = np.nan_to_num(df[cols].to_numpy(float))
    genres = df["genero"].astype(str).to_numpy()
    k = len(set(genres))

    azar = comparar_con_azar(X_raw, genres, repeticiones)
    est = estabilidad(X_raw, k)
    abl = ablacion_familias(df, genres)
    abl.to_csv(out / "diag_familias.csv", index=False)
    resumen = {"n_canciones": len(df), "n_rasgos": len(cols), "k": k, "hopkins": hopkins(_reducir(X_raw)),
               "azar": azar, "estabilidad": est}

    meta_path = DATA_DIR / "canciones_metadata.csv"
    asig_path = out / "asignaciones.csv"
    if meta_path.exists() and asig_path.exists():
        meta = pd.read_csv(meta_path, dtype=str, encoding="utf-8-sig").fillna("")
        art, por_gen = efecto_artista(pd.read_csv(asig_path), meta)
        art.to_csv(out / "diag_artistas.csv", index=False)
        por_gen.to_csv(out / "diag_artistas_por_genero.csv", index=False)
    (out / "diag_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Hopkins={resumen['hopkins']:.2f} | p(silueta)={azar['p_silueta']:.3f} p(ARI)={azar['p_ari']:.3f} | estabilidad={est['media']:.2f}")


def run_diagnostico(seconds: int | None = None) -> None:
    for sec in [seconds] if seconds else clip_seconds():
        try:
            diagnosticar(sec)
        except FileNotFoundError as exc:
            print(exc)


if __name__ == "__main__":
    run_diagnostico()
