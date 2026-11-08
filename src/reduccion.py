"""Laboratorio de reducción de dimensionalidad × algoritmos de agrupación.

Compara PCA, PPCA, SVD iterativa, t-SNE y UMAP combinados con los 6 algoritmos del proyecto.
Las métricas internas se recalculan en un espacio común (el de PCA) para que sean comparables
y las externas (ARI, NMI, V-measure) comparan los grupos con los géneros reales.

Uso:  python src/main.py --reduccion        (o: python src/reduccion.py)
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cluster_compare as cc  # noqa: E402
from config import RANDOM_STATE, clip_seconds, results_dir  # noqa: E402

REDUCCIONES = ["PCA", "PPCA", "SVD iterativa", "t-SNE", "UMAP"]
ESTOCASTICAS = {"t-SNE", "UMAP"}
ALGORITMOS = ["KMeans", "Jerarquico", "GMM", "DBSCAN", "OPTICS", "HDBSCAN"]
DENSIDAD = {"DBSCAN", "OPTICS", "HDBSCAN"}
MAX_RUIDO_PARA_ELEGIR = 40.0  # una combinación con más % de ruido no se considera para "la mejor"


# ------------------------------------------------------------------ técnicas implementadas a mano
class PPCA:
    """PCA probabilístico (Tipping y Bishop, 1999): PCA con un modelo de ruido isotrópico.

    La varianza del ruido es el promedio de los valores propios descartados y la proyección es la
    media a posteriori, que "encoge" cada componente según su relación señal/ruido."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PPCA":
        self.mean_ = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        ev = S**2 / max(len(X) - 1, 1)
        q = min(self.n_components, len(ev))
        self.sigma2_ = float(ev[q:].mean()) if len(ev) > q else 1e-9
        self.components_ = Vt[:q]
        self.explained_variance_ = ev[:q]
        W = Vt[:q].T * np.sqrt(np.maximum(ev[:q] - self.sigma2_, 1e-12))
        self.W_ = W
        self.Minv_ = np.linalg.inv(W.T @ W + self.sigma2_ * np.eye(q))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.W_ @ self.Minv_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SVDIterativa:
    """Componentes principales por iteración de potencias con deflación (sin SVD completa)."""

    def __init__(self, n_components: int, tol: float = 1e-10, max_iter: int = 2000, seed: int = RANDOM_STATE):
        self.n_components, self.tol, self.max_iter, self.seed = n_components, tol, max_iter, seed

    def fit(self, X: np.ndarray) -> "SVDIterativa":
        self.mean_ = X.mean(axis=0)
        R = X - self.mean_
        rng = np.random.default_rng(self.seed)
        comps, self.n_iter_, self.explained_variance_ = [], [], []
        for _ in range(min(self.n_components, min(X.shape))):
            v = rng.normal(size=X.shape[1])
            v /= np.linalg.norm(v)
            it = 0
            for it in range(1, self.max_iter + 1):
                w = R.T @ (R @ v)
                nw = np.linalg.norm(w)
                if nw < 1e-14:
                    break
                w /= nw
                converge = 1 - abs(float(w @ v)) < self.tol
                v = w
                if converge:
                    break
            s = R @ v
            comps.append(v)
            self.n_iter_.append(it)
            self.explained_variance_.append(float(s @ s / max(len(X) - 1, 1)))
            R = R - np.outer(s, v)
        self.components_ = np.array(comps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def similitud_subespacios(a: np.ndarray, b: np.ndarray) -> float:
    """Coseno promedio de los ángulos principales entre dos subespacios (1 = el mismo subespacio)."""
    qa, _ = np.linalg.qr(a.T)
    qb, _ = np.linalg.qr(b.T)
    return float(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1).mean())


# ------------------------------------------------------------------ reducción y agrupación
def reducir(nombre: str, Xs: np.ndarray, n_keep: int, n_den: int, seed: int) -> dict:
    """Devuelve el espacio para agrupar (Z), el de densidad (Zd) y las coordenadas 2D (Z2)."""
    n = len(Xs)
    q = max(n_keep, n_den, 2)
    if nombre in ("PCA", "PPCA", "SVD iterativa"):
        modelo = {"PCA": lambda: PCA(n_components=q, random_state=seed),
                  "PPCA": lambda: PPCA(q), "SVD iterativa": lambda: SVDIterativa(q)}[nombre]()
        P = modelo.fit_transform(Xs)
        return {"Z": P[:, :n_keep], "Zd": P[:, :n_den], "Z2": P[:, :2], "modelo": modelo}
    if nombre == "t-SNE":
        perp = float(min(30, max(5, (n - 1) / 3)))
        Z2 = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(Xs)
    elif nombre == "UMAP":
        import umap  # requiere umap-learn

        Z2 = umap.UMAP(n_components=2, n_neighbors=int(min(15, n - 1)), min_dist=0.1, random_state=seed).fit_transform(Xs)
    else:
        raise ValueError(nombre)
    return {"Z": Z2, "Zd": Z2, "Z2": Z2, "modelo": None}


def agrupar_todos(Z: np.ndarray, Zd: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    """Corre los 6 algoritmos con su calibración de parámetros y devuelve etiquetas y parámetros."""
    out: dict[str, tuple[np.ndarray, str]] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        km, _ = cc.run_kmeans_sweep(Z)
        out["KMeans"] = (km["labels"], f"k={km['k']}")
        hier = cc.run_hierarchical(Z, km["k"])
        out["Jerarquico"] = (hier["labels"], f"k={hier['k']}")
        gmm = cc.run_gmm(Z)
        out["GMM"] = (gmm["labels"], f"k={gmm['k']} {gmm['cov']}")
        db = cc.run_dbscan(Zd)
        out["DBSCAN"] = (db["labels"], f"eps={db['eps']:.2f} min={db['min_samples']}")
        op = cc.run_optics(Zd)
        out["OPTICS"] = (op["labels"], f"min={op['min_samples']} xi={op['xi']}")
        hd = cc.run_hdbscan(Zd)
        if hd is not None:
            out["HDBSCAN"] = (hd["labels"], f"mcs={hd['min_cluster_size']} min={hd['min_samples']}")
    return out


def elegir_mejor(resumen: pd.DataFrame) -> dict | None:
    """Mejor combinación: mayor ARI promedio entre las que dejan poco ruido (desempate por V-measure)."""
    ok = resumen[resumen["pct_ruido_mean"] <= MAX_RUIDO_PARA_ELEGIR]
    ok = (ok if len(ok) else resumen).sort_values(["ari_mean", "v_measure_mean"], ascending=False)
    if ok.empty:
        return None
    r = ok.iloc[0]
    return {"reduccion": r["reduccion"], "algoritmo": r["algoritmo"], "ari": float(r["ari_mean"]),
            "v_measure": float(r["v_measure_mean"]), "pct_ruido": float(r["pct_ruido_mean"])}


def laboratorio(seconds: int, n_semillas: int = 3, reducciones: list[str] | None = None) -> dict | None:
    out = results_dir(seconds)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== Laboratorio de reducción {seconds}s =====")
    df = cc.load_features(seconds)
    df, X_raw, genres = cc.split_xy(df)
    Xs, X_comun, _, _, pca_full, _, info = cc.preprocess(X_raw)
    n_keep, n_den = info["n_keep"], info["n_den"]

    # --- información de las técnicas lineales
    var = pd.DataFrame({"componente": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
                        "varianza": pca_full.explained_variance_ratio_})
    var["acumulada"] = var["varianza"].cumsum()
    var.to_csv(out / "red_varianza.csv", index=False)
    q = max(n_keep, n_den, 2)
    pca, ppca, svd = PCA(n_components=q, random_state=RANDOM_STATE).fit(Xs), PPCA(q).fit(Xs), SVDIterativa(q).fit(Xs)
    modelos = {"PCA": pca.components_[:n_keep], "PPCA": ppca.components_[:n_keep], "SVD iterativa": svd.components_[:n_keep]}
    sub = pd.DataFrame([[similitud_subespacios(a, b) for b in modelos.values()] for a in modelos.values()],
                       index=list(modelos), columns=list(modelos))
    sub.to_csv(out / "red_subespacios.csv")
    pd.DataFrame({"componente": np.arange(1, len(svd.n_iter_) + 1), "iteraciones": svd.n_iter_,
                  "varianza_pca": pca.explained_variance_[: len(svd.n_iter_)],
                  "varianza_svd": svd.explained_variance_}).to_csv(out / "red_svd_iteraciones.csv", index=False)

    # --- reducción × algoritmo
    filas, proy = [], df[["ruta", "archivo", "genero"]].copy()
    for red in reducciones or REDUCCIONES:
        semillas = [RANDOM_STATE + i for i in range(n_semillas)] if red in ESTOCASTICAS else [RANDOM_STATE]
        for sd in semillas:
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = reducir(red, Xs, n_keep, n_den, sd)
            except Exception as exc:  # noqa: BLE001
                print(f"  [omitida] {red}: {exc}")
                break
            if sd == semillas[0]:
                proy[f"{red}_x"], proy[f"{red}_y"] = r["Z2"][:, 0], r["Z2"][:, 1]
            for alg, (labels, params) in agrupar_todos(r["Z"], r["Zd"]).items():
                m = cc.cluster_metrics(X_comun, labels, genres)
                propia = cc.cluster_metrics(r["Zd"] if alg in DENSIDAD else r["Z"], labels, genres)["silhouette"]
                filas.append({"reduccion": red, "algoritmo": alg, "semilla": sd, "params": params,
                              **{k: m[k] for k in ("n_clusters", "pct_ruido", "ari", "nmi", "v_measure", "homogeneidad", "completitud")},
                              "silueta_comun": m["silhouette"], "db_comun": m["davies_bouldin"], "silueta_propia": propia})
            print(f"  {red} (semilla {sd}) listo en {time.time() - t0:.1f}s")
    if not filas:
        return None
    largo = pd.DataFrame(filas)
    largo.to_csv(out / "red_resultados.csv", index=False)
    proy.to_csv(out / "red_proyecciones.csv", index=False)
    cols = ["n_clusters", "pct_ruido", "ari", "nmi", "v_measure", "homogeneidad", "completitud", "silueta_comun", "silueta_propia"]
    g = largo.groupby(["reduccion", "algoritmo"])[cols]
    resumen = g.mean().add_suffix("_mean").join(g.std(ddof=0).fillna(0).add_suffix("_std")).reset_index()
    resumen.to_csv(out / "red_resumen.csv", index=False)
    mejor = elegir_mejor(resumen)
    (out / "red_info.json").write_text(json.dumps({
        "n_canciones": len(df), "n_componentes_cluster": n_keep, "n_componentes_densidad": n_den,
        "ppca_sigma2": ppca.sigma2_, "n_semillas": n_semillas, "mejor": mejor,
        "max_ruido": MAX_RUIDO_PARA_ELEGIR}, ensure_ascii=False, indent=2), encoding="utf-8")
    if mejor:
        print(f"\nMejor combinación: {mejor['reduccion']} + {mejor['algoritmo']} (ARI {mejor['ari']:.2f})")
    return mejor


def run_reduccion(seconds: int | None = None, n_semillas: int = 3) -> None:
    for sec in [seconds] if seconds else clip_seconds():
        try:
            laboratorio(sec, n_semillas)
        except FileNotFoundError as exc:
            print(exc)


if __name__ == "__main__":
    run_reduccion()
