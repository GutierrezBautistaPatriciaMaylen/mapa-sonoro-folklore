"""Compara algoritmos de clustering y genera métricas + gráficos."""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import OPTICS, AgglomerativeClustering, DBSCAN, KMeans
try:  # HDBSCAN viene en scikit-learn >= 1.3 (no hace falta la librería externa hdbscan)
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:  # pragma: no cover
    SklearnHDBSCAN = None
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    RANDOM_STATE,
    clip_seconds,
    ensure_dirs,
    features_csv,
    results_dir,
    version_name,
)
from mlops import hash_archivo, registrar_corrida  # noqa: E402
from predict_new import cargar_features, guardar_bundle  # noqa: E402

META_COLS = ["archivo", "genero", "duracion_clip", "duracion_real", "ruta"]


def _minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _knee_eps(kdist: np.ndarray) -> float:
    """Punto de máxima distancia a la cuerda primer-último (codo)."""
    y = np.asarray(kdist, dtype=float)
    n = len(y)
    if n < 3:
        return float(np.median(y)) if n else 0.5
    x = np.arange(n, dtype=float)
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    denom = np.hypot(x2 - x1, y2 - y1) or 1.0
    dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    return float(y[int(np.argmax(dist))])


def load_features(seconds: int) -> pd.DataFrame:
    return cargar_features(seconds, "train", verbose=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    genres = df["genero"].astype(str).to_numpy()
    return df, X, genres


def preprocess(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA, StandardScaler, dict]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n, p = Xs.shape
    max_comp = max(2, min(n - 1, p, 20))
    pca_full = PCA(n_components=max_comp, random_state=RANDOM_STATE)
    Xp = pca_full.fit_transform(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cum, 0.90) + 1)
    n_keep = min(max(n_keep, 2), max_comp)
    X_cluster = Xp[:, :n_keep]
    # Densidad se degrada en 20-D: recortar a ~70 % de varianza (máx. 8 comps).
    n_den = int(np.searchsorted(cum, 0.70) + 1)
    n_den = min(max(n_den, 2), 8, n_keep)
    X_density = Xp[:, :n_den]
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs)
    X_2d = pca2.transform(Xs)
    print(
        f"  PCA: {p} feat -> {n_keep} comps ({cum[n_keep - 1] * 100:.1f}% var) "
        f"para KMeans/GMM/jerárquico; {n_den} comps "
        f"({cum[n_den - 1] * 100:.1f}% var) para DBSCAN/OPTICS/HDBSCAN; n={n}"
    )
    info = {"pca2": pca2, "n_keep": n_keep, "n_den": n_den, "var_keep": float(cum[n_keep - 1]), "var_den": float(cum[n_den - 1])}
    return Xs, X_cluster, X_2d, X_density, pca_full, scaler, info


def _n_clusters(labels: np.ndarray) -> int:
    return int(len(set(labels) - {-1}))


def _noise_ratio(labels: np.ndarray) -> float:
    return float(np.mean(labels == -1)) if len(labels) else 0.0


def cluster_metrics(X: np.ndarray, labels: np.ndarray, genres: np.ndarray) -> dict:
    labels = np.asarray(labels)
    n_cl = _n_clusters(labels)
    noise = _noise_ratio(labels)
    mask = labels != -1
    sil = ch = db = np.nan
    if n_cl >= 2 and mask.sum() > n_cl:
        try:
            sil = float(silhouette_score(X[mask], labels[mask]))
        except Exception:  # noqa: BLE001
            sil = np.nan
        try:
            ch = float(calinski_harabasz_score(X[mask], labels[mask]))
        except Exception:  # noqa: BLE001
            ch = np.nan
        try:
            db = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception:  # noqa: BLE001
            db = np.nan

    n_genres = len(set(genres))
    ari = nmi = np.nan
    if n_genres >= 2 and n_cl >= 1:
        pred = labels.copy()
        ari = float(adjusted_rand_score(genres, pred))
        nmi = float(normalized_mutual_info_score(genres, pred))

    valido = n_cl >= 2 and noise <= 0.50 and np.isfinite(sil)
    return {
        "n_clusters": n_cl,
        "pct_ruido": round(noise * 100, 2),
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "ari": ari,
        "nmi": nmi,
        "valido": valido,
    }


def k_range(n: int) -> range:
    return range(2, min(10, n - 1) + 1)


def run_kmeans_sweep(X: np.ndarray) -> tuple[dict, pd.DataFrame]:
    rows = []
    for k in k_range(len(X)):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k < len(X) else np.nan
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(sil),
                "calinski": float(calinski_harabasz_score(X, labels)),
            }
        )
    sweep = pd.DataFrame(rows)
    if sweep.empty:
        dummy = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(X)
        return {"model": dummy, "labels": dummy.labels_, "k": 2, "inertia": dummy.inertia_}, sweep

    sil_n = _minmax(sweep["silhouette"].to_numpy())
    ch_n = _minmax(sweep["calinski"].to_numpy())
    # La inercia siempre baja al subir k, por eso no entra en el score (sigue en la curva del codo).
    sweep["score_combinado"] = (sil_n + ch_n) / 2.0
    best = sweep.loc[sweep["score_combinado"].idxmax()]
    k = int(best["k"])
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    return {"model": km, "labels": labels, "k": k, "inertia": float(km.inertia_)}, sweep


def run_gmm(X: np.ndarray) -> dict:
    n = len(X)
    cov = "diag" if n < X.shape[1] * 2 else "full"
    bic_rows = []
    best_gmm = None
    best_bic = np.inf
    for k in k_range(n):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov,
            random_state=RANDOM_STATE,
            n_init=3,
        )
        gmm.fit(X)
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
        bic_rows.append({"k": k, "bic": bic, "aic": aic})
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=2, covariance_type=cov, random_state=RANDOM_STATE)
        best_gmm.fit(X)
    return {
        "model": best_gmm,
        "labels": best_gmm.predict(X),
        "k": int(best_gmm.n_components),
        "bic": best_bic,
        "cov": cov,
        "sweep": pd.DataFrame(bic_rows),
    }


def run_hierarchical(X: np.ndarray, k: int) -> dict:
    k = max(2, min(k, len(X) - 1))
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="euclidean", linkage="ward")
    labels = model.fit_predict(X)
    Z = linkage(X, method="ward")
    return {"model": model, "labels": labels, "k": k, "Z": Z}


def _density_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Prefiere ≥2 clusters, poco ruido y silueta alta. Penaliza 1 blob o 100% ruido."""
    labels = np.asarray(labels)
    n_cl = _n_clusters(labels)
    noise = _noise_ratio(labels)
    if n_cl < 2:
        return -10.0 - noise
    if noise > 0.55:
        return -5.0 - noise
    mask = labels != -1
    if int(mask.sum()) <= n_cl:
        return -8.0
    try:
        sil = float(silhouette_score(X[mask], labels[mask]))
    except Exception:  # noqa: BLE001
        sil = -1.0
    return sil - 0.35 * noise - 0.02 * abs(n_cl - 5)


def run_dbscan(X: np.ndarray) -> dict:
    n = len(X)
    min_grid = [3, 4, 5]
    if n >= 40:
        min_grid.append(8)

    best: dict | None = None
    best_score = -np.inf
    fallback = None

    for min_samples in min_grid:
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        kdist = np.sort(dists[:, -1])
        eps_list = [_knee_eps(kdist)]
        for q in (0.20, 0.30, 0.40, 0.50, 0.60, 0.75):
            eps_list.append(float(np.quantile(kdist, q)))
        for eps in {round(max(float(e), 1e-3), 4) for e in eps_list}:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            payload = {
                "model": model,
                "labels": labels,
                "eps": eps,
                "min_samples": min_samples,
                "kdist": kdist,
            }
            if fallback is None:
                fallback = payload
            score = _density_score(X, labels)
            if score > best_score:
                best_score = score
                best = payload

    chosen = best if best is not None else fallback
    print(
        f"  DBSCAN calibrado: eps={chosen['eps']:.3f} min_samples={chosen['min_samples']} "
        f"clusters={_n_clusters(chosen['labels'])} ruido={_noise_ratio(chosen['labels'])*100:.1f}%"
    )
    return chosen


def run_optics(X: np.ndarray) -> dict:
    n = len(X)
    min_grid = [3, 5]
    size_grid = sorted({5, 8, max(5, n // 20), max(8, n // 12)})
    xi_grid = (0.03, 0.05, 0.08, 0.12, 0.18)

    best: dict | None = None
    best_score = -np.inf
    fallback = None

    for min_samples in min_grid:
        for min_cluster_size in size_grid:
            for xi in xi_grid:
                model = OPTICS(
                    min_samples=min_samples,
                    xi=xi,
                    min_cluster_size=min_cluster_size,
                )
                labels = model.fit_predict(X)
                payload = {
                    "model": model,
                    "labels": labels,
                    "min_samples": min_samples,
                    "min_cluster_size": min_cluster_size,
                    "xi": xi,
                    "reachability": np.asarray(model.reachability_),
                    "ordering": np.asarray(model.ordering_),
                }
                if fallback is None:
                    fallback = payload
                score = _density_score(X, labels)
                if score > best_score:
                    best_score = score
                    best = payload

    chosen = best if best is not None else fallback
    print(
        f"  OPTICS calibrado: min_samples={chosen['min_samples']} "
        f"min_cluster_size={chosen['min_cluster_size']} xi={chosen['xi']} "
        f"clusters={_n_clusters(chosen['labels'])} ruido={_noise_ratio(chosen['labels'])*100:.1f}%"
    )
    return chosen


def run_hdbscan(X: np.ndarray) -> dict | None:
    if SklearnHDBSCAN is None:
        print("  HDBSCAN no disponible (requiere scikit-learn >= 1.3). Se omite.")
        return None
    n = len(X)
    size_grid = sorted({5, 8, 10, max(5, n // 20), max(8, n // 12)})
    ms_grid = (1, 2, 3, 5)
    methods = ("eom", "leaf")

    best: dict | None = None
    best_score = -np.inf
    fallback = None

    for min_cluster_size in size_grid:
        for min_samples in ms_grid:
            if min_samples > min_cluster_size:
                continue
            for method in methods:
                model = SklearnHDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples),
                    cluster_selection_method=method,
                    metric="euclidean",
                )
                labels = model.fit_predict(X)
                payload = {
                    "model": model,
                    "labels": labels,
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": int(min_samples),
                    "method": method,
                }
                if fallback is None:
                    fallback = payload
                score = _density_score(X, labels)
                if score > best_score:
                    best_score = score
                    best = payload

    chosen = best if best is not None else fallback
    print(
        f"  HDBSCAN calibrado: min_cluster_size={chosen['min_cluster_size']} "
        f"min_samples={chosen['min_samples']} method={chosen['method']} "
        f"clusters={_n_clusters(chosen['labels'])} ruido={_noise_ratio(chosen['labels'])*100:.1f}%"
    )
    return chosen


def _save(fig: plt.Figure, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  figura: {path.name}")
    if show:
        plt.show()
    plt.close(fig)


def _slug(name: str) -> str:
    return str(name).lower().replace(" ", "_")


def plot_kmeans_curves(sweep: pd.DataFrame, out: Path, show: bool) -> None:
    if sweep.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    axes[0].plot(sweep["k"], sweep["inertia"], marker="o")
    axes[0].set_title("Codo (inercia)")
    axes[1].plot(sweep["k"], sweep["silhouette"], marker="o")
    axes[1].set_title("Silueta")
    axes[2].plot(sweep["k"], sweep["calinski"], marker="o")
    axes[2].set_title("Calinski-Harabasz")
    axes[3].plot(sweep["k"], sweep["score_combinado"], marker="o")
    axes[3].set_title("Score combinado")
    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)
    _save(fig, out / "kmeans_curvas.png", show)


def plot_gmm_curves(sweep: pd.DataFrame, out: Path, show: bool) -> None:
    if sweep is None or sweep.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["k"], sweep["bic"], marker="o", label="BIC")
    ax.plot(sweep["k"], sweep["aic"], marker="s", label="AIC")
    ax.set_title("GMM: BIC / AIC vs k")
    ax.set_xlabel("k (componentes)")
    ax.set_ylabel("criterio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out / "gmm_curvas.png", show)


def plot_ranking(ranked: pd.DataFrame, out: Path, show: bool) -> None:
    if ranked.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ranked["modelo"].tolist()

    def _bars(ax, series: pd.Series, title: str) -> None:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        colors = ["C0" if np.isfinite(v) else "0.85" for v in vals]
        plotted = np.where(np.isfinite(vals), vals, 0.0)
        ax.bar(names, plotted, color=colors)
        ax.set_title(title)

    _bars(axes[0], ranked["silhouette"], "Silueta (mayor mejor)")
    _bars(axes[1], ranked["ari"], "ARI vs género (mayor mejor)")
    _bars(axes[2], ranked["davies_bouldin"], "Davies-Bouldin (menor mejor)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Barras grises = métrica no definida (modelo no válido)", fontsize=9)
    _save(fig, out / "ranking_metricas.png", show)


def plot_pca(
    X_2d: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    title: str,
    out: Path,
    show: bool,
    filename: str = "pca_clusters.png",
) -> None:
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    noise = labels == -1
    if noise.any():
        ax.scatter(
            X_2d[noise, 0],
            X_2d[noise, 1],
            c="0.55",
            s=40,
            marker="x",
            label="ruido",
        )
    ok = ~noise
    scatter = None
    if ok.any():
        scatter = ax.scatter(
            X_2d[ok, 0],
            X_2d[ok, 1],
            c=labels[ok],
            cmap="tab10",
            s=50,
            edgecolors="k",
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if len(names) <= 25:
        for i, name in enumerate(names):
            ax.annotate(name[:18], (X_2d[i, 0], X_2d[i, 1]), fontsize=7, alpha=0.8)
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="cluster")
    if noise.any():
        ax.legend(loc="best")
    _save(fig, out / filename, show)


def plot_pca_genre(X_2d: np.ndarray, genres: np.ndarray, out: Path, show: bool) -> None:
    if len(set(genres)) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    codes, uniques = pd.factorize(genres)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=codes, cmap="tab10", s=60, edgecolors="k")
    ax.set_title("PCA 2D por género (carpeta)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, list(uniques), title="género")
    _save(fig, out / "pca_genero.png", show)


def plot_dendrogram(Z: np.ndarray, labels: list[str], out: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=[n[:16] for n in labels], leaf_rotation=90, ax=ax)
    ax.set_title("Dendrograma (Ward)")
    _save(fig, out / "dendrograma.png", show)


def plot_kdistance(kdist: np.ndarray, eps: float, out: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kdist, marker=".", linestyle="-")
    ax.axhline(eps, color="r", linestyle="--", label=f"eps={eps:.3f}")
    ax.set_title("k-distancia (DBSCAN)")
    ax.set_xlabel("puntos ordenados")
    ax.set_ylabel("distancia al k-ésimo vecino")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out / "kdistance.png", show)


def plot_optics(reach: np.ndarray, ordering: np.ndarray, out: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    space = np.arange(len(ordering))
    ax.bar(space, reach[ordering], width=1.0)
    ax.set_title("Reachability OPTICS")
    ax.set_xlabel("orden")
    ax.set_ylabel("reachability")
    _save(fig, out / "optics_reachability.png", show)


def plot_sizes(
    labels: np.ndarray,
    name: str,
    out: Path,
    show: bool,
    filename: str = "tamanos_cluster.png",
) -> None:
    ser = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["0.4" if i == -1 else "C0" for i in ser.index]
    ax.bar([str(i) for i in ser.index], ser.values, color=colors)
    ax.set_title(f"Tamaño de clusters ({name})")
    ax.set_xlabel("cluster (-1 = ruido)")
    ax.set_ylabel("canciones")
    _save(fig, out / filename, show)


def plot_heatmap(
    df: pd.DataFrame,
    labels: np.ndarray,
    out: Path,
    show: bool,
    filename: str = "heatmap_genero_cluster.png",
    title: str | None = None,
) -> None:
    if df["genero"].nunique() < 2:
        return
    tab = pd.crosstab(df["genero"], pd.Series(labels, name="cluster"))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(tab, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title or "Género × cluster")
    _save(fig, out / filename, show)


def plot_rhythm_features(df: pd.DataFrame, out: Path, show: bool) -> None:
    """Visualiza las features de ritmo (tempo, beats, intervalos) por género."""
    rhythm_cols = ["tempo_bpm", "beat_interval_mean", "beat_interval_std", "beat_count"]
    available = [c for c in rhythm_cols if c in df.columns]
    if not available:
        print("  No hay features de ritmo disponibles para graficar")
        return

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        genres = df["genero"].unique()
        for genre in genres:
            subset = df[df["genero"] == genre][col]
            ax.hist(subset, alpha=0.6, label=genre, bins=10)
        ax.set_xlabel(col)
        ax.set_ylabel("frecuencia")
        ax.set_title(f"Distribución de {col} por género")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out / "rhythm_features.png", show)


def plot_model_views(
    name: str,
    labels: np.ndarray,
    X_2d: np.ndarray,
    df: pd.DataFrame,
    seconds: int,
    out: Path,
    show: bool,
) -> None:
    slug = _slug(name)
    plot_pca(
        X_2d,
        labels,
        df["archivo"].tolist(),
        f"PCA 2D — {name} ({seconds}s)",
        out,
        show,
        filename=f"{slug}_pca.png",
    )
    plot_sizes(labels, name, out, show, filename=f"{slug}_tamanos.png")
    plot_heatmap(
        df,
        labels,
        out,
        show,
        filename=f"{slug}_heatmap.png",
        title=f"Género × cluster ({name})",
    )


def print_members(df: pd.DataFrame, labels: np.ndarray, model_name: str) -> None:
    print(f"\nCanciones por cluster ({model_name}):")
    tmp = df[["archivo", "genero"]].copy()
    tmp["cluster"] = labels
    for cl, grp in tmp.groupby("cluster"):
        tag = "ruido" if cl == -1 else f"cluster {cl}"
        print(f"  [{tag}] n={len(grp)}")
        for _, row in grp.iterrows():
            print(f"    - {row['genero']} / {row['archivo']}")


def rank_models(table: pd.DataFrame) -> pd.DataFrame:
    df = table.copy()
    has_ext = df["ari"].notna().any()
    if has_ext:
        df = df.sort_values(["valido", "ari", "silhouette"], ascending=[False, False, False])
    else:
        df["_db"] = -df["davies_bouldin"]
        df = df.sort_values(["valido", "silhouette", "_db"], ascending=[False, False, False])
        df = df.drop(columns=["_db"])
    return df.reset_index(drop=True)


def compare_one(seconds: int, show: bool = False) -> pd.DataFrame:
    ensure_dirs()
    out = results_dir(seconds)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n===== Experimento {seconds}s ({version_name(seconds)}) =====")
    df = load_features(seconds)
    df, X_raw, genres = split_xy(df)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    _, X, X_2d, X_den, pca_full, scaler, info = preprocess(X_raw)

    fitted: dict[str, dict] = {}
    rows = []

    km, sweep = run_kmeans_sweep(X)
    fitted["KMeans"] = km
    m = cluster_metrics(X, km["labels"], genres)
    m.update({"modelo": "KMeans", "params": f"k={km['k']}", "inertia": km["inertia"]})
    rows.append(m)
    plot_kmeans_curves(sweep, out, show)
    sweep.to_csv(out / "kmeans_sweep.csv", index=False)

    hier = run_hierarchical(X, km["k"])
    fitted["Jerarquico"] = hier
    m = cluster_metrics(X, hier["labels"], genres)
    m.update({"modelo": "Jerarquico", "params": f"ward k={hier['k']}", "inertia": np.nan})
    rows.append(m)
    plot_dendrogram(hier["Z"], df["archivo"].tolist(), out, show)

    gmm = run_gmm(X)
    fitted["GMM"] = gmm
    m = cluster_metrics(X, gmm["labels"], genres)
    m.update({"modelo": "GMM", "params": f"k={gmm['k']} cov={gmm['cov']}", "inertia": np.nan})
    rows.append(m)
    if not gmm["sweep"].empty:
        gmm["sweep"].to_csv(out / "gmm_bic.csv", index=False)
        plot_gmm_curves(gmm["sweep"], out, show)

    db = run_dbscan(X_den)
    fitted["DBSCAN"] = db
    m = cluster_metrics(X, db["labels"], genres)
    m.update(
        {"modelo": "DBSCAN", "params": f"eps={db['eps']:.3f} min={db['min_samples']}", "inertia": np.nan}
    )
    rows.append(m)
    plot_kdistance(db["kdist"], db["eps"], out, show)

    opt = run_optics(X_den)
    fitted["OPTICS"] = opt
    m = cluster_metrics(X, opt["labels"], genres)
    m.update(
        {
            "modelo": "OPTICS",
            "params": (
                f"min_samples={opt['min_samples']} "
                f"mcs={opt['min_cluster_size']} xi={opt['xi']}"
            ),
            "inertia": np.nan,
        }
    )
    rows.append(m)
    finite_reach = np.where(np.isfinite(opt["reachability"]), opt["reachability"], 0.0)
    plot_optics(finite_reach, opt["ordering"], out, show)

    hdb = run_hdbscan(X_den)
    if hdb is not None:
        fitted["HDBSCAN"] = hdb
        m = cluster_metrics(X, hdb["labels"], genres)
        m.update(
            {
                "modelo": "HDBSCAN",
                "params": (
                    f"min_cluster_size={hdb['min_cluster_size']} "
                    f"min_samples={hdb['min_samples']} {hdb['method']}"
                ),
                "inertia": np.nan,
            }
        )
        rows.append(m)

    table = pd.DataFrame(rows)
    cols = [
        "modelo",
        "params",
        "n_clusters",
        "pct_ruido",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "ari",
        "nmi",
        "inertia",
        "valido",
    ]
    table = table[cols]
    ranked = rank_models(table)
    ranked.to_csv(out / "ranking.csv", index=False)
    plot_ranking(ranked, out, show)

    # Graficar features de ritmo
    plot_rhythm_features(df, out, show)

    print("\nRanking de modelos:")
    with pd.option_context("display.max_columns", None, "display.width", 140, "display.precision", 3):
        print(ranked.to_string(index=False))

    best_name = ranked.iloc[0]["modelo"]
    best_labels = fitted[best_name]["labels"]
    print(f"\nMejor según ranking: {best_name} ({ranked.iloc[0]['params']})")
    print_members(df, best_labels, best_name)

    assignments = df[["archivo", "genero", "ruta"]].copy()
    assignments["pc1"] = X_2d[:, 0]
    assignments["pc2"] = X_2d[:, 1]
    for name, payload in fitted.items():
        assignments[name] = payload["labels"]
        plot_model_views(name, payload["labels"], X_2d, df, seconds, out, show)
    assignments.to_csv(out / "asignaciones.csv", index=False)

    plot_pca(X_2d, best_labels, df["archivo"].tolist(), f"PCA 2D — {best_name} ({seconds}s)", out, show)
    plot_pca_genre(X_2d, genres, out, show)
    plot_sizes(best_labels, best_name, out, show)
    plot_heatmap(df, best_labels, out, show)

    # Referencia: KMeans con k = número de géneros reales.
    k_gen = max(2, min(len(set(genres)), len(X) - 1))
    km_ref = KMeans(n_clusters=k_gen, n_init=10, random_state=RANDOM_STATE).fit(X)
    ref = cluster_metrics(X, km_ref.labels_, genres)
    (out / "referencia_k_generos.json").write_text(
        json.dumps({"k": k_gen, **{k: (None if pd.isna(v) else float(v)) for k, v in ref.items()
                                    if k in ("silhouette", "ari", "nmi", "davies_bouldin")}}, indent=2),
        encoding="utf-8",
    )

    # MLOps: guardar los modelos de esta versión y dejar constancia de la corrida.
    path = guardar_bundle(seconds, df, feat_cols, scaler, pca_full, info, X, X_den, genres, fitted)
    run_id = registrar_corrida(
        seconds, ranked, len(df), len(set(genres)), len(feat_cols), ref.get("ari", np.nan),
        hash_archivo(features_csv(seconds)), time.time() - t0,
    )
    print(f"\nModelos guardados en {path}\nCorrida registrada: {run_id}")

    return ranked


def run_cluster(seconds: int | None = None, show: bool = False) -> dict[int, pd.DataFrame]:
    warnings.filterwarnings("ignore")
    targets = [seconds] if seconds else list(clip_seconds())
    out = {}
    for sec in targets:
        try:
            out[sec] = compare_one(sec, show=show)
        except FileNotFoundError as exc:
            print(exc)
    if len(out) >= 2:
        print("\n===== Comparación entre versiones (mejor de cada una) =====")
        for sec, tab in out.items():
            top = tab.iloc[0]
            print(
                f"  {sec}s -> {top['modelo']} silueta={top['silhouette']!s} "
                f"ARI={top['ari']!s} valido={top['valido']}"
            )
    return out


if __name__ == "__main__":
    run_cluster()
