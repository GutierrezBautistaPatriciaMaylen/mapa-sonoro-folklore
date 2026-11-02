"""Figuras interactivas (Plotly) y textos de interpretación para el dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import f_classif
from sklearn.metrics import adjusted_rand_score, silhouette_samples

# Paleta inspirada en los aguayos y textiles andinos (rojo, ocre, verde azulado, violeta...).
PALETA = ["#B3261E", "#D98A0B", "#1F6F6B", "#6B2D5C", "#4A6B2A", "#C2571A", "#2A4A8C", "#8C6A3F", "#C2417A", "#5B6770"]
CREMA = "#FAF6EE"
TINTA = "#2B2623"
RUIDO = "Ruido"
ESCALA_CALOR = [[0, "#FBF6EE"], [0.5, "#E9B44C"], [1, "#B3261E"]]

NOMBRES_RASGO = {
    "mfcc": "MFCC (timbre)", "chroma": "Cromagrama (notas)", "contrast": "Contraste espectral",
    "centroid": "Brillo (centroide)", "bandwidth": "Ancho espectral", "rolloff": "Rolloff",
    "zcr": "Cruces por cero", "rms": "Energía (RMS)", "onset": "Ataques",
    "tempo_bpm": "Tempo (BPM)", "beat_rate": "Pulsos por segundo", "beat_interval": "Intervalo entre pulsos",
}
RASGOS_RADAR = {
    "Tempo": "tempo_bpm", "Pulsos por seg.": "beat_rate", "Brillo": "centroid_mean_00",
    "Energía": "rms_mean_00", "Ruidosidad": "zcr_mean_00", "Ancho espectral": "bandwidth_mean_00",
    "Ataques": "onset_mean",
}


def _base(fig: go.Figure, titulo: str = "", alto: int = 440) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, x=0.0, font=dict(size=16, color=TINTA)),
        height=alto, margin=dict(l=10, r=10, t=50 if titulo else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TINTA), legend=dict(orientation="v"),
    )
    return fig


def nombre_rasgo(col: str) -> str:
    for pref, nombre in NOMBRES_RASGO.items():
        if col == pref or col.startswith(pref + "_"):
            partes = col[len(pref):].strip("_").split("_")
            extra = " ".join(p for p in partes if p)
            extra = extra.replace("mean", "media").replace("std", "desv.")
            return f"{nombre} {extra}".strip()
    return col


def etiquetas_cluster(serie: pd.Series) -> pd.Series:
    return serie.astype(int).map(lambda v: RUIDO if v == -1 else f"Cluster {v}")


def mapa_colores(categorias) -> dict[str, str]:
    cats = sorted({str(c) for c in categorias if str(c) != RUIDO})
    colores = {c: PALETA[i % len(PALETA)] for i, c in enumerate(cats)}
    colores[RUIDO] = "#9AA0A6"
    return colores


# ------------------------------------------------------------------ mapa y métricas
def fig_mapa(df: pd.DataFrame, color: str, titulo: str, estrellas: pd.DataFrame | None = None) -> go.Figure:
    d = df.copy()
    colores = mapa_colores(d[color])
    fig = px.scatter(d, x="pc1", y="pc2", color=color, color_discrete_map=colores, hover_name="archivo",
                     hover_data={"genero": True, "pc1": False, "pc2": False, color: True})
    fig.update_traces(marker=dict(size=10, line=dict(width=0.8, color="white")))
    if estrellas is not None and len(estrellas):
        fig.add_trace(go.Scatter(
            x=estrellas["pc1"], y=estrellas["pc2"], mode="markers", name="Canción de prueba",
            text=estrellas["archivo"], hovertemplate="%{text}<extra>prueba</extra>",
            marker=dict(symbol="star", size=17, color="#111", line=dict(width=1.5, color="white")),
        ))
    fig.update_xaxes(title="Componente 1", showgrid=False, zeroline=False)
    fig.update_yaxes(title="Componente 2", showgrid=False, zeroline=False)
    return _base(fig, titulo, 520)


def fig_barras_metricas(ranking: pd.DataFrame) -> go.Figure:
    specs = [("silhouette", "Silueta (más alto = mejor)"), ("ari", "ARI vs géneros (más alto = mejor)"),
             ("nmi", "NMI vs géneros (más alto = mejor)"), ("davies_bouldin", "Davies-Bouldin (más bajo = mejor)")]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[s[1] for s in specs], horizontal_spacing=0.06)
    d = ranking.copy()
    for i, (col, _) in enumerate(specs, start=1):
        fig.add_trace(go.Bar(x=d["modelo"], y=d[col], marker_color=[PALETA[j % len(PALETA)] for j in range(len(d))],
                             text=d[col].round(2), textposition="outside", showlegend=False), row=1, col=i)
        fig.update_xaxes(tickangle=-40, row=1, col=i)
    fig.update_annotations(font_size=12)
    return _base(fig, "", 380)


def fig_heatmap_genero_cluster(asig: pd.DataFrame, col: str) -> go.Figure:
    ct = pd.crosstab(asig["genero"], etiquetas_cluster(asig[col]))
    fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale=ESCALA_CALOR)
    fig.update_xaxes(title="Cluster encontrado", side="bottom")
    fig.update_yaxes(title="Género real")
    return _base(fig, "¿Qué géneros cayeron en cada cluster?", 420)


def fig_tamanos(asig: pd.DataFrame, col: str) -> go.Figure:
    cnt = etiquetas_cluster(asig[col]).value_counts().sort_index().reset_index()
    cnt.columns = ["cluster", "canciones"]
    fig = px.bar(cnt, x="cluster", y="canciones", color="cluster", color_discrete_map=mapa_colores(cnt["cluster"]), text="canciones")
    fig.update_layout(showlegend=False)
    return _base(fig, "Canciones por cluster", 340)


def pureza_por_cluster(asig: pd.DataFrame, col: str) -> pd.DataFrame:
    filas = []
    for lab, g in asig.groupby(col):
        if lab == -1:
            continue
        top = g["genero"].value_counts()
        filas.append({"cluster": f"Cluster {lab}", "genero_dominante": top.index[0], "canciones": len(g),
                      "pureza": top.iloc[0] / len(g)})
    return pd.DataFrame(filas)


def fig_pureza(asig: pd.DataFrame, col: str) -> go.Figure:
    p = pureza_por_cluster(asig, col)
    if p.empty:
        return _base(go.Figure(), "Sin clusters")
    fig = px.bar(p, x="cluster", y="pureza", color="genero_dominante", text="genero_dominante",
                 color_discrete_map=mapa_colores(p["genero_dominante"]))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", title="Pureza (% del género dominante)")
    fig.update_layout(showlegend=False)
    return _base(fig, "Pureza de cada cluster", 360)


# ------------------------------------------------------------------ gráficos de resultados adicionales
def fig_consenso(asig: pd.DataFrame, modelos: list[str]) -> go.Figure:
    modelos = [m for m in modelos if m in asig.columns]
    mat = np.array([[adjusted_rand_score(asig[a], asig[b]) for b in modelos] for a in modelos])
    fig = px.imshow(pd.DataFrame(mat, index=modelos, columns=modelos), text_auto=".2f", zmin=0, zmax=1,
                    color_continuous_scale=ESCALA_CALOR, aspect="auto")
    return _base(fig, "¿Qué tanto coinciden los algoritmos entre sí? (ARI)", 420)


def fig_silueta(X: np.ndarray, labels: np.ndarray) -> go.Figure:
    labels = np.asarray(labels)
    mask = labels != -1
    if len(set(labels[mask].tolist())) < 2:
        return _base(go.Figure(), "Silueta por canción: hacen falta al menos 2 clusters")
    vals = silhouette_samples(X[mask], labels[mask])
    d = pd.DataFrame({"cluster": labels[mask], "s": vals}).sort_values(["cluster", "s"]).reset_index(drop=True)
    d["etq"] = etiquetas_cluster(d["cluster"])
    fig = go.Figure()
    colores = mapa_colores(d["etq"])
    for etq, g in d.groupby("etq", sort=True):
        fig.add_trace(go.Bar(x=g["s"], y=g.index, orientation="h", name=etq, marker_color=colores[etq]))
    fig.add_vline(x=float(vals.mean()), line_dash="dash", line_color=TINTA, annotation_text=f"media {vals.mean():.2f}")
    fig.update_yaxes(showticklabels=False, title="Canciones (ordenadas por cluster)")
    fig.update_xaxes(title="Coeficiente de silueta")
    return _base(fig, "Silueta de cada canción (qué tan bien encaja en su cluster)", 480)


def fig_importancia(feats: pd.DataFrame, top: int = 15) -> go.Figure:
    cols = [c for c in feats.columns if c not in ("archivo", "genero", "duracion_clip", "duracion_real", "ruta")]
    X = feats[cols].to_numpy(float)
    X = np.nan_to_num(X)
    ok = X.std(axis=0) > 1e-9
    f, _ = f_classif(X[:, ok], feats["genero"].astype(str))
    d = pd.DataFrame({"rasgo": [nombre_rasgo(c) for c in np.array(cols)[ok]], "F": np.nan_to_num(f)})
    d = d.sort_values("F").tail(top)
    fig = px.bar(d, x="F", y="rasgo", orientation="h", color_discrete_sequence=[PALETA[0]])
    fig.update_xaxes(title="Poder para separar géneros (F de ANOVA)")
    fig.update_yaxes(title="")
    return _base(fig, "Qué rasgos del sonido separan mejor a los géneros", 480)


def fig_radar(feats: pd.DataFrame, generos: list[str]) -> go.Figure:
    rasgos = {k: v for k, v in RASGOS_RADAR.items() if v in feats.columns}
    medias = feats.groupby("genero")[list(rasgos.values())].mean()
    rng = (medias.max() - medias.min()).replace(0, 1)
    norm = (medias - medias.min()) / rng
    colores = mapa_colores(feats["genero"].unique())
    fig = go.Figure()
    for g in generos:
        if g not in norm.index:
            continue
        vals = norm.loc[g].tolist()
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=list(rasgos) + [list(rasgos)[0]], name=g, fill="toself",
                                      opacity=0.45, line=dict(color=colores.get(g, PALETA[0]))))
    fig.update_polars(radialaxis=dict(range=[0, 1], showticklabels=False), bgcolor="rgba(0,0,0,0)")
    return _base(fig, "Huella sonora por género (cada eje va de lo mínimo a lo máximo entre géneros)", 480)


def similitud_generos(feats: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in feats.columns if c not in ("archivo", "genero", "duracion_clip", "duracion_real", "ruta")]
    X = np.nan_to_num(feats[cols].to_numpy(float))
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd < 1e-9, 1, sd)
    medias = pd.DataFrame(Z, columns=cols).groupby(feats["genero"].astype(str).to_numpy()).mean()
    M = medias.to_numpy()
    M = M / np.linalg.norm(M, axis=1, keepdims=True).clip(1e-9)
    return pd.DataFrame(M @ M.T, index=medias.index, columns=medias.index)


def fig_similitud(feats: pd.DataFrame) -> go.Figure:
    sim = similitud_generos(feats)
    fig = px.imshow(sim, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    return _base(fig, "¿Qué géneros suenan parecido? (similitud coseno)", 460)


def pares_parecidos(sim: pd.DataFrame, n: int = 3) -> list[tuple[str, str, float]]:
    pares = [(a, b, float(sim.loc[a, b])) for i, a in enumerate(sim.index) for b in sim.columns[i + 1:]]
    return sorted(pares, key=lambda t: -t[2])[:n]


def fig_distancias(X: np.ndarray, genres: np.ndarray) -> go.Figure:
    orden = np.argsort(genres, kind="stable")
    Xo, go_ = X[orden], np.asarray(genres)[orden]
    D = np.linalg.norm(Xo[:, None, :] - Xo[None, :, :], axis=2)
    cats = list(dict.fromkeys(go_.tolist()))
    mids = [float(np.mean(np.where(go_ == c)[0])) for c in cats]
    fig = go.Figure(go.Heatmap(z=D, colorscale="YlOrRd_r", showscale=True, colorbar=dict(title="Distancia")))
    fig.update_xaxes(tickmode="array", tickvals=mids, ticktext=cats, tickangle=-40)
    fig.update_yaxes(tickmode="array", tickvals=mids, ticktext=cats, autorange="reversed")
    return _base(fig, "Distancia entre canciones (ordenadas por género): los bloques oscuros son géneros compactos", 520)


def fig_sankey(a: pd.DataFrame, b: pd.DataFrame, col_a: str, col_b: str, nom_a: str, nom_b: str) -> go.Figure | None:
    m = a[["ruta", col_a]].merge(b[["ruta", col_b]], on="ruta", suffixes=("_a", "_b"))
    if m.empty:
        return None
    ea = nom_a + " · " + etiquetas_cluster(m[col_a + "_a" if col_a == col_b else col_a])
    eb = nom_b + " · " + etiquetas_cluster(m[col_b + "_b" if col_a == col_b else col_b])
    nodos = sorted(ea.unique()) + sorted(eb.unique())
    idx = {n: i for i, n in enumerate(nodos)}
    flujo = pd.DataFrame({"a": ea, "b": eb}).value_counts().reset_index(name="n")
    fig = go.Figure(go.Sankey(
        node=dict(label=nodos, pad=14, thickness=16, color=[PALETA[i % len(PALETA)] for i in range(len(nodos))]),
        link=dict(source=flujo["a"].map(idx), target=flujo["b"].map(idx), value=flujo["n"], color="rgba(120,110,100,0.30)"),
    ))
    return _base(fig, f"Cómo se mueven las canciones de {nom_a} a {nom_b}", 460)


# ------------------------------------------------------------------ MLOps
def fig_evolucion(reg: pd.DataFrame, metrica: str, titulo: str) -> go.Figure:
    d = reg.copy()
    d["corrida"] = d.groupby("version").cumcount() + 1
    fig = px.line(d, x="corrida", y=metrica, color="version", markers=True, color_discrete_sequence=PALETA,
                  hover_data=["run_id", "mejor_modelo", "motivo"])
    fig.update_xaxes(dtick=1, title="Número de corrida")
    return _base(fig, titulo, 360)


def fig_deltas(reg: pd.DataFrame) -> go.Figure | None:
    d = reg[reg["veredicto"] != "primera corrida"].copy()
    if d.empty:
        return None
    fig = px.bar(d, x="run_id", y="d_ari", color="veredicto", hover_data=["version", "motivo", "d_silhouette"],
                 color_discrete_map={"mejora": "#2E7D32", "empeora": "#B3261E", "sin cambio": "#9AA0A6"})
    fig.update_yaxes(title="Cambio del ARI vs corrida anterior")
    fig.update_xaxes(title="", tickangle=-40)
    return _base(fig, "¿Cada corrida mejoró o empeoró?", 360)


def fig_modelos_en_corridas(largo: pd.DataFrame, metrica: str, titulo: str) -> go.Figure:
    d = largo.sort_values(["fecha", "hora"]).copy()
    d["corrida"] = d.groupby("modelo").cumcount() + 1
    fig = px.line(d, x="corrida", y=metrica, color="modelo", markers=True, color_discrete_sequence=PALETA)
    fig.update_xaxes(dtick=1, title="Número de corrida")
    return _base(fig, titulo, 360)


def fig_versiones(rankings: dict[str, pd.DataFrame], metrica: str, titulo: str) -> go.Figure:
    filas = [{"version": v, "modelo": r["modelo"], metrica: r[metrica]} for v, t in rankings.items() for _, r in t.iterrows()]
    fig = px.bar(pd.DataFrame(filas), x="modelo", y=metrica, color="version", barmode="group", color_discrete_sequence=PALETA)
    return _base(fig, titulo, 360)


def fig_aciertos(ac: pd.DataFrame) -> go.Figure:
    fig = px.bar(ac, x="modelo", y="aciertos_pct", text=ac["aciertos_pct"].round(0).astype(int).astype(str) + "%",
                 color_discrete_sequence=[PALETA[2]])
    fig.update_yaxes(range=[0, 110], title="% de canciones de prueba bien ubicadas")
    return _base(fig, "¿Acertó el clustering con las canciones de prueba?", 360)


# ------------------------------------------------------------------ frontera (probabilidades GMM)
def canciones_frontera(bundle: dict, top: int = 12) -> pd.DataFrame:
    m = bundle["models"].get("GMM")
    if m is None:
        return pd.DataFrame()
    proba = m["model"].predict_proba(bundle["X_cluster"])
    orden = np.argsort(-proba, axis=1)
    p1 = proba[np.arange(len(proba)), orden[:, 0]]
    p2 = proba[np.arange(len(proba)), orden[:, 1]]
    cg = m["cluster_genre"]
    d = pd.DataFrame({
        "archivo": bundle["archivos"], "genero": bundle["genres"], "ruta": bundle["rutas"],
        "grupo_1": [cg.get(int(l), ("?", 0))[0] for l in orden[:, 0]], "prob_1": p1,
        "grupo_2": [cg.get(int(l), ("?", 0))[0] for l in orden[:, 1]], "prob_2": p2,
    })
    d["indecision"] = 1 - (d["prob_1"] - d["prob_2"])
    return d.sort_values("indecision", ascending=False).head(top).reset_index(drop=True)


# ------------------------------------------------------------------ interpretación (reglas, sin redes neuronales)
def lectura_resultados(ranking: pd.DataFrame, asig: pd.DataFrame, ref: dict | None) -> list[str]:
    top = ranking.iloc[0]
    sil, ari = float(top["silhouette"]), float(top["ari"])
    txt = [f"El modelo mejor posicionado es {top['modelo']} ({top['params']}): forma {int(top['n_clusters'])} grupos "
           f"con {top['pct_ruido']:.0f}% de canciones sin grupo."]
    nivel_s = "fuerte" if sil >= 0.5 else "moderada" if sil >= 0.25 else "débil"
    txt.append(f"Silueta {sil:.2f}: la separación entre grupos es {nivel_s}. Mide qué tan compactos y distintos son los grupos usando solo el sonido, sin mirar los géneros.")
    nivel_a = "alta" if ari >= 0.5 else "parcial" if ari >= 0.2 else "baja"
    txt.append(f"ARI {ari:.2f}: la coincidencia con los géneros reales es {nivel_a}. Esta métrica sí compara los grupos con la etiqueta de género (el ruido cuenta como un grupo más).")
    if ref and ref.get("ari") is not None:
        diff = ari - float(ref["ari"])
        txt.append(f"Como referencia, KMeans forzado a k={ref['k']} (el número de géneros) obtiene ARI {float(ref['ari']):.2f}; "
                   f"el modelo elegido está {'por encima' if diff > 0.02 else 'por debajo' if diff < -0.02 else 'al mismo nivel'}.")
    col = str(top["modelo"])
    if col in asig.columns:
        p = pureza_por_cluster(asig, col)
        if not p.empty:
            mezcla = p[p["pureza"] < 0.7]
            if len(mezcla):
                lista = ", ".join(f"{r.cluster} ({r.pureza:.0%} {r.genero_dominante})" for r in mezcla.itertuples())
                txt.append(f"Grupos que mezclan géneros: {lista}. Son las canciones donde los géneros suenan parecido.")
            else:
                txt.append("Todos los grupos tienen un género claramente dominante (pureza de 70% o más).")
            rep = p["genero_dominante"].value_counts()
            rep = rep[rep > 1]
            if len(rep):
                txt.append("Géneros repartidos en varios grupos (hay subestilos o variaciones sonoras): " + ", ".join(rep.index) + ".")
    return txt
