"""Dashboard: mapa sonoro del folklore boliviano (clustering no supervisado de canciones).

Ejecutar desde la raíz del proyecto:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import bitacora  # noqa: E402
import graficos as gf  # noqa: E402
import mlops  # noqa: E402
import predict_new as pn  # noqa: E402
from config import DATA_DIR, MUSICA_DIR, clips_dir, features_csv, load_versions, results_dir  # noqa: E402

st.set_page_config(page_title="Mapa sonoro del folklore boliviano", layout="wide")

MODO_LOCAL = MUSICA_DIR.exists()  # en la nube no se sube la carpeta musica/
ASSETS = ROOT / "assets"
PAGINAS = ["Inicio", "Canciones", "Algoritmos y métricas", "Análisis profundo", "Mapa sonoro", "Géneros",
           "Probar canción", "Versiones y MLOps", "Bitácora", "Método"]
PNG_ENTRENAMIENTO = {
    "KMeans": [("kmeans_curvas.png", "Curva del codo y silueta según k")],
    "Jerarquico": [("dendrograma.png", "Dendrograma (cómo se van uniendo las canciones)")],
    "GMM": [("gmm_curvas.png", "BIC y AIC según el número de componentes")],
    "DBSCAN": [("kdistance.png", "Distancia al k-ésimo vecino (elige eps)")],
    "OPTICS": [("optics_reachability.png", "Gráfico de alcanzabilidad")],
    "HDBSCAN": [],
}
EXPLICA = {
    "KMeans": "Reparte las canciones en k grupos buscando centros que minimicen la distancia interna.",
    "Jerarquico": "Une las canciones más parecidas paso a paso (método Ward) y corta el árbol en k grupos.",
    "GMM": "Supone que cada grupo es una nube gaussiana; cada canción tiene una probabilidad por grupo.",
    "DBSCAN": "Busca zonas densas de canciones; lo que queda aislado se marca como ruido.",
    "OPTICS": "Como DBSCAN pero sin fijar una sola densidad; ordena las canciones por alcanzabilidad.",
    "HDBSCAN": "Versión jerárquica de DBSCAN: encuentra grupos de distinta densidad y descarta el ruido.",
}

CSS = """
<style>
.franja{height:9px;border-radius:2px;margin:0 0 18px 0;
 background:repeating-linear-gradient(90deg,#B3261E 0 40px,#D98A0B 40px 64px,#1F6F6B 64px 104px,#6B2D5C 104px 122px,#FAF6EE 122px 130px)}
h1,h2,h3{font-family:Georgia,'Times New Roman',serif;letter-spacing:-0.01em}
[data-testid="stMetric"]{background:#fff;border:1px solid #E6DCCB;border-radius:6px;padding:10px 14px}
.tarjeta{background:#fff;border:1px solid #E6DCCB;border-left:5px solid #B3261E;border-radius:6px;padding:14px 18px;margin:8px 0}
.nota{color:#6b6259;font-size:.9rem}
</style>
"""


# ------------------------------------------------------------------ utilidades de datos
@st.cache_data(show_spinner=False)
def _csv(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def leer_csv(path: Path) -> pd.DataFrame | None:
    return _csv(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_data(show_spinner=False)
def _features(seg: int, split: str, stamp: float) -> pd.DataFrame:
    return pn.cargar_features(seg, split)


def leer_features(seg: int, split: str = "train") -> pd.DataFrame | None:
    p = features_csv(seg, split)
    return _features(seg, split, p.stat().st_mtime) if p.exists() else None


@st.cache_resource(show_spinner=False)
def _bundle(seg: int, stamp: float) -> dict:
    return pn.cargar_bundle(seg)


def leer_bundle(seg: int) -> dict | None:
    p = pn.bundle_path(seg)
    if not p.exists():
        return None
    try:
        return _bundle(seg, p.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo abrir el paquete de modelos de esta versión ({exc}). Vuelve a entrenar con la misma versión de scikit-learn.")
        return None


def imagen(*partes: str) -> Path | None:
    base = ASSETS.joinpath(*partes)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = base.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def contexto(seg: int, vname: str) -> dict:
    r = results_dir(seg)
    ref = r / "referencia_k_generos.json"
    meta = leer_csv(DATA_DIR / "canciones_metadata.csv")
    return {
        "seg": seg, "version": vname,
        "ranking": leer_csv(r / "ranking.csv"), "asig": leer_csv(r / "asignaciones.csv"),
        "ref": json.loads(ref.read_text(encoding="utf-8")) if ref.exists() else None,
        "feats": leer_features(seg), "bundle": leer_bundle(seg), "meta": meta,
    }


def sin_resultados(ctx: dict) -> bool:
    if ctx["ranking"] is None or ctx["asig"] is None:
        st.warning(f"Todavía no hay resultados para {ctx['version']} ({ctx['seg']} s). "
                   f"Corre: python src/main.py --all --duracion {ctx['seg']}")
        return True
    return False


def sin_modelos(ctx: dict) -> bool:
    if ctx["bundle"] is None:
        st.warning("Esta versión no tiene modelos guardados. Corre `python src/main.py --cluster` para generarlos.")
        return True
    return False


def reproducir(ruta: str, seg: int, split: str = "train") -> bool:
    p = clips_dir(seg, split) / ruta
    if p.exists():
        st.audio(str(p))
        return True
    st.caption("El clip recortado no está en esta copia del proyecto (solo se guarda en la computadora donde se entrenó).")
    return False


def enlace_youtube(meta: pd.DataFrame | None, ruta: str) -> str:
    if meta is None or "youtube_url" not in meta.columns:
        return ""
    fila = meta[meta["ruta"] == ruta]
    return str(fila["youtube_url"].iloc[0]).strip() if len(fila) and pd.notna(fila["youtube_url"].iloc[0]) else ""


def titulo(texto: str, sub: str = "") -> None:
    st.markdown("<div class='franja'></div>", unsafe_allow_html=True)
    st.title(texto)
    if sub:
        st.markdown(f"<span class='nota'>{sub}</span>", unsafe_allow_html=True)


# ------------------------------------------------------------------ páginas
def pagina_inicio(ctx: dict) -> None:
    portada = imagen("portada")
    if portada:
        st.image(str(portada), width="stretch")
    titulo("Mapa sonoro del folklore boliviano",
           "¿Se pueden agrupar los géneros del folklore boliviano solo por cómo suenan, sin decirle al algoritmo cuál es cuál?")
    if sin_resultados(ctx):
        return
    rk, asig = ctx["ranking"], ctx["asig"]
    top = rk.iloc[0]
    c = st.columns(4)
    c[0].metric("Canciones analizadas", len(asig))
    c[1].metric("Géneros", asig["genero"].nunique())
    c[2].metric("Mejor modelo", str(top["modelo"]))
    c[3].metric("Coincidencia con los géneros (ARI)", f"{top['ari']:.2f}")
    st.caption(f"Versión activa: {ctx['version']} · clips de {ctx['seg']} s")

    a, b, d = st.columns(3)
    a.markdown("<div class='tarjeta'><b>Qué hicimos</b><br>Recortamos el centro de cada canción, extrajimos rasgos del sonido "
               "(timbre, tonalidad, ritmo, energía) y probamos 6 algoritmos de agrupación sin usar las etiquetas.</div>", unsafe_allow_html=True)
    b.markdown("<div class='tarjeta'><b>Cómo se evalúa</b><br>Con métricas internas (qué tan compactos son los grupos) "
               "y externas (cuánto coinciden con el género real).</div>", unsafe_allow_html=True)
    d.markdown("<div class='tarjeta'><b>Para qué sirve</b><br>Organizar archivos y catálogos de música boliviana, sugerir canciones "
               "parecidas y detectar piezas que mezclan géneros.</div>", unsafe_allow_html=True)

    st.subheader("Los géneros")
    generos = sorted(asig["genero"].unique())
    cols = st.columns(3)
    for i, g in enumerate(generos):
        with cols[i % 3]:
            img = imagen("generos", g)
            if img:
                st.image(str(img), caption=g, width="stretch")
            else:
                n = int((asig["genero"] == g).sum())
                st.markdown(f"<div class='tarjeta'><b>{g}</b><br><span class='nota'>{n} canciones</span></div>", unsafe_allow_html=True)
    if MODO_LOCAL and not any(imagen("generos", g) for g in generos):
        st.caption("Para mostrar fotos: guárdalas en assets/generos/<Género>.jpg (el nombre igual a la carpeta del género) y assets/portada.jpg.")


def pagina_canciones(ctx: dict) -> None:
    titulo("Canciones", "Lista de la muestra de entrenamiento: escucha el clip recortado o abre el video.")
    if sin_resultados(ctx):
        return
    asig, meta, seg = ctx["asig"].copy(), ctx["meta"], ctx["seg"]
    mejor = ctx["ranking"].iloc[0]["modelo"]
    asig["Cluster (mejor modelo)"] = gf.etiquetas_cluster(asig[mejor])
    if meta is not None:
        cols = [c for c in ("ruta", "artista", "titulo", "youtube_url") if c in meta.columns]
        asig = asig.merge(meta[cols], on="ruta", how="left")
    c1, c2 = st.columns([2, 1])
    gens = c1.multiselect("Género", sorted(asig["genero"].unique()))
    q = c2.text_input("Buscar por nombre")
    d = asig[asig["genero"].isin(gens)] if gens else asig
    if q:
        d = d[d["archivo"].str.contains(q, case=False, na=False)]
    mostrar = [c for c in ("archivo", "genero", "Cluster (mejor modelo)", "artista", "titulo") if c in d.columns]
    st.dataframe(d[mostrar].rename(columns={"archivo": "Archivo", "genero": "Género", "artista": "Artista", "titulo": "Título"}),
                 hide_index=True, width="stretch", height=300)
    st.caption(f"{len(d)} canciones. Cluster según {mejor}, el mejor modelo de la versión {ctx['version']}.")
    if d.empty:
        return
    elegida = st.selectbox("Escuchar", d["archivo"].tolist())
    fila = d[d["archivo"] == elegida].iloc[0]
    izq, der = st.columns(2)
    with izq:
        st.subheader(elegida)
        st.write(f"Género: **{fila['genero']}** · {fila['Cluster (mejor modelo)']}")
        reproducir(fila["ruta"], seg)
        img = imagen("generos", fila["genero"])
        if img:
            st.image(str(img), width=260)
    with der:
        url = enlace_youtube(meta, fila["ruta"])
        if url:
            st.video(url)
            st.link_button("Abrir en YouTube", url)
        else:
            st.caption("Esta canción no tiene enlace de YouTube en data/canciones_metadata.csv.")
    modelos = [m for m in pn.MODELOS if m in ctx["asig"].columns]
    detalle = ctx["asig"][ctx["asig"]["archivo"] == elegida][modelos].iloc[0]
    st.dataframe(pd.DataFrame({"Modelo": modelos, "Cluster asignado": [gf.etiquetas_cluster(pd.Series([detalle[m]])).iloc[0] for m in modelos]}),
                 hide_index=True)


def pagina_algoritmos(ctx: dict) -> None:
    titulo("Algoritmos y métricas", f"Los 6 algoritmos aplicados a la versión {ctx['version']} ({ctx['seg']} s).")
    if sin_resultados(ctx):
        return
    rk, asig = ctx["ranking"], ctx["asig"]
    top = rk.iloc[0]
    st.markdown(f"<div class='tarjeta'><b>Mejor según el ranking: {top['modelo']}</b> · {top['params']}</div>", unsafe_allow_html=True)
    for linea in gf.lectura_resultados(rk, asig, ctx["ref"]):
        st.markdown(f"- {linea}")
    pretty = rk.rename(columns={"modelo": "Modelo", "params": "Parámetros", "n_clusters": "Grupos", "pct_ruido": "% ruido",
                                "silhouette": "Silueta", "calinski_harabasz": "Calinski-Harabasz", "davies_bouldin": "Davies-Bouldin",
                                "ari": "ARI", "nmi": "NMI", "valido": "Válido"}).drop(columns=["inertia"], errors="ignore")
    estilo = pretty.style.apply(lambda r: ["background-color:#F6E3B4" if r.name == 0 else "" for _ in r], axis=1)\
        .format({"% ruido": "{:.1f}", "Silueta": "{:.3f}", "Calinski-Harabasz": "{:.1f}", "Davies-Bouldin": "{:.3f}", "ARI": "{:.3f}", "NMI": "{:.3f}"}, na_rep="—")
    st.dataframe(estilo, hide_index=True, width="stretch")
    st.plotly_chart(gf.fig_barras_metricas(rk), width="stretch")

    st.subheader("Detalle por algoritmo")
    modelos = [m for m in rk["modelo"].tolist() if m in asig.columns]
    for tab, m in zip(st.tabs(modelos), modelos):
        with tab:
            fila = rk[rk["modelo"] == m].iloc[0]
            st.caption(EXPLICA.get(m, ""))
            k = st.columns(5)
            k[0].metric("Grupos", int(fila["n_clusters"]))
            k[1].metric("Silueta", f"{fila['silhouette']:.2f}")
            k[2].metric("ARI", f"{fila['ari']:.2f}")
            k[3].metric("Davies-Bouldin", f"{fila['davies_bouldin']:.2f}")
            k[4].metric("% ruido", f"{fila['pct_ruido']:.0f}%")
            st.caption(f"Parámetros elegidos: {fila['params']}")
            d = asig.copy()
            d["grupo"] = gf.etiquetas_cluster(d[m])
            st.plotly_chart(gf.fig_mapa(d, "grupo", f"Mapa sonoro según {m}"), width="stretch", key=f"mapa_{m}")
            a, b = st.columns(2)
            a.plotly_chart(gf.fig_heatmap_genero_cluster(asig, m), width="stretch", key=f"hm_{m}")
            b.plotly_chart(gf.fig_pureza(asig, m), width="stretch", key=f"pu_{m}")
            figs = [(f, c) for f, c in PNG_ENTRENAMIENTO.get(m, []) if (results_dir(ctx["seg"]) / f).exists()]
            if figs:
                with st.expander("Gráficos del entrenamiento"):
                    for f, c in figs:
                        st.image(str(results_dir(ctx["seg"]) / f), caption=c)


def pagina_analisis(ctx: dict) -> None:
    titulo("Análisis profundo", "Gráficos para entender por qué los algoritmos agrupan así.")
    if sin_resultados(ctx) or sin_modelos(ctx):
        return
    b, asig, feats = ctx["bundle"], ctx["asig"], ctx["feats"]
    modelos = [m for m in pn.MODELOS if m in asig.columns]
    t = st.tabs(["Coincidencia entre algoritmos", "Silueta por canción", "Canciones frontera", "Distancias", "Rasgos que separan"])
    with t[0]:
        st.plotly_chart(gf.fig_consenso(asig, modelos), width="stretch")
        st.caption("Si dos algoritmos con lógicas distintas coinciden, el agrupamiento es más confiable.")
    with t[1]:
        m = st.selectbox("Modelo", modelos, key="sil_modelo")
        st.plotly_chart(gf.fig_silueta(b["X_cluster"], b["models"][m]["labels"]), width="stretch")
        st.caption("Barras cortas o negativas = canciones que quedarían mejor en otro grupo.")
    with t[2]:
        st.write("Con la mezcla gaussiana cada canción tiene una probabilidad por grupo. Estas son las más indecisas: "
                 "suenan a dos géneros a la vez.")
        fr = gf.canciones_frontera(b)
        if fr.empty:
            st.info("No hay modelo GMM guardado.")
        else:
            vista = fr.assign(**{"Grupo 1": fr["grupo_1"] + " (" + (fr["prob_1"] * 100).round(0).astype(int).astype(str) + "%)",
                                 "Grupo 2": fr["grupo_2"] + " (" + (fr["prob_2"] * 100).round(0).astype(int).astype(str) + "%)"})
            st.dataframe(vista[["archivo", "genero", "Grupo 1", "Grupo 2"]].rename(columns={"archivo": "Archivo", "genero": "Género real"}),
                         hide_index=True, width="stretch")
            sel = st.selectbox("Escuchar una canción frontera", fr["archivo"].tolist())
            reproducir(fr[fr["archivo"] == sel]["ruta"].iloc[0], ctx["seg"])
    with t[3]:
        st.plotly_chart(gf.fig_distancias(b["X_cluster"], b["genres"]), width="stretch")
    with t[4]:
        if feats is not None:
            st.plotly_chart(gf.fig_importancia(feats), width="stretch")
            st.caption("Valores altos: ese rasgo cambia mucho entre géneros y poco dentro de cada género.")


def pagina_mapa(ctx: dict) -> None:
    titulo("Mapa sonoro", "Cada punto es una canción; las más cercanas suenan más parecido.")
    if sin_resultados(ctx):
        return
    asig = ctx["asig"].copy()
    modelos = [m for m in pn.MODELOS if m in asig.columns]
    c1, c2 = st.columns([2, 1])
    modo = c1.selectbox("Colorear por", ["Género real"] + modelos)
    asig["grupo"] = asig["genero"] if modo == "Género real" else gf.etiquetas_cluster(asig[modo])
    estrellas = None
    if c2.checkbox("Mostrar canciones de prueba") and ctx["bundle"] is not None:
        tp = leer_features(ctx["seg"], "prueba")
        if tp is not None and len(tp):
            xy = [pn.transformar(ctx["bundle"], r.to_dict())[2] for _, r in tp.iterrows()]
            estrellas = pd.DataFrame({"archivo": tp["archivo"], "pc1": [p[0] for p in xy], "pc2": [p[1] for p in xy]})
        else:
            st.info("No hay canciones de prueba procesadas para esta versión.")
    st.plotly_chart(gf.fig_mapa(asig, "grupo", f"Mapa sonoro · {ctx['version']} · {modo}", estrellas), width="stretch")
    sel = st.selectbox("Escuchar una canción del mapa", asig["archivo"].tolist())
    fila = asig[asig["archivo"] == sel].iloc[0]
    st.write(f"**{sel}** · {fila['genero']}")
    reproducir(fila["ruta"], ctx["seg"])


def pagina_generos(ctx: dict) -> None:
    titulo("Los géneros", "Cómo suena cada género y cuáles se parecen.")
    feats = ctx["feats"]
    if feats is None:
        st.warning("Faltan las features de esta versión.")
        return
    generos = sorted(feats["genero"].unique())
    elegidos = st.multiselect("Géneros en el radar", generos, default=generos[:4])
    a, b = st.columns(2)
    a.plotly_chart(gf.fig_radar(feats, elegidos), width="stretch")
    b.plotly_chart(gf.fig_similitud(feats), width="stretch")
    sim = gf.similitud_generos(feats)
    for g1, g2, v in gf.pares_parecidos(sim):
        st.markdown(f"- **{g1}** y **{g2}** son los más parecidos (similitud {v:.2f}).")
    col = "tempo_bpm" if "tempo_bpm" in feats.columns else None
    if col:
        fig = px.box(feats, x="genero", y=col, color="genero", color_discrete_map=gf.mapa_colores(feats["genero"].unique()))
        fig.update_layout(showlegend=False)
        st.plotly_chart(gf._base(fig, "Tempo estimado por género (BPM)", 380), width="stretch")


# ------------------------------------------------------------------ probar canciones
@st.cache_data(show_spinner=False)
def _tabla_prueba(seg: int, stamp_b: float, stamp_f: float) -> pd.DataFrame:
    return pn.predecir_tabla(_bundle(seg, stamp_b), pn.cargar_features(seg, "prueba"))


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_youtube(consulta: str, clave: str, n: int = 5) -> list[dict]:
    """Busca videos con la API oficial de YouTube (100 unidades de cuota por búsqueda; se guarda 1 hora)."""
    params = {"part": "snippet", "q": consulta, "type": "video", "videoEmbeddable": "true", "maxResults": n, "key": clave}
    url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.load(r)
    return [{"id": it["id"]["videoId"], "titulo": html.unescape(it["snippet"]["title"]),
             "canal": html.unescape(it["snippet"]["channelTitle"])} for it in data.get("items", [])]


def clave_youtube() -> str:
    try:
        return str(st.secrets.get("YOUTUBE_API_KEY", "")).strip()
    except Exception:  # noqa: BLE001  (no hay archivo de secretos)
        return ""


def mostrar_prediccion(ctx: dict, res: dict, genero_real: str | None = None, clave: str = "p") -> None:
    tabla = res["tabla"].copy()
    if genero_real:
        tabla["¿Acierta?"] = np.where(tabla["genero_predicho"] == genero_real, "sí", "no")
    tabla["pureza"] = (tabla["pureza"] * 100).round(0).astype(int).astype(str) + "%"
    tabla["cluster"] = gf.etiquetas_cluster(tabla["cluster"])
    st.dataframe(tabla.rename(columns={"modelo": "Modelo", "cluster": "Cluster", "genero_predicho": "Género del cluster", "pureza": "Pureza del cluster"}),
                 hide_index=True, width="stretch")
    votos = res["tabla"]["genero_predicho"].value_counts()
    st.markdown(f"Género más votado: **{votos.index[0]}** ({votos.iloc[0]} de {len(res['tabla'])} modelos). "
                "El género de un cluster es el que más canciones de la muestra tiene en él.")
    asig = ctx["asig"].copy()
    if "pc1" in asig.columns:
        asig["grupo"] = asig["genero"]
        estrella = pd.DataFrame({"archivo": ["Canción probada"], "pc1": [res["xy"][0]], "pc2": [res["xy"][1]]})
        st.plotly_chart(gf.fig_mapa(asig, "grupo", "Dónde cae en el mapa sonoro (estrella)", estrella),
                        width="stretch", key=f"mapa_{clave}")
    st.markdown("**Canciones de la muestra más parecidas**")
    for _, v in res["vecinos"].iterrows():
        c1, c2 = st.columns([2, 3])
        c1.write(f"{v['archivo']} · {v['genero']}")
        with c2:
            reproducir(v["ruta"], ctx["seg"])


def pagina_probar(ctx: dict) -> None:
    titulo("Probar una canción", f"Los modelos de la versión {ctx['version']} ({ctx['seg']} s) clasifican canciones que no estaban en la muestra.")
    if sin_resultados(ctx) or sin_modelos(ctx):
        return
    seg, b = ctx["seg"], ctx["bundle"]
    t1, t2, t3 = st.tabs(["Canciones de prueba", "Subir un audio", "Buscar en YouTube"])
    with t1:
        tp = leer_features(seg, "prueba")
        if tp is None or tp.empty:
            st.info("No hay canciones de prueba para esta versión. Guárdalas en musica_prueba/<Género>/ y corre `python src/main.py --prueba`.")
        else:
            tabla = _tabla_prueba(seg, pn.bundle_path(seg).stat().st_mtime, features_csv(seg, "prueba").stat().st_mtime)
            ac = pn.aciertos(tabla)
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(gf.fig_aciertos(ac), width="stretch")
            with c2:
                mejor = ac.sort_values(["aciertos_pct", "sin_grupo_pct"], ascending=[False, True]).iloc[0]
                st.metric("Modelo que mejor ubicó las pruebas", mejor["modelo"], f"{mejor['aciertos_pct']:.0f}%")
                st.dataframe(ac.rename(columns={"modelo": "Modelo", "aciertos_pct": "% aciertos", "sin_grupo_pct": "% sin grupo"}).round(1),
                             hide_index=True)
            vista = tabla[["archivo", "genero_real"] + [m for m in pn.MODELOS if m in tabla.columns]].copy()
            for m in pn.MODELOS:
                if m in vista.columns:
                    vista[m] = vista[m] + np.where(vista[m] == vista["genero_real"], "  ✓", "  ✗")
            st.dataframe(vista.rename(columns={"archivo": "Canción", "genero_real": "Género real"}), hide_index=True, width="stretch")
            sel = st.selectbox("Ver una canción de prueba", tabla["archivo"].tolist())
            fila = tp[tp["archivo"] == sel].iloc[0]
            reproducir(fila["ruta"], seg, "prueba")
            mostrar_prediccion(ctx, pn.predecir(b, fila.to_dict()), fila["genero"], clave="prueba")
            with st.expander("Comparar esta canción entre versiones"):
                filas = []
                for v, s in load_versions().items():
                    bv, tv = leer_bundle(s), leer_features(s, "prueba")
                    if bv is None or tv is None or sel not in tv["archivo"].values:
                        continue
                    r = pn.predecir(bv, tv[tv["archivo"] == sel].iloc[0].to_dict())["tabla"]
                    for _, x in r.iterrows():
                        filas.append({"Modelo": x["modelo"], "Versión": f"{v} ({s} s)",
                                      "Género": x["genero_predicho"] + ("  ✓" if x["genero_predicho"] == fila["genero"] else "  ✗")})
                if filas:
                    st.dataframe(pd.DataFrame(filas).pivot(index="Modelo", columns="Versión", values="Género"), width="stretch")
    with t2:
        st.write("Sube un audio y se clasifica con cada versión elegida (se corta el centro de la canción, igual que en el entrenamiento).")
        up = st.file_uploader("Audio (mp3, wav, ogg, flac)", type=["mp3", "wav", "ogg", "flac"])
        generos = sorted(ctx["asig"]["genero"].unique())
        real = st.selectbox("Género real (si lo conoces)", ["No sé"] + generos)
        elegidas = st.multiselect("Versiones a probar", list(load_versions()), default=[ctx["version"]])
        if up is not None and st.button("Clasificar"):
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
                tmp.write(up.getvalue())
            try:
                resultados = []
                for v in elegidas:
                    s = load_versions()[v]
                    cv, bv = contexto(s, v), leer_bundle(s)
                    if bv is None or cv["asig"] is None:
                        st.warning(f"{v} no tiene modelos entrenados.")
                        continue
                    feats, _ = pn.features_de_audio(Path(tmp.name), s)
                    resultados.append((cv, pn.predecir(bv, feats)))
                st.session_state["subida"] = (resultados, None if real == "No sé" else real)
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo clasificar el audio: {exc}")
        if "subida" in st.session_state:
            resultados, real_guardado = st.session_state["subida"]
            for cv, res in resultados:
                st.subheader(f"{cv['version']} · {cv['seg']} s")
                mostrar_prediccion(cv, res, real_guardado, clave=f"up_{cv['version']}")
    with t3:
        st.write("Busca una canción para escucharla o abrirla en YouTube. La aplicación no descarga nada: "
                 "si ya tienes el archivo de audio, súbelo en la pestaña anterior para clasificarlo.")
        q = st.text_input("Canción o artista", placeholder="Ejemplo: caporales Tupay")
        if q.strip():
            st.link_button("Abrir la búsqueda en YouTube", "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q.strip()))
            clave = clave_youtube()
            if not clave:
                st.caption("Para ver los resultados aquí mismo agrega YOUTUBE_API_KEY en .streamlit/secrets.toml (ver el README).")
            else:
                try:
                    for v in buscar_youtube(q.strip(), clave):
                        st.markdown(f"**{v['titulo']}** · {v['canal']}")
                        st.video(f"https://www.youtube.com/watch?v={v['id']}")
                except urllib.error.HTTPError as exc:
                    st.error("YouTube rechazó la búsqueda (puede haberse agotado la cuota diaria de la clave)." if exc.code == 403
                             else f"Error de YouTube: {exc.code}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"No se pudo buscar en YouTube: {exc}")


# ------------------------------------------------------------------ versiones y MLOps
def pagina_versiones(ctx: dict) -> None:
    titulo("Versiones y MLOps", "Qué versiones existen, cómo rinde cada corrida y si el trabajo va mejorando.")
    versiones = load_versions()
    st.markdown("<div class='tarjeta'><b>¿Qué es una versión?</b> Una duración de clip: v1 usa 30 s del centro de cada canción, "
                "v2 usa 60 s, y así. Todo lo demás (rasgos, PCA, 6 algoritmos, métricas) se aplica igual, por eso cada versión guarda "
                "sus propios modelos y se puede probar la misma canción con cada una.</div>", unsafe_allow_html=True)
    reg = mlops.con_deltas(mlops.leer_registro())

    filas = []
    for v, s in versiones.items():
        prod = reg[(reg["version"] == v) & (reg["etapa"] == "produccion")]
        if len(prod):
            r = prod.iloc[-1]
            filas.append({"Versión": v, "Clip (s)": s, "Canciones": int(r["n_canciones"]), "Mejor modelo": r["mejor_modelo"],
                          "Silueta": r["silhouette"], "ARI": r["ari"], "Entrenada": f"{r['fecha']} {r['hora']}", "Responsable": r["responsable"],
                          "Corridas": int((reg["version"] == v).sum())})
        else:
            filas.append({"Versión": v, "Clip (s)": s, "Mejor modelo": "sin entrenar"})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    rankings = {v: leer_csv(results_dir(s) / "ranking.csv") for v, s in versiones.items()}
    rankings = {v: r for v, r in rankings.items() if r is not None}
    if len(rankings) >= 2:
        st.subheader("Comparación entre versiones")
        a, b = st.columns(2)
        a.plotly_chart(gf.fig_versiones(rankings, "silhouette", "Silueta por algoritmo y versión"), width="stretch")
        b.plotly_chart(gf.fig_versiones(rankings, "ari", "ARI por algoritmo y versión"), width="stretch")
        nombres = list(rankings)
        c1, c2, c3 = st.columns(3)
        va, vb = c1.selectbox("Versión A", nombres, index=0), c2.selectbox("Versión B", nombres, index=min(1, len(nombres) - 1))
        aa = leer_csv(results_dir(versiones[va]) / "asignaciones.csv")
        ab = leer_csv(results_dir(versiones[vb]) / "asignaciones.csv")
        comunes = [m for m in pn.MODELOS if aa is not None and ab is not None and m in aa.columns and m in ab.columns]
        if comunes and va != vb:
            modelo = c3.selectbox("Algoritmo", comunes)
            from sklearn.metrics import adjusted_rand_score
            m = aa[["ruta", modelo]].merge(ab[["ruta", modelo]], on="ruta", suffixes=("_a", "_b"))
            st.metric(f"Coincidencia de {va} y {vb} con {modelo} (ARI)", f"{adjusted_rand_score(m[modelo + '_a'], m[modelo + '_b']):.2f}")
            st.caption("Una versión corta es el centro de la larga, así que es normal que se parezcan; si difieren, el tramo extra aporta información.")
            fig = gf.fig_sankey(aa, ab, modelo, modelo, va, vb)
            if fig:
                st.plotly_chart(fig, width="stretch")

    st.subheader("Registro de corridas (MLOps)")
    if reg.empty:
        st.info("Todavía no hay corridas registradas. Cada `python src/main.py --cluster` agrega una fila aquí.")
    else:
        mejoras, peores = int((reg["veredicto"] == "mejora").sum()), int((reg["veredicto"] == "empeora").sum())
        mejor = reg.sort_values(["ari", "silhouette"], ascending=False).iloc[0]
        k = st.columns(4)
        k[0].metric("Corridas registradas", len(reg))
        k[1].metric("Que mejoraron", mejoras)
        k[2].metric("Que empeoraron", peores)
        k[3].metric("Mejor corrida", mejor["run_id"], f"ARI {mejor['ari']:.2f}")
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(gf.fig_evolucion(reg, "ari", "ARI por corrida"), width="stretch")
        g2.plotly_chart(gf.fig_evolucion(reg, "silhouette", "Silueta por corrida"), width="stretch")
        g3.plotly_chart(gf.fig_evolucion(reg, "davies_bouldin", "Davies-Bouldin por corrida (más bajo = mejor)"), width="stretch")
        fig_d = gf.fig_deltas(reg)
        if fig_d is not None:
            st.plotly_chart(fig_d, width="stretch")
        else:
            st.caption("Con una sola corrida por versión aún no hay mejoras que medir. Vuelve a entrenar después de un cambio y aparecerá aquí.")
        largo = mlops.metricas_por_modelo(reg)
        if not largo.empty:
            c1, c2 = st.columns(2)
            vsel = c1.selectbox("Versión", sorted(largo["version"].unique()), key="mlops_v")
            msel = c2.selectbox("Métrica", ["ari", "silhouette", "nmi", "davies_bouldin", "pct_ruido"], key="mlops_m")
            st.plotly_chart(gf.fig_modelos_en_corridas(largo[largo["version"] == vsel], msel, f"{msel} de cada algoritmo en cada corrida · {vsel}"),
                            width="stretch")
        vista = reg.drop(columns=["metricas_modelos"]).sort_values(["fecha", "hora"], ascending=False)
        vista["datos_cambiaron"] = vista.groupby("version")["hash_datos"].transform(lambda s: s != s.shift(-1)) & vista.groupby("version")["hash_datos"].transform("count").gt(1)
        st.dataframe(vista, hide_index=True, width="stretch")
        st.download_button("Descargar registro (CSV)", vista.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                           "registro_mlops.csv", "text/csv")

    st.subheader("Crear una versión nueva")
    if not MODO_LOCAL:
        st.caption("Crear versiones requiere la carpeta musica/, que solo está en la computadora donde se entrena.")
        return
    with st.expander("Otra duración de clip (por ejemplo, 2 minutos)"):
        minutos = st.number_input("Duración del clip (minutos)", 0.5, 10.0, 2.0, 0.5)
        segs = int(round(minutos * 60))
        motivo = st.text_input("¿Por qué creas esta versión?")
        if segs in versiones.values():
            st.info("Ya existe una versión con esa duración; se volverá a entrenar.")
        if not motivo.strip():
            st.caption("Escribe el motivo para habilitar la creación.")
        if st.button("Crear y entrenar", disabled=not motivo.strip()):
            import os
            env = {**os.environ, "BITACORA_USER": bitacora.responsable_actual()}
            with st.spinner("Recortando, extrayendo rasgos y entrenando (puede tardar varios minutos)..."):
                r = subprocess.run([sys.executable, str(ROOT / "src" / "main.py"), "--all", "--duracion", str(segs), "--motivo", motivo.strip()],
                                   capture_output=True, text=True, env=env, cwd=str(ROOT))
            st.cache_data.clear()
            st.cache_resource.clear()
            (st.success if r.returncode == 0 else st.error)("Versión entrenada." if r.returncode == 0 else "Falló el entrenamiento.")
            st.code("\n".join((r.stdout + r.stderr).splitlines()[-25:]))


# ------------------------------------------------------------------ bitácora
def pagina_bitacora() -> None:
    titulo("Bitácora", "Registro de cambios, corridas, versiones y decisiones del proyecto.")
    sesion = st.session_state.setdefault("bitacora_sesion", pd.DataFrame(columns=bitacora.COLUMNAS))
    df = bitacora.unir(bitacora.leer(), sesion)
    if MODO_LOCAL:
        st.markdown("<div class='tarjeta'><b>Modo local.</b> Las entradas se guardan de forma permanente en bitacora/bitacora.csv "
                    "con fecha, hora y responsable.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='tarjeta'><b>Modo en línea.</b> Puedes agregar entradas y se ven aquí, pero el servidor no las guarda: "
                    "descarga el CSV actualizado y envíalo al responsable del proyecto para incorporarlo. "
                    "La hora es la del servidor.</div>", unsafe_allow_html=True)

    if MODO_LOCAL and bitacora.hay_linea_base():
        cambios = bitacora.detectar_cambios()
        if cambios:
            st.subheader("Cambios detectados en el código")
            for c in cambios:
                with st.expander(f"{c['archivo']}  (+{c['agregadas']} / -{c['quitadas']} líneas)"):
                    st.code(c["diff"], language="diff")
            with st.form("cambios_detectados"):
                motivo = st.text_area("¿Por qué hiciste estos cambios?")
                seccion = st.text_input("Sección", "General")
                if st.form_submit_button("Guardar en la bitácora") and motivo.strip():
                    n = bitacora.registrar_cambios(cambios, motivo, bitacora.responsable_actual(), seccion or "General")
                    st.success(f"{n} cambio(s) guardados.")
                    st.rerun()

    st.subheader("Agregar una entrada")
    secciones = sorted(set(bitacora.SECCIONES_BASE) | set(df["seccion"].replace("", pd.NA).dropna()))
    with st.form("nueva_entrada", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        resp = c1.text_input("Responsable", bitacora.responsable_actual())
        tipo = c2.selectbox("Tipo", bitacora.TIPOS, index=bitacora.TIPOS.index("nota"))
        version = c3.selectbox("Versión", [""] + list(load_versions()))
        c4, c5 = st.columns(2)
        seccion = c4.selectbox("Sección", secciones)
        nueva = c5.text_input("O crea una sección nueva")
        archivo = st.text_input("Archivo o script (opcional)")
        motivo = st.text_area("Motivo o justificación")
        fragmento = st.text_area("Fragmento de código o detalle (opcional)")
        enviar = st.form_submit_button("Guardar entrada")
    if enviar:
        if not motivo.strip():
            st.error("Escribe el motivo o justificación.")
        else:
            fila = bitacora.nueva_fila(tipo, motivo, nueva.strip() or seccion, archivo, version, fragmento, resp.strip() or None)
            if MODO_LOCAL:
                bitacora.guardar_df(bitacora.unir(bitacora.leer(), pd.DataFrame([fila])))
            else:
                st.session_state["bitacora_sesion"] = bitacora.unir(sesion, pd.DataFrame([fila]))
            st.rerun()

    st.subheader("Historial")
    if df.empty:
        st.info("Aún no hay entradas. Corre `python src/bitacora.py inicial` o agrega una arriba.")
        return
    f1, f2, f3, f4 = st.columns(4)
    tipos = f1.multiselect("Tipo", sorted(df["tipo"].unique()))
    secs = f2.multiselect("Sección", sorted(df["seccion"].unique()))
    resps = f3.multiselect("Responsable", sorted(df["responsable"].unique()))
    texto = f4.text_input("Buscar en motivo")
    d = df
    if tipos:
        d = d[d["tipo"].isin(tipos)]
    if secs:
        d = d[d["seccion"].isin(secs)]
    if resps:
        d = d[d["responsable"].isin(resps)]
    if texto:
        d = d[d["motivo"].str.contains(texto, case=False, na=False)]
    st.dataframe(d.drop(columns=["id"]), hide_index=True, width="stretch")
    ver = st.selectbox("Ver el fragmento de una entrada", [""] + d[d["fragmento"] != ""]["id"].tolist(),
                       format_func=lambda i: "" if not i else f"{d[d['id'] == i]['fecha'].iloc[0]} · {d[d['id'] == i]['archivo'].iloc[0] or d[d['id'] == i]['tipo'].iloc[0]}")
    if ver:
        st.code(d[d["id"] == ver]["fragmento"].iloc[0], language="diff")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Descargar CSV", bitacora.a_csv_bytes(d), "bitacora.csv", "text/csv")
    e2.download_button("Descargar Excel", bitacora.a_excel_bytes(d), "bitacora.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with e3:
        subido = st.file_uploader("Unir un CSV de bitácora", type=["csv"], label_visibility="collapsed")
        if subido is not None and st.button("Unir al historial"):
            nuevo = pd.read_csv(subido, dtype=str, encoding="utf-8-sig").fillna("")
            if not set(bitacora.COLUMNAS) <= set(nuevo.columns):
                st.error("El CSV no tiene las columnas de la bitácora.")
            else:
                if MODO_LOCAL:
                    bitacora.guardar_df(bitacora.unir(bitacora.leer(), nuevo[bitacora.COLUMNAS]))
                else:
                    st.session_state["bitacora_sesion"] = bitacora.unir(sesion, nuevo[bitacora.COLUMNAS])
                st.rerun()


# ------------------------------------------------------------------ método
def pagina_metodo() -> None:
    titulo("Método", "Cómo se construyó el análisis y cómo leer cada métrica.")
    with st.expander("Pipeline paso a paso", expanded=True):
        st.markdown(
            "1. **Recorte:** se quita el silencio y se toma el centro de cada canción según la versión (30 s, 60 s...). Solo entran canciones que alcanzan esa duración.\n"
            "2. **Rasgos:** MFCC (timbre), cromagrama (notas), contraste espectral, brillo, energía, ataques y ritmo (tempo y pulsos por segundo).\n"
            "3. **Preparación:** se estandarizan los rasgos y se reducen con PCA (90% de la varianza para KMeans, GMM y jerárquico; menos componentes para los métodos de densidad).\n"
            "4. **Agrupación:** KMeans, jerárquico (Ward), GMM, DBSCAN, OPTICS y HDBSCAN, cada uno con sus parámetros elegidos por métricas internas.\n"
            "5. **Evaluación y ranking:** métricas internas y externas; se guardan los modelos de cada versión y la corrida queda en el registro.")
    with st.expander("Cómo leer las métricas"):
        st.markdown(
            "- **Silueta** (−1 a 1): qué tan cerca está cada canción de su grupo frente al grupo vecino. Más alto es mejor. No usa los géneros.\n"
            "- **Calinski-Harabasz:** separación entre grupos sobre dispersión dentro de ellos. Más alto es mejor.\n"
            "- **Davies-Bouldin:** parecido promedio entre cada grupo y su vecino más cercano. Más bajo es mejor.\n"
            "- **ARI y NMI:** cuánto coinciden los grupos con los géneros reales (1 = coincidencia perfecta). El ruido cuenta como un grupo más.\n"
            "- **Pureza:** porcentaje del género dominante dentro de cada cluster.")
    with st.expander("Qué hay de inteligencia artificial aquí"):
        st.markdown("Todo es **aprendizaje automático no supervisado clásico**: los algoritmos descubren grupos sin conocer los géneros. "
                    "Los textos de interpretación se generan con reglas sobre las métricas (no con redes neuronales ni modelos de lenguaje).")
    with st.expander("Limitaciones"):
        st.markdown("- La versión larga contiene a la corta (es el mismo tramo central ampliado), así que no son experimentos independientes.\n"
                    "- Un género puede sonar distinto según la grabación o el arreglo; el sonido no siempre coincide con la etiqueta.\n"
                    "- Con pocas canciones por género, las métricas son más inestables.\n"
                    "- Los clusters de una canción nueva se asignan por cercanía a la muestra (centroide o vecino más cercano), no reentrenando.")


# ------------------------------------------------------------------ aplicación
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    versiones = load_versions()
    etiquetas = {f"{v} · {s} s": (v, s) for v, s in versiones.items()}
    with st.sidebar:
        logo = imagen("logo")
        if logo:
            st.image(str(logo), width="stretch")
        st.markdown("### Mapa sonoro\nFolklore boliviano")
        eleg = st.selectbox("Versión", list(etiquetas), key="version")
        pagina = st.radio("Sección", PAGINAS, key="pagina")
        st.caption("Modo local: se guardan los cambios." if MODO_LOCAL else "Modo en línea: consulta y pruebas.")
    vname, seg = etiquetas[eleg]
    ctx = contexto(seg, vname)
    paginas = {
        "Inicio": lambda: pagina_inicio(ctx), "Canciones": lambda: pagina_canciones(ctx),
        "Algoritmos y métricas": lambda: pagina_algoritmos(ctx), "Análisis profundo": lambda: pagina_analisis(ctx),
        "Mapa sonoro": lambda: pagina_mapa(ctx), "Géneros": lambda: pagina_generos(ctx),
        "Probar canción": lambda: pagina_probar(ctx), "Versiones y MLOps": lambda: pagina_versiones(ctx),
        "Bitácora": pagina_bitacora, "Método": pagina_metodo,
    }
    paginas[pagina]()


main()
