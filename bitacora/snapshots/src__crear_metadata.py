"""Crea y completa data/canciones_metadata.csv (artista, título y enlace de YouTube por canción).

Uso (desde la raíz del proyecto):
  python src/crear_metadata.py                  # crea la plantilla o añade canciones nuevas
  python src/crear_metadata.py --auto           # además busca en YouTube las que faltan
  python src/crear_metadata.py --auto --max 90  # límite de búsquedas por corrida

Para --auto hace falta una clave de la API de YouTube Data v3 en la variable
de entorno YOUTUBE_API_KEY. Cada búsqueda gasta 100 unidades de la cuota diaria
(por defecto 10 000), así que alcanzan unas 100 búsquedas por día; el script
se detiene solo si se agota y se puede continuar al día siguiente.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import AUDIO_EXTS, DATA_DIR, MUSICA_DIR

META_CSV = DATA_DIR / "canciones_metadata.csv"
COLUMNS = [
    "ruta",
    "archivo",
    "genero",
    "artista",
    "titulo",
    "youtube_url",
    "youtube_titulo",
    "youtube_canal",
    "estado",
    "buscar_en_youtube",
]
SEPARADORES = (" - ", " – ", " — ", " | ")


class QuotaError(Exception):
    """La cuota diaria de la API de YouTube se agotó."""


def split_name(stem: str) -> tuple[str, str]:
    """Separa 'Artista - Título' del nombre del archivo."""
    for sep in SEPARADORES:
        if sep in stem:
            artista, titulo = stem.split(sep, 1)
            return artista.strip(), titulo.strip()
    return "", stem.strip()


def search_link(artista: str, titulo: str) -> str:
    query = f"{artista} {titulo}".strip()
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)


def list_tracks() -> list[dict]:
    """Una fila por canción de musica/; 'ruta' coincide con la columna ruta de features_*.csv."""
    if not MUSICA_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de música: {MUSICA_DIR}")
    rows = []
    for path in sorted(MUSICA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = path.relative_to(MUSICA_DIR)
        genero = rel.parts[0] if len(rel.parts) > 1 else "sin_genero"
        artista, titulo = split_name(path.stem)
        rows.append(
            {
                "ruta": f"{genero}/{path.stem}.wav",
                "archivo": path.stem,
                "genero": genero,
                "artista": artista,
                "titulo": titulo,
                "youtube_url": "",
                "youtube_titulo": "",
                "youtube_canal": "",
                "estado": "pendiente",
                "buscar_en_youtube": search_link(artista, titulo),
            }
        )
    return rows


def build_table() -> pd.DataFrame:
    """Une las canciones de la carpeta con lo ya completado en el CSV (no pisa nada)."""
    tracks = pd.DataFrame(list_tracks(), columns=COLUMNS)
    if META_CSV.exists():
        old = pd.read_csv(META_CSV, dtype=str, encoding="utf-8-sig").fillna("")
        for col in COLUMNS:
            if col not in old.columns:
                old[col] = ""
        new = tracks[~tracks["ruta"].isin(old["ruta"])]
        df = pd.concat([old[COLUMNS], new], ignore_index=True)
    else:
        df = tracks
    return df.sort_values(["genero", "archivo"]).reset_index(drop=True)


def youtube_search(query: str, key: str) -> dict | None:
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoEmbeddable": "true",
        "maxResults": 1,
        "key": key,
    }
    url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 403 and "quota" in body.lower():
            raise QuotaError from exc
        raise RuntimeError(f"Error {exc.code} de la API de YouTube: {body[:200]}") from exc
    items = data.get("items", [])
    if not items:
        return None
    item = items[0]
    return {
        "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
        "titulo": html.unescape(item["snippet"]["title"]),
        "canal": html.unescape(item["snippet"]["channelTitle"]),
    }


def autofill(df: pd.DataFrame, key: str, max_n: int) -> int:
    """Busca en YouTube las canciones sin enlace; deja estado='auto' para revisarlas."""
    sin_link = df["youtube_url"].str.strip() == ""
    con_artista = df["artista"].str.strip() != ""  # sin artista la búsqueda sale mal y gasta cuota
    pendientes = df.index[sin_link & con_artista].tolist()
    hechas = 0
    for idx in pendientes[:max_n]:
        row = df.loc[idx]
        query = f"{row['artista']} {row['titulo']}".strip()
        try:
            found = youtube_search(query, key)
        except QuotaError:
            print("  Cuota diaria agotada. Vuelve a correr --auto mañana para continuar.")
            break
        hechas += 1
        if found is None:
            print(f"  [sin resultado] {row['archivo']}")
            continue
        df.loc[idx, ["youtube_url", "youtube_titulo", "youtube_canal", "estado"]] = [
            found["url"],
            found["titulo"],
            found["canal"],
            "auto",
        ]
        print(f"  [auto] {row['archivo']} -> {found['titulo']}")
    return hechas


def save(df: pd.DataFrame) -> None:
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig para que Excel muestre bien las tildes y la ñ.
    df.to_csv(META_CSV, index=False, encoding="utf-8-sig")


def resumen(df: pd.DataFrame) -> None:
    con_link = int((df["youtube_url"].str.strip() != "").sum())
    sin_artista = df[df["artista"].str.strip() == ""]
    print(f"\n{len(df)} canciones | {con_link} con enlace | {len(df) - con_link} pendientes")
    if len(sin_artista):
        print(f"{len(sin_artista)} nombres sin formato 'Artista - Título' (completa artista/título a mano):")
        for name in sin_artista["archivo"].tolist():
            print(f"  - {name}")
    print(f"Archivo: {META_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description="Metadatos y enlaces de YouTube por canción")
    p.add_argument("--auto", action="store_true", help="Buscar en YouTube las canciones sin enlace")
    p.add_argument("--max", type=int, default=90, help="Máximo de búsquedas por corrida")
    args = p.parse_args()

    df = build_table()
    try:
        if args.auto:
            key = os.environ.get("YOUTUBE_API_KEY", "").strip()
            if not key:
                print("Falta la variable de entorno YOUTUBE_API_KEY; se crea solo la plantilla.")
            else:
                print(f"Buscando en YouTube (máx. {args.max}) ...")
                autofill(df, key, args.max)
    finally:
        save(df)
    resumen(df)


if __name__ == "__main__":
    main()
