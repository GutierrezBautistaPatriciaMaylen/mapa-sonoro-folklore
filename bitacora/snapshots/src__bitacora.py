"""Bitácora del proyecto: cambios de código, corridas, versiones y notas (CSV exportable).

Uso por consola (desde la raíz del proyecto):
  python src/bitacora.py inicial   # una sola vez: guarda tu nombre y la línea base del código
  python src/bitacora.py cambio    # detecta qué código cambió, pregunta por qué y lo guarda
  python src/bitacora.py nota      # agrega una nota o decisión
  python src/bitacora.py ver       # muestra las últimas entradas
"""

from __future__ import annotations

import difflib
import getpass
import io
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import BITACORA_DIR, ROOT

BITACORA_CSV = BITACORA_DIR / "bitacora.csv"
SNAP_DIR = BITACORA_DIR / "snapshots"
RESP_FILE = BITACORA_DIR / "responsable.txt"

COLUMNAS = ["id", "fecha", "hora", "responsable", "tipo", "seccion", "archivo", "version", "motivo", "fragmento"]
TIPOS = ["cambio", "corrida", "version", "prueba", "nota", "decision", "linea_base"]
SECCIONES_BASE = ["General", "Datos", "Features", "Algoritmos", "Métricas", "MLOps", "Dashboard", "Versiones"]
MAX_FRAGMENTO = 3500

# Correcciones aplicadas sobre el código base; `python src/bitacora.py inicial` las registra
# con tu nombre y la fecha en que las incorporas.
CAMBIOS_INICIALES = [
    {
        "archivo": "src/extract_features.py",
        "seccion": "Features",
        "motivo": "beat_count dependía de la duración del clip (60 s siempre tiene más beats que 30 s); se cambia a beats por segundo para comparar versiones.",
        "fragmento": '- feats["beat_count"] = float(len(beats))\n+ feats["beat_rate"] = float(len(beats)) / duracion_s',
    },
    {
        "archivo": "src/extract_features.py",
        "seccion": "Features",
        "motivo": "beat_track recalculaba el audio completo y un tempo que no se usaba; ahora reutiliza la envolvente de onsets.",
        "fragmento": "- tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)\n+ _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)",
    },
    {
        "archivo": "src/clip_audio.py",
        "seccion": "Datos",
        "motivo": "Una canción de 25 s generaba un 'clip de 60 s' de 25 s; ahora cada versión solo usa canciones que alcanzan su duración.",
        "fragmento": "- if duration < MIN_DURATION_SEC:\n+ if duration < seconds * MIN_FULL_RATIO:   # por versión",
    },
    {
        "archivo": "src/cluster_compare.py",
        "seccion": "Métricas",
        "motivo": "La inercia siempre baja al subir k, así que el score combinado de KMeans favorecía k grandes; se quita del score (sigue en la curva del codo).",
        "fragmento": '- sweep["score_combinado"] = (inv_in + sil_n + ch_n) / 3.0\n+ sweep["score_combinado"] = (sil_n + ch_n) / 2.0',
    },
    {
        "archivo": "src/cluster_compare.py",
        "seccion": "Algoritmos",
        "motivo": "La librería externa hdbscan falla al instalarse en la nube; se usa HDBSCAN de scikit-learn.",
        "fragmento": "- import hdbscan as hdbscan_lib\n+ from sklearn.cluster import HDBSCAN as SklearnHDBSCAN",
    },
    {
        "archivo": "src/cluster_compare.py",
        "seccion": "MLOps",
        "motivo": "Los modelos no se guardaban, así que no se podía predecir una canción nueva. Ahora cada versión guarda scaler, PCA y los 6 modelos, y cada corrida queda en el registro.",
        "fragmento": "+ guardar_bundle(seconds, df, feat_cols, scaler, pca_full, info, X, X_den, genres, fitted)\n+ run_id = registrar_corrida(seconds, resumen, ranked, ...)",
    },
    {
        "archivo": "src/config.py",
        "seccion": "Versiones",
        "motivo": "Las versiones eran fijas (30 y 60 s). Ahora viven en data/versiones.json y se puede crear otra (por ejemplo 120 s) sin tocar el código.",
        "fragmento": "- CLIP_SECONDS = (30, 60)\n+ def clip_seconds(): return tuple(load_versions().values())",
    },
]


# ------------------------------------------------------------------ utilidades básicas
def responsable_actual(preguntar: bool = False) -> str:
    """Nombre del responsable: variable BITACORA_USER > bitacora/responsable.txt > usuario del sistema."""
    env = os.environ.get("BITACORA_USER", "").strip()
    if env:
        return env
    if RESP_FILE.exists():
        txt = RESP_FILE.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    if preguntar and sys.stdin.isatty():
        nombre = input("¿Cuál es tu nombre (responsable de los cambios)? ").strip()
        if nombre:
            BITACORA_DIR.mkdir(parents=True, exist_ok=True)
            RESP_FILE.write_text(nombre, encoding="utf-8")
            return nombre
    return getpass.getuser()


def guardar_responsable(nombre: str) -> None:
    BITACORA_DIR.mkdir(parents=True, exist_ok=True)
    RESP_FILE.write_text(nombre.strip(), encoding="utf-8")


def nueva_fila(
    tipo: str,
    motivo: str,
    seccion: str = "General",
    archivo: str = "",
    version: str = "",
    fragmento: str = "",
    responsable: str | None = None,
) -> dict:
    ahora = datetime.now()
    return {
        "id": f"{ahora:%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}",
        "fecha": f"{ahora:%Y-%m-%d}",
        "hora": f"{ahora:%H:%M:%S}",
        "responsable": responsable or responsable_actual(),
        "tipo": tipo,
        "seccion": seccion or "General",
        "archivo": archivo,
        "version": version,
        "motivo": motivo.strip(),
        "fragmento": fragmento[:MAX_FRAGMENTO],
    }


def leer(path: Path | None = None) -> pd.DataFrame:
    path = path or BITACORA_CSV
    if not path.exists():
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    for col in COLUMNAS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNAS]


def unir(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Une dos bitácoras sin duplicar entradas (por id) y ordena de la más reciente a la más antigua."""
    df = pd.concat([a, b], ignore_index=True).drop_duplicates(subset="id", keep="first")
    return df.sort_values(["fecha", "hora"], ascending=False).reset_index(drop=True)[COLUMNAS]


def guardar_df(df: pd.DataFrame) -> None:
    BITACORA_DIR.mkdir(parents=True, exist_ok=True)
    df[COLUMNAS].to_csv(BITACORA_CSV, index=False, encoding="utf-8-sig")


def agregar(tipo: str, motivo: str, **kwargs) -> dict:
    """Agrega una entrada permanente al CSV de la bitácora."""
    fila = nueva_fila(tipo, motivo, **kwargs)
    guardar_df(unir(leer(), pd.DataFrame([fila])))
    return fila


def a_csv_bytes(df: pd.DataFrame) -> bytes:
    return df[COLUMNAS].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def a_excel_bytes(df: pd.DataFrame) -> bytes:
    from openpyxl.styles import Alignment, Font

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df[COLUMNAS].to_excel(writer, index=False, sheet_name="Bitacora")
        ws = writer.sheets["Bitacora"]
        anchos = {"id": 22, "fecha": 12, "hora": 10, "responsable": 16, "tipo": 12, "seccion": 14,
                  "archivo": 24, "version": 9, "motivo": 60, "fragmento": 70}
        for idx, col in enumerate(COLUMNAS, start=1):
            letra = ws.cell(row=1, column=idx).column_letter
            ws.column_dimensions[letra].width = anchos[col]
            ws.cell(row=1, column=idx).font = Font(bold=True)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
    return buf.getvalue()


# ------------------------------------------------------------------ detección de cambios en el código
def archivos_seguidos() -> list[Path]:
    files = sorted((ROOT / "src").glob("*.py"))
    app = ROOT / "app.py"
    if app.exists():
        files.append(app)
    return [f for f in files if f.name != "__init__.py"]


def _snap(path: Path) -> Path:
    return SNAP_DIR / path.relative_to(ROOT).as_posix().replace("/", "__")


def _leer_texto(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def hay_linea_base() -> bool:
    return SNAP_DIR.exists() and any(SNAP_DIR.iterdir())


def crear_snapshot(archivos: list[Path] | None = None) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    for f in archivos or archivos_seguidos():
        _snap(f).write_text(_leer_texto(f), encoding="utf-8")


def detectar_cambios() -> list[dict]:
    """Compara el código actual contra la última línea base y devuelve los fragmentos cambiados."""
    if not hay_linea_base():
        return []
    cambios = []
    for f in archivos_seguidos():
        snap = _snap(f)
        nuevo = _leer_texto(f)
        viejo = _leer_texto(snap) if snap.exists() else ""
        if nuevo == viejo:
            continue
        diff = list(difflib.unified_diff(viejo.splitlines(), nuevo.splitlines(), "antes", "despues", n=2, lineterm=""))
        cambios.append(
            {
                "archivo": f.relative_to(ROOT).as_posix(),
                "ruta": f,
                "diff": "\n".join(diff)[:MAX_FRAGMENTO],
                "agregadas": sum(1 for l in diff if l.startswith("+") and not l.startswith("+++")),
                "quitadas": sum(1 for l in diff if l.startswith("-") and not l.startswith("---")),
            }
        )
    return cambios


def registrar_cambios(cambios: list[dict], motivo: str, responsable: str | None = None, seccion: str = "General") -> int:
    """Guarda cada cambio con su fragmento y actualiza la línea base."""
    for c in cambios:
        agregar("cambio", motivo, seccion=seccion, archivo=c["archivo"], fragmento=c["diff"], responsable=responsable)
    crear_snapshot([c["ruta"] for c in cambios])
    return len(cambios)


def revisar_cambios_interactivo() -> None:
    """Se llama al correr el pipeline: si el código cambió, pregunta por qué y lo registra."""
    if not hay_linea_base():
        return
    cambios = detectar_cambios()
    if not cambios:
        return
    print("\nSe detectaron cambios en el código desde la última entrada de la bitácora:")
    for c in cambios:
        print(f"  - {c['archivo']}  (+{c['agregadas']} / -{c['quitadas']} líneas)")
    if not sys.stdin.isatty():
        print("  (sin terminal interactiva: usa `python src/bitacora.py cambio` para registrarlos)")
        return
    motivo = input("¿Por qué hiciste estos cambios? (Enter para omitir) ").strip()
    if not motivo:
        print("  Omitido; seguirán apareciendo hasta que los registres.")
        return
    seccion = input(f"Sección [{', '.join(SECCIONES_BASE[:4])}...] (Enter = General): ").strip() or "General"
    n = registrar_cambios(cambios, motivo, responsable_actual(preguntar=True), seccion)
    print(f"  {n} cambio(s) guardados en {BITACORA_CSV}")


# ------------------------------------------------------------------ consola
def _inicial() -> None:
    nombre = os.environ.get("BITACORA_USER", "").strip() or input("Tu nombre (responsable): ").strip() or getpass.getuser()
    guardar_responsable(nombre)
    ya = leer()
    if not ya.empty and (ya["tipo"] == "linea_base").any():
        print("La línea base ya existe; no se repiten las entradas iniciales.")
        return
    agregar("linea_base", "Estado del código al activar la bitácora (incluye las correcciones listadas a continuación).",
            seccion="General", responsable=nombre)
    for c in CAMBIOS_INICIALES:
        agregar("cambio", c["motivo"], seccion=c["seccion"], archivo=c["archivo"], fragmento=c["fragmento"], responsable=nombre)
    crear_snapshot()
    print(f"Bitácora creada en {BITACORA_CSV} con {len(CAMBIOS_INICIALES) + 1} entradas.")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ver"
    if cmd == "inicial":
        _inicial()
    elif cmd == "cambio":
        if not hay_linea_base():
            print("Primero corre: python src/bitacora.py inicial")
            return
        if not detectar_cambios():
            print("No hay cambios en el código desde la última entrada.")
            return
        revisar_cambios_interactivo()
    elif cmd == "nota":
        motivo = input("Nota o decisión: ").strip()
        seccion = input("Sección (Enter = General): ").strip() or "General"
        if motivo:
            agregar("nota", motivo, seccion=seccion, responsable=responsable_actual(preguntar=True))
            print("Guardada.")
    else:
        df = leer()
        print(df.head(15)[["fecha", "hora", "responsable", "tipo", "seccion", "archivo", "motivo"]].to_string(index=False)
              if not df.empty else "La bitácora está vacía.")


if __name__ == "__main__":
    main()
