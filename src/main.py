"""Pipeline: recorte -> features -> clustering de folklor boliviano, por versión.

Una versión es una duración de clip (v1 = 30 s, v2 = 60 s, ...).

Uso (desde la raíz del proyecto):
  python src/main.py --all                  # clips + features + clustering, todas las versiones
  python src/main.py --clips
  python src/main.py --features
  python src/main.py --cluster
  python src/main.py --reduccion            # laboratorio: PCA, PPCA, SVD iterativa, t-SNE, UMAP x 6 algoritmos
  python src/main.py --diagnostico          # comparación con el azar, familias de rasgos, estabilidad, artistas
  python src/main.py --prueba               # solo las canciones de musica_prueba/
  python src/main.py --all --duracion 120   # crea (si no existe) y entrena la versión de 2 minutos
  python src/main.py --all --motivo "Probar clips más largos"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bitacora
from clip_audio import run_clips
from cluster_compare import run_cluster
from config import MUSICA_PRUEBA_DIR, ensure_dirs, load_versions, register_version
from diagnostico import run_diagnostico
from extract_features import run_features
from reduccion import run_reduccion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clustering de música folklórica boliviana")
    p.add_argument("--clips", action="store_true", help="Recortar audios según cada versión")
    p.add_argument("--features", action="store_true", help="Extraer features de los clips")
    p.add_argument("--cluster", action="store_true", help="Entrenar modelos, métricas y guardar la corrida")
    p.add_argument("--reduccion", action="store_true", help="Laboratorio de reducción de dimensionalidad x algoritmos")
    p.add_argument("--diagnostico", action="store_true", help="Diagnósticos de calidad del agrupamiento")
    p.add_argument("--all", action="store_true", help="Ejecutar todo: clips, features, cluster, reducción y diagnóstico")
    p.add_argument("--prueba", action="store_true", help="Procesar también (o solo) musica_prueba/")
    p.add_argument("--force", action="store_true", help="Rehacer clips aunque ya existan")
    p.add_argument("--show", action="store_true", help="Mostrar figuras además de guardar PNG")
    p.add_argument("--duracion", type=int, default=None,
                   help="Segundos del clip. Si no es una versión existente, se crea una nueva.")
    p.add_argument("--motivo", default="", help="Por qué corres esto (queda en la bitácora)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    bitacora.revisar_cambios_interactivo()

    if args.duracion and args.duracion not in load_versions().values():
        nombre = register_version(args.duracion)
        print(f"Nueva versión creada: {nombre} ({args.duracion} s)")
        bitacora.agregar("version", f"Se crea la versión {nombre} con clips de {args.duracion} s. {args.motivo}".strip(),
                         seccion="Versiones", version=nombre, responsable=bitacora.responsable_actual(preguntar=True))
        ensure_dirs()

    pasos = {"clips": args.clips, "features": args.features, "cluster": args.cluster,
             "reduccion": args.reduccion, "diagnostico": args.diagnostico}
    solo_prueba = args.prueba and not (args.all or any(pasos.values()))
    if solo_prueba:
        pasos.update(clips=True, features=True)
    elif args.all or not any(pasos.values()):
        pasos.update({k: True for k in pasos})
    con_train = not solo_prueba
    sin_banderas = not (args.all or args.prueba or args.clips or args.features or args.cluster or args.reduccion or args.diagnostico)
    con_prueba = (args.prueba or args.all or sin_banderas) and MUSICA_PRUEBA_DIR.exists()

    motivo = args.motivo.strip()
    if not motivo and pasos["cluster"] and sys.stdin.isatty():
        motivo = input("¿Por qué corres el entrenamiento? (Enter = corrida rutinaria) ").strip()
    os.environ["MOTIVO_CORRIDA"] = motivo

    if pasos["clips"]:
        if con_train:
            run_clips(force=args.force, split="train", seconds=args.duracion)
        if con_prueba:
            run_clips(force=args.force, split="prueba", seconds=args.duracion)
    if pasos["features"]:
        if con_train:
            run_features(seconds=args.duracion, split="train")
        if con_prueba:
            run_features(seconds=args.duracion, split="prueba")
    if pasos["cluster"]:
        run_cluster(seconds=args.duracion, show=args.show)
    if pasos["reduccion"]:
        run_reduccion(seconds=args.duracion)
        bitacora.agregar("corrida", f"Laboratorio de reducción x algoritmos. {motivo}".strip(), seccion="Reducción",
                         responsable=bitacora.responsable_actual())
    if pasos["diagnostico"]:
        run_diagnostico(seconds=args.duracion)


if __name__ == "__main__":
    main()
