# Mapa sonoro del folklore boliviano

Clustering no supervisado de canciones de 9 géneros del folklore boliviano, con un dashboard en Streamlit,
versiones de modelo (MLOps) y una bitácora exportable.

## Dónde va cada archivo

```
proyecto/
├── app.py                      ← dashboard (NUEVO)
├── requirements.txt            ← librerías (NUEVO)
├── README.md
├── .gitignore                  ← qué NO sube a GitHub (NUEVO)
├── .streamlit/
│   ├── config.toml             ← colores del dashboard (NUEVO)
│   └── secrets.toml.ejemplo    ← plantilla para la clave de YouTube (NUEVO)
├── assets/                     ← IMÁGENES (ver sección 6)
│   ├── portada.jpg
│   ├── logo.png
│   └── generos/<Género>.jpg
├── colab/buscar_links_youtube.ipynb   ← para llenar los enlaces en Colab (NUEVO)
├── src/
│   ├── config.py               ← MODIFICADO (versiones dinámicas, rutas de modelos y bitácora)
│   ├── clip_audio.py           ← MODIFICADO (solo clips completos por versión; musica_prueba/)
│   ├── extract_features.py     ← MODIFICADO (beat_rate en vez de beat_count; duracion_real)
│   ├── cluster_compare.py      ← MODIFICADO (guarda modelos, registra la corrida, HDBSCAN de sklearn)
│   ├── main.py                 ← MODIFICADO (--prueba, --duracion libre, --motivo)
│   ├── predict_new.py          ← NUEVO (guardar modelos y predecir canciones nuevas)
│   ├── mlops.py                ← NUEVO (registro de corridas por versión)
│   ├── bitacora.py             ← NUEVO (bitácora: cambios, corridas, notas)
│   ├── graficos.py             ← NUEVO (gráficos interactivos del dashboard)
│   └── crear_metadata.py       ← NUEVO (CSV de títulos y enlaces de YouTube)
├── musica/<9 géneros>/         ← canciones completas de entrenamiento (como ya lo tienes)
├── musica_prueba/<género>/     ← canciones de prueba, NO usadas para entrenar (NUEVO)
├── data/                       ← features_*.csv, clips, versiones.json, canciones_metadata.csv
├── results/<seg>s/             ← métricas y gráficos de cada versión
├── models/<seg>s/bundle.joblib ← modelos guardados de cada versión
├── models/registro_mlops.csv   ← historial de corridas
└── bitacora/bitacora.csv       ← bitácora permanente
```


## Versiones y MLOps

Una **versión es solo una duración de clip** (v1 = 30 s, v2 = 60 s). Todo lo demás se aplica igual.

- Crear otra (por ejemplo 2 minutos): `python src/main.py --all --duracion 120 --motivo "Probar clips largos"`,
  o desde el dashboard en *Versiones y MLOps → Crear una versión nueva*. Aparece sola en el selector de versión y en *Probar canción*.
  Las canciones que duren menos que la versión no entran en ella.
- Cada entrenamiento guarda en `models/<seg>s/bundle.joblib` el scaler, el PCA y los 6 modelos, y agrega una fila a
  `models/registro_mlops.csv` (fecha, responsable, métricas, parámetros, huella de los datos, versión de scikit-learn, motivo).
  La corrida más reciente de cada versión queda como *produccion* y las anteriores como *archivado*.
- *Versiones y MLOps* muestra gráficos de evolución (ARI, silueta, Davies-Bouldin), si cada corrida mejoró o empeoró,
  el rendimiento de cada algoritmo por corrida y la comparación entre versiones.
- Para ver mejoras: cambia algo (otra versión, otros rasgos, otros parámetros), vuelve a entrenar y mira las gráficas.

