# Mapa sonoro del folklore boliviano

Clustering no supervisado de canciones de 9 géneros del folklore boliviano, con un dashboard en Streamlit,
versiones de modelo (MLOps) y una bitácora exportable.

## 1. Dónde va cada archivo

Descomprime este paquete **encima de tu carpeta del proyecto** (haz una copia de `src/` antes, por si acaso).

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

## 2. Instalación (una sola vez)

```
python -m venv .venv
.venv\Scripts\activate          # en Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`librosa` necesita `ffmpeg` para leer algunos mp3. Si un audio no carga, instálalo (https://ffmpeg.org).

## 3. Orden para correrlo

```
python src/bitacora.py inicial        # una vez: guarda tu nombre y deja registradas las correcciones al código
python src/main.py --all              # recorta, extrae rasgos, entrena los 6 algoritmos de cada versión
streamlit run app.py                  # abre el dashboard
```

- **Vuelve a extraer los rasgos** (`python src/main.py --all --force` la primera vez): el rasgo `beat_count` cambió a
  `beat_rate` y los CSV nuevos guardan `duracion_real` para excluir canciones más cortas que la versión.
- **Canciones de prueba:** guárdalas en `musica_prueba/<Género>/` (mismo nombre de carpeta que en `musica/`) y corre
  `python src/main.py --prueba`. El dashboard dice si cada modelo las ubicó en el género correcto.
- Partes sueltas: `--clips`, `--features`, `--cluster`. Una sola versión: `--duracion 30`.

## 4. Versiones y MLOps

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

## 5. Bitácora

- **En tu computadora** todo queda permanente en `bitacora/bitacora.csv` (fecha, hora, responsable, tipo, sección, archivo, versión, motivo, fragmento).
  - Cada vez que corres `main.py`, si el código cambió te pregunta **por qué** y guarda el fragmento modificado.
  - También puedes correr `python src/bitacora.py cambio` o usar *Bitácora → Cambios detectados en el código* en el dashboard.
  - Cada entrenamiento y cada versión nueva se registran solos.
  - Para notas manuales o una sección nueva: formulario del dashboard o `python src/bitacora.py nota`.
- **En línea** (Streamlit Cloud) las entradas nuevas se ven en pantalla pero el servidor no las guarda: se descarga el CSV/Excel,
  se envía al responsable, y este lo une con *Unir un CSV de bitácora*. Así funcionó el proyecto anterior.
- Descarga en CSV o Excel con los botones de la parte inferior (respetan los filtros).

## 6. Imágenes

Todas opcionales (ver `assets/LEEME.txt`):

| Ruta | Dónde se ve |
|---|---|
| `assets/portada.jpg` | banner de Inicio |
| `assets/logo.png` | barra lateral |
| `assets/generos/Caporales.jpg` (nombre = carpeta del género) | Inicio, Canciones |

## 7. Enlaces de YouTube

1. `python src/crear_metadata.py` crea `data/canciones_metadata.csv` (una fila por canción, con artista y título tomados del nombre del archivo).
2. Llena `youtube_url`: a mano (cada fila trae un enlace de búsqueda), o con el cuaderno de Colab `colab/buscar_links_youtube.ipynb`
   (la API permite unas 100 búsquedas al día; vuelve a correrlo otro día para el resto).
3. Revisa las filas con `estado = auto` y copia el CSV final a `data/canciones_metadata.csv`.
4. Guárdalo como CSV UTF-8 si lo editas en Excel, para no perder tildes ni ñ.

Para el buscador dentro del dashboard (*Probar canción → Buscar en YouTube*), copia `.streamlit/secrets.toml.ejemplo`
como `.streamlit/secrets.toml` y pon tu clave. Sin clave, ese botón abre la búsqueda en YouTube.

## 8. Publicarlo (Streamlit Community Cloud)

1. Sube el proyecto a GitHub. El `.gitignore` deja fuera `musica/`, `musica_prueba/` y los clips (pesados y con derechos de autor);
   **sí** suben `src/`, `app.py`, `data/*.csv`, `data/versiones.json`, `results/`, `models/`, `bitacora/bitacora.csv`, `assets/`.
2. En https://share.streamlit.io → *New app* → tu repositorio, rama principal, archivo `app.py`.
3. En *Advanced settings → Secrets* pega `YOUTUBE_API_KEY = "..."` (opcional).
4. En la nube no hay audio interno (no se subió); las canciones se escuchan por YouTube y se pueden subir audios para clasificarlos
   (si el servidor se queda sin memoria al procesar el audio, esa pestaña falla pero el resto sigue funcionando).
5. Los modelos guardados deben abrirse con la misma versión de scikit-learn con la que se entrenaron: antes de subir, mira tu versión
   (`pip show scikit-learn`) y fíjala en `requirements.txt` (por ejemplo `scikit-learn==1.8.0`).

## 9. Problemas frecuentes

| Síntoma | Solución |
|---|---|
| "Todavía no hay resultados para esta versión" | `python src/main.py --all --duracion <segundos>` |
| "No hay canciones de prueba" | Carpeta `musica_prueba/<Género>/` y `python src/main.py --prueba` |
| Un audio no carga | Instala `ffmpeg` |
| No puede abrir el paquete de modelos | Reentrena con `python src/main.py --cluster` en el mismo entorno donde corres `streamlit` |
| Letras raras (`Ã³`) al abrir un CSV en Excel | Datos > Obtener datos > Desde texto/CSV > UTF-8 |
