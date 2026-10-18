# footy-highlights

Highlights y estadísticas de un jugador a partir de un video de fútbol grabado con celular.
El brief completo del proyecto está en [CLAUDE.md](CLAUDE.md).

| Fase | Estado |
|---|---|
| 1. Detección | Código listo y probado en CPU. Falta correr el fine-tuning y la evaluación en GPU, y probar con un partido real. |
| 2. Seguimiento del jugador | Pendiente |
| 3. Detección de toques | Pendiente |
| 4. Video de highlights | Pendiente |
| 5. App mínima | Pendiente |

## Instalación

Requiere Python 3.11 o superior y FFmpeg.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

- Todos los parámetros están en [`config.yaml`](config.yaml).
- Videos, modelos y datasets van en `data/`, y los resultados en `outputs/`. Las dos carpetas están en `.gitignore`: nunca se suben al repositorio.
- Ultralytics envía estadísticas anónimas de uso. Para desactivarlas: `yolo settings sync=False`.

## Fase 1: detección

### Detectar en un video

```bash
python scripts/detect.py --video data/raw/partido1.mp4
```

Genera `outputs/partido1/annotated.mp4` (H.264, se ve en el celular) y `outputs/partido1/detections.json`.

| Opción | Para qué |
|---|---|
| `--max-seconds 60` | Procesar solo el primer minuto, para probar rápido |
| `--ball-slicing` | Activar SAHI (inferencia por recortes) para el balón |
| `--fps 5` | Cambiar los cuadros por segundo analizados (10 por defecto) |
| `--weights …` / `--device cpu` | Cambiar el modelo o el dispositivo |

Mientras no haya un modelo ajustado, se usa YOLO26m entrenado en COCO: `person` cuenta como `player` y `sports ball` como `ball`. No distingue porteros ni árbitros.

Formato de `detections.json`:

```json
{
  "video": "partido1.mp4",
  "source_fps": 29.97,
  "fps_processed": 10,
  "width": 1920,
  "height": 1080,
  "classes": ["player", "goalkeeper", "referee", "ball"],
  "model": "data/models/footy-detector.pt",
  "imgsz": 1280,
  "confidence": 0.25,
  "ball_slicing": false,
  "frames": [
    {"frame": 0, "timestamp_s": 0.0, "detections": [
      {"class": "player", "confidence": 0.91, "bbox_xyxy": [742.2, 40.1, 1150.8, 708.5]}
    ]}
  ],
  "summary": {"frames_processed": 54000, "ms_per_frame": 25.3, "frames_with_ball": 41210}
}
```

- `frame` es el índice del cuadro en el video original, y `timestamp_s` el segundo real en que aparece. El muestreo usa la marca de tiempo de cada cuadro, así que funciona con los videos de fps variable que graban los celulares.
- `bbox_xyxy` son las esquinas de la caja en píxeles del video original.
- `summary` también trae el tiempo total y las detecciones por clase.
- Si el proceso se interrumpe, queda `detections.json.partial` en lugar de un archivo que parezca completo.

### Fine-tuning (en una máquina con GPU)

```bash
export ROBOFLOW_API_KEY=...     # cuenta gratuita en roboflow.com
python scripts/train_detector.py
```

- Descarga [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc), versión 10. Es la misma versión que usa el notebook oficial de Roboflow, y tiene las clases `ball`, `goalkeeper`, `player` y `referee`.
- Si el export se descargó a mano desde la página del dataset (formato YOLOv8), se puede pasar con `--zip archivo.zip`.
- Los mejores pesos quedan en `data/models/footy-detector.pt`. Después hay que cambiar `detection.weights` en `config.yaml` a esa ruta.

### Evaluación

```bash
python scripts/eval_detector.py --weights data/models/footy-detector.pt
```

Genera `outputs/eval/detector_report.md` y `.json` con:

- el mAP por clase en el set de validación;
- el recall y la precisión del balón con y sin SAHI;
- los milisegundos por imagen de cada modo y el tiempo estimado para un partido de 90 minutos.

### Resultados

**Línea base de referencia (no es nuestra).** YOLOv8x entrenado por Roboflow con este mismo dataset (versión 10, 1280 px). Validación de 43 imágenes, tomada del [notebook oficial](https://github.com/roboflow/sports/blob/main/examples/soccer/notebooks/train_player_detector.ipynb):

| Clase | Precisión | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| ball | 0.949 | 0.481 | 0.599 | 0.327 |
| goalkeeper | 0.918 | 0.875 | 0.948 | 0.794 |
| player | 0.949 | 0.992 | 0.994 | 0.862 |
| referee | 0.943 | 0.931 | 0.977 | 0.710 |
| todas | 0.940 | 0.820 | 0.880 | 0.673 |

El balón es el punto débil: ese modelo encuentra menos de la mitad. Por eso la comparación con SAHI es la medición clave de esta fase.

**Lo verificado hasta ahora**, en CPU y sin GPU (los tiempos en CPU no representan los de producción):

- 49 tests con datos sintéticos pasan (video, detección, dataset, métricas y evaluación).
- Se probó `detect.py` de punta a punta con YOLO26m (COCO) sobre un video de 4 s armado con las imágenes de ejemplo de Ultralytics. Procesó 40 cuadros en 39.5 s (~1 s por cuadro a 1280 px).
- Con YOLO26n, SAHI subió el tiempo de 315 a 536 ms por cuadro.
- `train_detector.py` y `eval_detector.py` funcionan de punta a punta con un dataset sintético de estructura idéntica a la de Roboflow: 2 épocas en CPU, métricas sin valor.

**Pendiente para cerrar la fase:**

- [ ] Correr `detect.py` sobre un partido real (`data/raw/partido1.mp4`).
- [ ] Hacer el fine-tuning y sacar el reporte de mAP por clase en GPU.
- [ ] Comparar el recall del balón con y sin SAHI (sale en el mismo reporte).

## Licencias

YOLO de Ultralytics es AGPL-3.0. Hay que decidir el camino antes de lanzar; ver [docs/licencias.md](docs/licencias.md).

## Estructura

```
config.yaml             parámetros: umbrales, rutas y modelo
src/footy/
  video_io.py           lectura de video, muestreo por tiempo, escritura H.264
  detection.py          YOLO + SAHI para el balón, detections.json y video anotado
  detection_eval.py     recall del balón con y sin SAHI
  dataset.py            descarga del dataset de Roboflow y etiquetas YOLO
  metrics.py            emparejamiento por IoU, precisión y recall
  config.py             carga de config.yaml y logging
scripts/                detect.py, train_detector.py, eval_detector.py
tests/                  pytest con datos sintéticos (nunca videos reales)
docs/licencias.md       evaluación de licencias (§9 del brief)
```
