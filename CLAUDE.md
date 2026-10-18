# CLAUDE.md — Highlights y estadísticas de un jugador desde video de fútbol

Este archivo es el brief del proyecto para Claude Code. Léelo completo antes de escribir código.

## 1. Contexto del producto

**Problema:** jugadores juveniles y amateur en Latinoamérica graban sus partidos con celular, pero revisar 90 minutos de video para encontrar sus jugadas toma horas. Las soluciones existentes (Veo, BEPRO) dependen de cámaras propias caras o están pensadas para clubes con presupuesto.

**Usuario objetivo (v1):** padres de jugadores juveniles, jugadores amateur y entrenadores de equipos pequeños en Costa Rica.

**Propuesta de valor del MVP:** subo un video de un partido grabado con celular, toco a un jugador y recibo un video con todos sus toques de balón, más un resumen simple de estadísticas.

## 2. Objetivos del MVP

1. Detectar los toques de balón de un jugador seleccionado con **precisión ≥ 80 % y recall ≥ 80 %** frente a una anotación manual (tolerancia ±1 s).
2. Procesar un partido de 90 minutos en **menos de 30 minutos** en una GPU en la nube.
3. Costo de procesamiento por partido **menor a USD 1**.
4. Generar un video de highlights listo para compartir (MP4, 1080p o la resolución original, con el jugador marcado).

## 3. Fuera de alcance (non-goals) en v1

- **Pases, tiros, goles, distancia recorrida y mapas de calor:** vienen después de validar que los toques funcionan.
- **Lectura automática del número de camiseta:** el usuario selecciona al jugador manualmente; es más simple y confiable.
- **App nativa iOS/Android:** v1 es una web app.
- **Procesamiento en tiempo real o livestream.**
- **Red social, perfiles públicos o scouting:** solo videos privados en v1.

## 4. Stack técnico

- **Lenguaje:** Python 3.11+
- **Detección:** Ultralytics YOLO (fine-tuning con dataset de fútbol). Clases: `player`, `goalkeeper`, `referee`, `ball`.
- **Detección del balón en objetos pequeños:** SAHI (inferencia por recortes).
- **Seguimiento:** ByteTrack vía la librería `supervision` de Roboflow.
- **Video:** OpenCV para leer cuadros, FFmpeg para cortar y concatenar.
- **Backend:** FastAPI + cola de trabajos (empezar con algo simple: RQ + Redis).
- **GPU:** servicio de pago por uso (Modal o RunPod). No mantener servidores encendidos.
- **Almacenamiento:** Azure Blob Storage.
- **Frontend v1:** página web simple (HTML + JS o Next.js) para subir video, seleccionar jugador y descargar resultado.
- **Configuración:** todos los umbrales y parámetros en `config.yaml`, nunca hardcodeados.

## 5. Estructura del repositorio

```
footy-highlights/
├── CLAUDE.md
├── README.md
├── config.yaml
├── pyproject.toml
├── src/footy/
│   ├── video_io.py        # lectura de video, muestreo de fps
│   ├── detection.py       # YOLO + SAHI
│   ├── tracking.py        # ByteTrack, selección y re-identificación del jugador
│   ├── ball.py            # filtrado e interpolación de la trayectoria del balón
│   ├── events.py          # detección de toques
│   ├── highlights.py      # corte y unión de clips con FFmpeg, anotaciones
│   ├── stats.py           # resumen de estadísticas
│   └── pipeline.py        # orquesta todo de punta a punta
├── scripts/
│   ├── detect.py
│   ├── track_player.py
│   ├── find_touches.py
│   ├── make_highlights.py
│   └── evaluate.py        # compara contra ground truth
├── api/                   # FastAPI + worker (Fase 5)
├── web/                   # frontend (Fase 5)
├── tests/
├── notebooks/             # exploración, no código de producción
└── data/                  # en .gitignore: videos, modelos, anotaciones
```

## 6. Formatos de datos

**Ground truth de toques** (`data/gt/<video>.csv`):
```
timestamp_s,event
12.4,touch
13.1,touch
```

**Salida de toques** (`outputs/<video>/touches.json`):
```json
{
  "video": "partido1.mp4",
  "player_track_id": 7,
  "fps_processed": 10,
  "touches": [{"timestamp_s": 12.5, "frame": 125, "confidence": 0.82}]
}
```

**Salida de estadísticas** (`outputs/<video>/stats.json`): número de toques, minutos en cámara, toques por tiempo (1.º y 2.º).

## 7. Plan por fases

Trabaja **una fase a la vez**. No avances a la siguiente hasta cumplir los criterios de aceptación y que yo lo confirme.

### Fase 1 — Detección
- Script que corre YOLO sobre un video y guarda un video anotado y un JSON de detecciones por cuadro.
- Procesar a fps configurable (por defecto 10).
- Script de fine-tuning con un dataset de fútbol de Roboflow Universe.
- Probar SAHI solo para la clase `ball` y comparar.

**Aceptación:**
- [ ] `python scripts/detect.py --video data/raw/partido1.mp4` genera `annotated.mp4` y `detections.json`.
- [ ] Reporte de mAP por clase en el set de validación.
- [ ] Comparación de recall del balón con y sin SAHI.

### Fase 2 — Seguimiento del jugador
- Integrar ByteTrack con `supervision`.
- Selección del jugador por CLI: `--frame N --x X --y Y` (simula el toque del usuario); se asigna el track cuya caja contiene el punto.
- Re-identificación cuando se pierde el ID: comparar histograma de color de la camiseta, tamaño y posición esperada. Si la confianza es baja, marcar el intervalo como "jugador no encontrado" en lugar de adivinar.
- Asignación de equipo por clustering (K-means) del color de camiseta.

**Aceptación:**
- [ ] Video anotado mostrando solo al jugador seleccionado con una elipse bajo los pies.
- [ ] Reporte del % del partido en que el jugador fue seguido y número de re-identificaciones.
- [ ] Ninguna re-identificación asigna al jugador del otro equipo.

### Fase 3 — Detección de toques
- Limpiar la trayectoria del balón: descartar saltos imposibles e interpolar huecos cortos (≤ 0.5 s).
- Un toque ocurre cuando el balón está dentro de una distancia de los pies del jugador (centro inferior de su caja), normalizada por la altura de la caja, durante al menos N cuadros seguidos.
- Agrupar detecciones consecutivas en un solo evento (no contar un control largo como 10 toques).
- Script de evaluación contra ground truth con tolerancia ±1 s.

**Aceptación:**
- [ ] `scripts/evaluate.py` reporta precisión, recall y F1.
- [ ] Precisión ≥ 80 % y recall ≥ 80 % en al menos 2 partidos anotados.
- [ ] Umbrales en `config.yaml`, con un script para barrerlos y elegir los mejores.

### Fase 4 — Video de highlights
- Cortar 3 s antes y 3 s después de cada toque (configurable), unir ventanas que se solapen y concatenar con FFmpeg.
- Marcar al jugador con una elipse y mostrar un contador discreto ("Toque 5/23").
- Generar `stats.json`.

**Aceptación:**
- [ ] `highlights.mp4` se reproduce en WhatsApp y en iPhone/Android sin reconvertir (H.264 + AAC).
- [ ] Sin cortes bruscos ni clips duplicados.

### Fase 5 — App mínima
- FastAPI con endpoints: subir video (URL firmada a Azure Blob), crear trabajo, consultar estado, descargar resultado.
- Endpoint que devuelve un cuadro del video para que el usuario toque al jugador.
- Worker en GPU de pago por uso que ejecuta `pipeline.py`.
- Frontend: subir, seleccionar jugador sobre el cuadro, ver progreso, descargar.

**Aceptación:**
- [ ] Flujo completo de punta a punta funcionando con un video real.
- [ ] Medición del tiempo y costo por partido.
- [ ] Manejo de errores: video corrupto, jugador no encontrado, video demasiado largo.

## 8. Privacidad y aspectos legales (obligatorio desde el inicio)

- La mayoría de jugadores serán **menores de edad**. Todo video es **privado por defecto** y solo accesible para quien lo subió.
- Pedir confirmación de consentimiento del padre, madre o tutor al subir un video.
- Cumplir la Ley 8968 de protección de datos personales de Costa Rica.
- Borrar videos originales del almacenamiento después de un plazo configurable (por ejemplo, 30 días).
- No usar videos de usuarios para entrenar modelos sin consentimiento explícito.

## 9. Preguntas abiertas

- **Licencia de Ultralytics (bloqueante antes de lanzar):** YOLO de Ultralytics es AGPL-3.0. Para un producto comercial cerrado se necesita licencia empresarial o usar una alternativa con licencia permisiva (por ejemplo RT-DETR de otra implementación, o YOLOX bajo Apache 2.0). Evaluar en la Fase 1.
- ¿Qué tan alto debe estar el celular para que el sistema funcione? Definir una guía de grabación para usuarios.
- ¿Funciona igual en fútbol 11, fútbol 7 y fútbol sala? Probar en v1 solo fútbol 11.

## 10. Convenciones de trabajo para Claude Code

- Explícame brevemente el plan antes de implementar cada fase.
- Código con type hints, docstrings y funciones pequeñas y probables.
- Tests con `pytest` para la lógica de `events.py`, `ball.py` y `highlights.py` (usar datos sintéticos, no videos reales).
- Usa `logging`, no `print`.
- No subas videos, modelos ni datos al repositorio.
- Fija versiones de dependencias en `pyproject.toml`.
- Si una decisión técnica tiene trade-offs importantes (costo, licencia, precisión), pregúntame antes de elegir.
- Al terminar cada fase, actualiza el README con cómo correrla y los resultados obtenidos.
