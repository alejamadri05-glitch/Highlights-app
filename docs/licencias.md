# Licencia del detector y de los datos de entrenamiento

Evaluación de la pregunta abierta del §9 de `CLAUDE.md`, hecha en la Fase 1 (octubre de 2026).

**Estado:** sin decidir. Bloquea el lanzamiento (Fase 5), no el desarrollo de las fases 1 a 4.
Es una evaluación técnica, no asesoría legal: antes de lanzar conviene que la revise un abogado.

## 1. Ultralytics YOLO

- El paquete `ultralytics` (y los pesos YOLO11 y YOLO26) tiene licencia **AGPL-3.0**, con la alternativa de una **licencia Enterprise** de pago ([licencias de Ultralytics](https://ultralytics.com/license)).
- La AGPL-3.0 obliga a ofrecer el código fuente completo a cualquier persona que use el servicio por internet. Para nuestra web app, eso significa publicar el código de todo el producto.
- Ultralytics interpreta que su licencia también cubre los **modelos entrenados o ajustados** con su software, aunque se entrene desde cero o se use solo internamente. Según su página, hay que publicar todo el proyecto bajo AGPL-3.0 o comprar la licencia Enterprise ([solicitud de licencia](https://www.ultralytics.com/request-license)).
- El precio de la licencia Enterprise no es público; se cotiza por despliegue.

## 2. Alternativas con licencia permisiva

| Modelo | Licencia | Comentario |
|---|---|---|
| [RF-DETR](https://rfdetr.roboflow.com/latest/learn/run/detection/) Nano a Large (Roboflow) | Apache 2.0 (paquete `rfdetr`) | El más práctico: moderno, se instala con pip y es del mismo ecosistema que `supervision`. Los tamaños XLarge y 2XLarge tienen otra licencia (Platform Model License) y quedan descartados. |
| D-FINE | Apache 2.0 | Muy preciso; es un repositorio de investigación, más trabajo para entrenar y desplegar. |
| RT-DETR / RT-DETRv2 (implementación original) | Apache 2.0 | La versión de RT-DETR que viene dentro de `ultralytics` es AGPL; habría que usar la original. |
| YOLOX (Megvii) | Apache 2.0 | De 2021, menos preciso que los anteriores. |

El resto del stack ya es permisivo: `supervision` (MIT; incluye ByteTrack y la inferencia por recortes que usamos para SAHI) y OpenCV (Apache 2.0).

## 3. Riesgo en los datos de entrenamiento

El dataset de Roboflow de la Fase 1 sale de los videos del concurso [DFL – Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) de Kaggle, según el [README de roboflow/sports](https://github.com/roboflow/sports/tree/main/examples/soccer). Las reglas de los concursos de Kaggle suelen limitar el uso de los datos al propio concurso. Mientras no revisemos esas reglas, ese dataset sirve para validar la técnica, pero **no conviene que el modelo del producto se entrene con él**.

De todas formas hace falta un dataset propio: las transmisiones de la Bundesliga se ven muy distinto a un partido amateur grabado con celular desde la gradería. Al bajar el dataset, `train_detector.py` muestra la licencia que viene en su `data.yaml`.

## 4. Recomendación

1. **Fases 1 a 3: seguir con Ultralytics**, como indica el brief. Según el texto de la AGPL, el uso interno sin usuarios externos no obliga a publicar código, aunque Ultralytics interpreta su licencia de forma más amplia (sección 1).
2. **Mantener el detector detrás de la interfaz `FrameDetector`** (`src/footy/detection.py`), como ya está, para que cambiar de modelo toque un solo archivo.
3. **Comparar con RF-DETR cuando exista el dataset propio:** entrenarlo con los mismos datos y medirlo con `scripts/eval_detector.py`. Si queda cerca de YOLO en mAP, recall del balón y velocidad, cambiar antes de la Fase 5 y evitar la licencia.
4. **Pedir cotización a Ultralytics en paralelo**, para comparar con el costo de cambiar de modelo.
