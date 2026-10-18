"""Detección de jugadores, porteros, árbitros y balón con YOLO, con SAHI opcional para el balón."""

import itertools
import json
import logging
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, TypeVar

import numpy as np
import supervision as sv

from footy.video_io import (
    EncodeConfig,
    Frame,
    VideoReadError,
    VideoWriter,
    iter_frames,
    probe_video,
)

logger = logging.getLogger(__name__)

BALL_CLASS = "ball"
_PROGRESS_LOG_INTERVAL_S = 30.0

T = TypeVar("T")


@dataclass(frozen=True)
class BallSlicingConfig:
    """Inferencia por recortes (SAHI) solo para la clase ``ball``."""

    enabled: bool
    slice_wh: tuple[int, int]
    overlap_wh: tuple[int, int]
    merge_iou: float
    batch_size: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Construye la configuración desde ``detection.ball_slicing`` de config.yaml."""
        return cls(
            enabled=bool(data["enabled"]),
            slice_wh=_pair(data["slice_wh"]),
            overlap_wh=_pair(data["overlap_wh"]),
            merge_iou=float(data["merge_iou"]),
            batch_size=int(data["batch_size"]),
        )


@dataclass(frozen=True)
class DetectorConfig:
    """Sección ``detection`` de config.yaml."""

    weights: str
    device: str | None
    imgsz: int
    confidence: float
    nms_iou: float
    batch_size: int
    classes: tuple[str, ...]
    class_map: Mapping[str, str]
    ball_slicing: BallSlicingConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Construye la configuración desde la sección ``detection`` de config.yaml."""
        device = data["device"]
        return cls(
            weights=str(data["weights"]),
            device=None if device is None else str(device),
            imgsz=int(data["imgsz"]),
            confidence=float(data["confidence"]),
            nms_iou=float(data["nms_iou"]),
            batch_size=int(data["batch_size"]),
            classes=tuple(data["classes"]),
            class_map=dict(data["class_map"] or {}),
            ball_slicing=BallSlicingConfig.from_dict(data["ball_slicing"]),
        )


def _pair(values: Sequence[int]) -> tuple[int, int]:
    width, height = values
    return int(width), int(height)


def build_class_mapping(
    model_names: Mapping[int, str], classes: Sequence[str], class_map: Mapping[str, str]
) -> dict[int, int]:
    """Relaciona cada clase del modelo con el índice de su clase en ``classes``.

    Las clases del modelo sin equivalente (por ejemplo "car" en COCO) quedan fuera del
    resultado, así que se descartan.
    """
    mapping: dict[int, int] = {}
    for model_id, name in model_names.items():
        project_name = class_map.get(name, name)
        if project_name in classes:
            mapping[int(model_id)] = classes.index(project_name)
    return mapping


def remap_classes(detections: sv.Detections, mapping: Mapping[int, int]) -> sv.Detections:
    """Traduce los ``class_id`` del modelo a los del proyecto y descarta los que sobran.

    También borra ``data["class_name"]`` (que trae los nombres del modelo): los nombres
    se obtienen siempre de ``class_id``, y así todas las detecciones tienen las mismas
    llaves al unirlas con ``sv.Detections.merge``.
    """
    keep = np.array([int(class_id) in mapping for class_id in detections.class_id], dtype=bool)
    remapped = detections[keep]
    remapped.class_id = np.array([mapping[int(c)] for c in remapped.class_id], dtype=int)
    remapped.data = {}
    return remapped


def merge_ball_detections(
    detections: sv.Detections, sliced_balls: sv.Detections, ball_id: int, iou_threshold: float
) -> sv.Detections:
    """Agrega los balones encontrados por recortes y elimina los repetidos con NMS."""
    is_ball = detections.class_id == ball_id
    balls = sv.Detections.merge([detections[is_ball], sliced_balls])
    if len(balls) > 1:
        balls = balls.with_nms(threshold=iou_threshold)
    return sv.Detections.merge([detections[~is_ball], balls])


class FrameDetector(Protocol):
    """Lo que el resto del pipeline necesita de un detector.

    Mantener esta interfaz permite cambiar YOLO de Ultralytics por otro modelo (por la
    licencia AGPL-3.0) sin tocar video, tracking ni eventos.
    """

    @property
    def classes(self) -> tuple[str, ...]:
        """Nombres de clase; ``class_id`` de las detecciones indexa esta tupla."""
        ...

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[sv.Detections]:
        """Detecta objetos en imágenes BGR; devuelve una ``sv.Detections`` por imagen."""
        ...


class YoloDetector:
    """Detector YOLO de Ultralytics con SAHI opcional para el balón."""

    def __init__(self, config: DetectorConfig, model: Any | None = None) -> None:
        """Carga el modelo y relaciona sus clases con las del proyecto.

        Args:
            config: sección ``detection`` de config.yaml.
            model: modelo YOLO ya cargado (para compartirlo entre detectores). Si es
                ``None`` se carga ``config.weights``.

        Raises:
            ValueError: si el modelo no detecta ninguna clase del proyecto, o si se pide
                SAHI y el modelo no detecta balones.
        """
        if model is None:
            from ultralytics import YOLO  # importación diferida: es pesada

            model = YOLO(config.weights)
        self._model = model
        self._config = config
        self._mapping = build_class_mapping(model.names, config.classes, config.class_map)
        if not self._mapping:
            raise ValueError(
                f"El modelo {config.weights} no detecta ninguna de las clases {config.classes}"
            )
        detected = {config.classes[i] for i in self._mapping.values()}
        missing = [name for name in config.classes if name not in detected]
        if missing:
            logger.warning("El modelo %s no detecta: %s", config.weights, ", ".join(missing))
        self._slicer = self._build_ball_slicer() if config.ball_slicing.enabled else None

    @property
    def classes(self) -> tuple[str, ...]:
        """Nombres de clase del proyecto, en el orden de config.yaml."""
        return self._config.classes

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[sv.Detections]:
        """Detecta en el cuadro completo y, si SAHI está activo, busca el balón por recortes."""
        detections = self._predict(images, self._config.imgsz, list(self._mapping))
        if self._slicer is None:
            return detections
        ball_id = self.classes.index(BALL_CLASS)
        merge_iou = self._config.ball_slicing.merge_iou
        return [
            merge_ball_detections(frame_detections, self._slicer(image), ball_id, merge_iou)
            for frame_detections, image in zip(detections, images, strict=True)
        ]

    def _predict(
        self, images: Sequence[np.ndarray], imgsz: int, model_class_ids: list[int]
    ) -> list[sv.Detections]:
        raw = self._run_model(images, imgsz, model_class_ids)
        return [remap_classes(detections, self._mapping) for detections in raw]

    def _run_model(
        self, images: Sequence[np.ndarray], imgsz: int, model_class_ids: list[int]
    ) -> list[sv.Detections]:
        """Única llamada al modelo; devuelve detecciones con los ``class_id`` del modelo."""
        results = self._model.predict(
            list(images),
            imgsz=imgsz,
            conf=self._config.confidence,
            iou=self._config.nms_iou,
            classes=model_class_ids,
            device=self._config.device,
            verbose=False,
        )
        return [sv.Detections.from_ultralytics(result) for result in results]

    def _build_ball_slicer(self) -> sv.InferenceSlicer:
        if BALL_CLASS not in self.classes:
            raise ValueError("ball_slicing requiere la clase 'ball' en detection.classes")
        ball_id = self.classes.index(BALL_CLASS)
        ball_model_ids = [model_id for model_id, c in self._mapping.items() if c == ball_id]
        if not ball_model_ids:
            raise ValueError(f"El modelo {self._config.weights} no detecta balones: sin SAHI")
        slicing = self._config.ball_slicing
        slice_imgsz = max(slicing.slice_wh)

        def predict_slices(
            slices: np.ndarray | list[np.ndarray],
        ) -> sv.Detections | list[sv.Detections]:
            # Con batch_size=1 InferenceSlicer pasa un recorte; con más, una lista.
            if isinstance(slices, np.ndarray):
                return self._predict([slices], slice_imgsz, ball_model_ids)[0]
            return self._predict(slices, slice_imgsz, ball_model_ids)

        return sv.InferenceSlicer(
            callback=predict_slices,
            slice_wh=slicing.slice_wh,
            overlap_wh=slicing.overlap_wh,
            iou_threshold=slicing.merge_iou,
            batch_size=slicing.batch_size,
        )


def detections_to_records(
    detections: sv.Detections, classes: Sequence[str]
) -> list[dict[str, Any]]:
    """Convierte detecciones a diccionarios para JSON (coordenadas en píxeles del cuadro)."""
    return [
        {
            "class": classes[int(class_id)],
            "confidence": round(float(confidence), 3),
            "bbox_xyxy": [round(float(value), 1) for value in box],
        }
        for box, confidence, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id, strict=True
        )
    ]


class DetectionsJsonWriter:
    """Escribe detections.json cuadro por cuadro, sin guardar todo el partido en memoria.

    El JSON tiene los campos de ``header``, luego ``frames`` (uno por cuadro procesado) y
    al final ``summary``. Se escribe primero en ``<nombre>.partial`` y solo se renombra
    al llamar a :meth:`finish`, así un proceso interrumpido no deja un archivo que
    parezca completo.
    """

    def __init__(self, path: Path, header: Mapping[str, Any], classes: Sequence[str]) -> None:
        if "frames" in header or "summary" in header:
            raise ValueError("header no puede usar las llaves 'frames' ni 'summary'")
        self.path = path
        self._partial_path = path.with_name(path.name + ".partial")
        self._classes = classes
        self._frames_written = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._partial_path.open("w", encoding="utf-8")
        fields = [
            f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in header.items()
        ]
        self._file.write("{" + ", ".join([*fields, '"frames": [']) + "\n")

    def write_frame(self, frame: Frame, detections: sv.Detections) -> None:
        """Agrega las detecciones de un cuadro."""
        record = {
            "frame": frame.index,
            "timestamp_s": round(frame.timestamp_s, 3),
            "detections": detections_to_records(detections, self._classes),
        }
        separator = ",\n" if self._frames_written else ""
        self._file.write(separator + json.dumps(record, ensure_ascii=False))
        self._frames_written += 1

    def finish(self, summary: Mapping[str, Any]) -> None:
        """Cierra el JSON con el resumen y lo deja en su nombre final."""
        self._file.write("\n], " + f'"summary": {json.dumps(summary, ensure_ascii=False)}' + "}\n")
        self._file.close()
        self._partial_path.replace(self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._file.closed:
            self._file.close()
            logger.warning("Detección incompleta; resultados parciales en %s", self._partial_path)


class DetectionAnnotator:
    """Dibuja cajas y etiquetas (clase y confianza) con un color por clase."""

    def __init__(self, classes: Sequence[str], resolution_wh: tuple[int, int]) -> None:
        thickness = sv.calculate_optimal_line_thickness(resolution_wh)
        self._classes = classes
        self._boxes = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.CLASS)
        self._labels = sv.LabelAnnotator(
            text_scale=sv.calculate_optimal_text_scale(resolution_wh),
            text_thickness=thickness,
            color_lookup=sv.ColorLookup.CLASS,
        )

    def annotate(self, image: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Devuelve una copia de ``image`` con las detecciones dibujadas."""
        labels = [
            f"{self._classes[int(class_id)]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence, strict=True)
        ]
        annotated = self._boxes.annotate(image.copy(), detections)
        return self._labels.annotate(annotated, detections, labels=labels)


@dataclass(frozen=True)
class DetectionSummary:
    """Resumen de una corrida de detección sobre un video."""

    frames_processed: int
    video_seconds_processed: float
    processing_time_s: float
    ms_per_frame: float
    detections_per_class: dict[str, int]
    frames_with_ball: int  # cuadros con al menos un balón detectado


def _has_class(detections: sv.Detections, class_id: int | None) -> bool:
    return class_id is not None and bool(np.any(detections.class_id == class_id))


def _batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_detection(
    video_path: Path,
    detector: FrameDetector,
    output_dir: Path,
    processing_fps: float,
    batch_size: int,
    encode: EncodeConfig,
    max_seconds: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DetectionSummary:
    """Corre el detector sobre el video y guarda ``annotated.mp4`` y ``detections.json``.

    Args:
        video_path: video de entrada.
        detector: cualquier objeto que cumpla :class:`FrameDetector`.
        output_dir: carpeta de salida; se crea si no existe.
        processing_fps: cuadros por segundo a analizar.
        batch_size: cuadros por llamada al detector.
        encode: parámetros de H.264 del video anotado.
        max_seconds: si se indica, procesa solo el inicio del video.
        metadata: campos extra para el encabezado del JSON (modelo, umbrales...).

    Raises:
        VideoReadError: si el video no se puede abrir o no tiene cuadros legibles.
    """
    info = probe_video(video_path)
    frames = iter_frames(video_path, processing_fps, max_seconds)
    first = next(frames, None)
    if first is None:
        raise VideoReadError(f"No se pudo leer ningún cuadro de {video_path}")
    height, width = first.image.shape[:2]
    header = {
        "video": video_path.name,
        "source_fps": round(info.fps, 3),
        "fps_processed": processing_fps,
        "width": width,
        "height": height,
        "classes": list(detector.classes),
        **(metadata or {}),
    }
    expected_s = min(info.duration_s, max_seconds or info.duration_s)
    ball_id = detector.classes.index(BALL_CLASS) if BALL_CLASS in detector.classes else None
    annotator = DetectionAnnotator(detector.classes, (width, height))
    counts: Counter[str] = Counter()
    frames_with_ball = 0
    last_timestamp_s = 0.0
    start = last_log = time.perf_counter()

    with (
        VideoWriter(output_dir / "annotated.mp4", min(processing_fps, info.fps), encode) as video,
        DetectionsJsonWriter(output_dir / "detections.json", header, detector.classes) as out,
    ):
        for batch in _batched(itertools.chain([first], frames), batch_size):
            results = detector.detect_batch([frame.image for frame in batch])
            for frame, detections in zip(batch, results, strict=True):
                out.write_frame(frame, detections)
                video.write(annotator.annotate(frame.image, detections))
                counts.update(detector.classes[int(c)] for c in detections.class_id)
                frames_with_ball += _has_class(detections, ball_id)
                last_timestamp_s = frame.timestamp_s
            if time.perf_counter() - last_log >= _PROGRESS_LOG_INTERVAL_S:
                last_log = time.perf_counter()
                logger.info("Procesados %.0f s de %.0f s de video", last_timestamp_s, expected_s)

        elapsed_s = time.perf_counter() - start
        summary = DetectionSummary(
            frames_processed=video.frames_written,
            video_seconds_processed=round(last_timestamp_s, 3),
            processing_time_s=round(elapsed_s, 2),
            ms_per_frame=round(1000 * elapsed_s / video.frames_written, 1),
            detections_per_class=dict(counts),
            frames_with_ball=frames_with_ball,
        )
        out.finish(asdict(summary))
    return summary
