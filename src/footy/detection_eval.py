"""Evaluación de la detección del balón sobre un split etiquetado (por ejemplo, con y sin SAHI)."""

import itertools
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2

from footy.dataset import iter_split, read_yolo_labels
from footy.detection import BALL_CLASS, FrameDetector
from footy.metrics import MatchCounts, match_boxes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BallResult:
    """Aciertos del balón y velocidad de un detector."""

    counts: MatchCounts
    ms_per_image: float


@dataclass(frozen=True)
class BallEvaluation:
    """Resultado de comparar varios detectores sobre las mismas imágenes."""

    images: int
    real_balls: int
    results: dict[str, BallResult]


def evaluate_ball_detection(
    detectors: Mapping[str, FrameDetector],
    dataset_dir: Path,
    split: str,
    ball_label_id: int,
    iou_threshold: float,
    max_images: int | None = None,
) -> BallEvaluation:
    """Mide recall, precisión y tiempo por imagen del balón para cada detector.

    Args:
        detectors: detectores a comparar, por nombre (ej. "Sin SAHI", "Con SAHI").
        dataset_dir: dataset en formato YOLO.
        split: "train", "val" o "test".
        ball_label_id: id de la clase ball en las etiquetas del dataset.
        iou_threshold: IoU mínimo para que un balón detectado cuente como acierto.
        max_images: si se indica, usa solo las primeras N imágenes.
    """
    counts = dict.fromkeys(detectors, MatchCounts())
    seconds = dict.fromkeys(detectors, 0.0)
    images = real_balls = 0
    for image_path, label_path in itertools.islice(iter_split(dataset_dir, split), max_images):
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("No se pudo leer %s; se omite", image_path)
            continue
        height, width = image.shape[:2]
        label_ids, label_boxes = read_yolo_labels(label_path, width, height)
        balls_in_image = label_boxes[label_ids == ball_label_id]
        if images == 0:  # calentamiento: la primera inferencia incluye cargar el modelo
            for detector in detectors.values():
                detector.detect_batch([image])
        images += 1
        real_balls += len(balls_in_image)
        for name, detector in detectors.items():
            start = time.perf_counter()
            [detections] = detector.detect_batch([image])
            seconds[name] += time.perf_counter() - start
            found = detections[detections.class_id == detector.classes.index(BALL_CLASS)]
            counts[name] += match_boxes(found.xyxy, found.confidence, balls_in_image, iou_threshold)
    return BallEvaluation(
        images=images,
        real_balls=real_balls,
        results={
            name: BallResult(counts[name], 1000 * seconds[name] / max(images, 1))
            for name in detectors
        },
    )
