"""Métricas de evaluación: emparejamiento de cajas por IoU, precisión y recall."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import supervision as sv


@dataclass(frozen=True)
class MatchCounts:
    """Aciertos, falsos positivos y objetos no encontrados; se pueden sumar entre imágenes."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def __add__(self, other: Self) -> Self:
        return type(self)(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )

    @property
    def precision(self) -> float:
        """Fracción de detecciones correctas (0 si no hubo detecciones)."""
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Fracción de objetos reales encontrados (0 si no había objetos)."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        """Media armónica de precisión y recall."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def match_boxes(
    pred_xyxy: np.ndarray, pred_scores: np.ndarray, gt_xyxy: np.ndarray, iou_threshold: float
) -> MatchCounts:
    """Empareja predicciones con cajas reales de una imagen.

    Emparejamiento codicioso: las predicciones de mayor confianza escogen primero la
    caja real libre con la que tienen mayor IoU, siempre que llegue a ``iou_threshold``.
    Cada caja real cuenta como acierto una sola vez.
    """
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return MatchCounts(0, len(pred_xyxy), len(gt_xyxy))
    iou = sv.box_iou_batch(pred_xyxy, gt_xyxy)
    matched = np.zeros(len(gt_xyxy), dtype=bool)
    for pred_index in np.argsort(-pred_scores):
        candidates = np.flatnonzero(~matched & (iou[pred_index] >= iou_threshold))
        if len(candidates):
            matched[candidates[np.argmax(iou[pred_index, candidates])]] = True
    true_positives = int(matched.sum())
    return MatchCounts(
        true_positives, len(pred_xyxy) - true_positives, len(gt_xyxy) - true_positives
    )
