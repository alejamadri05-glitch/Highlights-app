from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from footy.detection_eval import evaluate_ball_detection
from footy.metrics import MatchCounts

CLASSES = ("player", "goalkeeper", "referee", "ball")
BALL_LABEL_ID = 0  # orden del dataset: ball, goalkeeper, player, referee


def write_split(root: Path, balls: list[tuple[int, int]]) -> Path:
    """Split "valid" con una imagen 200x100 por balón (cuadro blanco de 10x10) y su etiqueta."""
    images, labels = root / "valid/images", root / "valid/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i, (x, y) in enumerate(balls):
        image = np.zeros((100, 200, 3), np.uint8)
        image[y : y + 10, x : x + 10] = 255
        cv2.imwrite(str(images / f"{i}.png"), image)
        (labels / f"{i}.txt").write_text(
            f"{BALL_LABEL_ID} {(x + 5) / 200} {(y + 5) / 100} {10 / 200} {10 / 100}\n"
            "2 0.1 0.5 0.1 0.5\n"  # un jugador, que no cuenta para el balón
        )
    return root


class WhitePixelDetector:
    """Detecta como balón la zona blanca, corrida ``shift`` píxeles."""

    classes = CLASSES

    def __init__(self, shift: int = 0) -> None:
        self.shift = shift

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[sv.Detections]:
        results = []
        for image in images:
            ys, xs = np.nonzero(image[..., 0] == 255)
            box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            results.append(
                sv.Detections(
                    xyxy=np.array([box], dtype=float) + self.shift,
                    confidence=np.array([0.9]),
                    class_id=np.array([CLASSES.index("ball")]),
                )
            )
        return results


def test_compares_detectors_on_the_same_images(tmp_path: Path) -> None:
    root = write_split(tmp_path, balls=[(20, 30), (150, 70), (90, 10)])

    evaluation = evaluate_ball_detection(
        {"exacto": WhitePixelDetector(), "corrido": WhitePixelDetector(shift=7)},
        root,
        "val",
        BALL_LABEL_ID,
        iou_threshold=0.5,
    )

    assert (evaluation.images, evaluation.real_balls) == (3, 3)
    assert evaluation.results["exacto"].counts == MatchCounts(3, 0, 0)
    assert evaluation.results["corrido"].counts == MatchCounts(0, 3, 3)
    assert evaluation.results["exacto"].ms_per_image >= 0


def test_max_images_limits_the_evaluation(tmp_path: Path) -> None:
    root = write_split(tmp_path, balls=[(20, 30), (150, 70), (90, 10)])
    evaluation = evaluate_ball_detection(
        {"exacto": WhitePixelDetector()}, root, "val", BALL_LABEL_ID, 0.5, max_images=2
    )
    assert evaluation.images == 2
    assert evaluation.results["exacto"].counts.recall == pytest.approx(1.0)
