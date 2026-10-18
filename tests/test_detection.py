import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from footy.detection import (
    BallSlicingConfig,
    DetectionsJsonWriter,
    DetectorConfig,
    YoloDetector,
    build_class_mapping,
    merge_ball_detections,
    remap_classes,
    run_detection,
)
from footy.video_io import EncodeConfig, Frame

CLASSES = ("player", "goalkeeper", "referee", "ball")
COCO_MAP = {"person": "player", "sports ball": "ball"}
COCO_NAMES = {0: "person", 2: "car", 32: "sports ball"}
PLAYER, BALL = 0, 3


def make_detections(
    boxes: list[list[float]], class_ids: list[int], conf: float = 0.9
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float).reshape(-1, 4),
        confidence=np.full(len(boxes), conf),
        class_id=np.array(class_ids, dtype=int),
    )


def make_config(slicing: bool = False) -> DetectorConfig:
    return DetectorConfig(
        weights="falso.pt",
        device="cpu",
        imgsz=1280,
        confidence=0.25,
        nms_iou=0.7,
        batch_size=4,
        classes=CLASSES,
        class_map=COCO_MAP,
        ball_slicing=BallSlicingConfig(
            enabled=slicing, slice_wh=(640, 640), overlap_wh=(128, 128), merge_iou=0.5, batch_size=4
        ),
    )


def test_class_mapping_translates_coco_names() -> None:
    assert build_class_mapping(COCO_NAMES, CLASSES, COCO_MAP) == {0: PLAYER, 32: BALL}


def test_class_mapping_uses_finetuned_names_as_is() -> None:
    names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    assert build_class_mapping(names, CLASSES, COCO_MAP) == {0: 3, 1: 1, 2: 0, 3: 2}


def test_remap_drops_unmapped_classes_and_names() -> None:
    raw = make_detections([[0, 0, 10, 10], [5, 5, 9, 9], [20, 20, 25, 25]], [0, 2, 32])
    raw.data["class_name"] = np.array(["person", "car", "sports ball"])

    remapped = remap_classes(raw, {0: PLAYER, 32: BALL})

    assert remapped.class_id.tolist() == [PLAYER, BALL]
    assert remapped.xyxy.tolist() == [[0, 0, 10, 10], [20, 20, 25, 25]]
    assert remapped.data == {}


def test_remap_handles_empty_detections() -> None:
    assert len(remap_classes(sv.Detections.empty(), {0: PLAYER})) == 0


def test_merge_removes_duplicate_balls_and_keeps_players() -> None:
    full = make_detections([[0, 0, 50, 100], [100, 100, 110, 110]], [PLAYER, BALL])
    sliced = make_detections([[101, 101, 111, 111], [500, 500, 510, 510]], [BALL, BALL], conf=0.6)

    merged = merge_ball_detections(full, sliced, BALL, iou_threshold=0.5)

    assert sorted(merged.class_id.tolist()) == [PLAYER, BALL, BALL]
    balls = merged.xyxy[merged.class_id == BALL].tolist()
    assert [100, 100, 110, 110] in balls and [500, 500, 510, 510] in balls


def test_merge_without_any_ball() -> None:
    full = make_detections([[0, 0, 50, 100]], [PLAYER])
    merged = merge_ball_detections(full, sv.Detections.empty(), BALL, iou_threshold=0.5)
    assert merged.class_id.tolist() == [PLAYER]


class FakeModel:
    names = COCO_NAMES


class WhiteSquareDetector(YoloDetector):
    """YOLO simulado: siempre ve un jugador fijo y el balón es cualquier zona blanca."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config, model=FakeModel())
        self.calls: list[tuple[int, int]] = []  # (imgsz, número de imágenes)

    def _run_model(
        self, images: Sequence[np.ndarray], imgsz: int, model_class_ids: list[int]
    ) -> list[sv.Detections]:
        self.calls.append((imgsz, len(images)))
        results = []
        for image in images:
            boxes, ids = [], []
            if 0 in model_class_ids:
                boxes.append([0, 0, 50, 100])
                ids.append(0)
            ys, xs = np.nonzero(image[..., 0] == 255)
            if 32 in model_class_ids and len(xs):
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
                ids.append(32)
            results.append(make_detections(boxes, ids))
        return results


def image_with_ball(x: int, y: int) -> np.ndarray:
    image = np.zeros((720, 1280, 3), np.uint8)
    image[y : y + 10, x : x + 10] = 255
    return image


def test_detector_without_slicing_runs_one_full_frame_pass() -> None:
    detector = WhiteSquareDetector(make_config(slicing=False))
    [detections] = detector.detect_batch([image_with_ball(600, 550)])

    assert sorted(detections.class_id.tolist()) == [PLAYER, BALL]
    assert detector.calls == [(1280, 1)]


def test_detector_with_slicing_finds_ball_once_in_frame_coordinates() -> None:
    # El balón está en la zona donde se traslapan 4 recortes: debe quedar uno solo.
    detector = WhiteSquareDetector(make_config(slicing=True))
    [detections] = detector.detect_batch([image_with_ball(600, 550)])

    balls = detections.xyxy[detections.class_id == BALL]
    assert balls.tolist() == [[600, 550, 610, 560]]
    assert (detections.class_id == PLAYER).sum() == 1
    assert any(imgsz == 640 for imgsz, _ in detector.calls)


def test_slicing_requires_a_model_that_detects_balls() -> None:
    class NoBallModel:
        names = {0: "person"}

    with pytest.raises(ValueError, match="balones"):
        YoloDetector(make_config(slicing=True), model=NoBallModel())


def test_model_without_project_classes_is_rejected() -> None:
    class CarModel:
        names = {0: "car"}

    with pytest.raises(ValueError, match="ninguna"):
        YoloDetector(make_config(), model=CarModel())


def test_json_writer_streams_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "detections.json"
    frame = Frame(index=3, timestamp_s=0.1, image=np.zeros((4, 4, 3), np.uint8))
    with DetectionsJsonWriter(path, {"video": "partido.mp4"}, CLASSES) as writer:
        writer.write_frame(frame, make_detections([[1.234, 2, 3, 4]], [BALL], conf=0.87654))
        writer.write_frame(frame, sv.Detections.empty())
        writer.finish({"frames_processed": 2})

    data = json.loads(path.read_text())
    assert data["video"] == "partido.mp4"
    assert data["summary"] == {"frames_processed": 2}
    assert data["frames"][0] == {
        "frame": 3,
        "timestamp_s": 0.1,
        "detections": [{"class": "ball", "confidence": 0.877, "bbox_xyxy": [1.2, 2.0, 3.0, 4.0]}],
    }
    assert data["frames"][1]["detections"] == []


def test_json_writer_leaves_partial_file_on_error(tmp_path: Path) -> None:
    path = tmp_path / "detections.json"
    with pytest.raises(RuntimeError), DetectionsJsonWriter(path, {"video": "x"}, CLASSES):
        raise RuntimeError("se cayó a mitad del video")
    assert not path.exists()
    assert (tmp_path / "detections.json.partial").exists()


class FixedPlayerDetector:
    """Cumple FrameDetector: un jugador fijo por imagen."""

    classes = CLASSES

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[sv.Detections]:
        return [make_detections([[1, 1, 20, 40]], [PLAYER]) for _ in images]


def test_run_detection_writes_annotated_video_and_json(tmp_path: Path, cfr_video: Path) -> None:
    out_dir = tmp_path / "salida"
    summary = run_detection(
        video_path=cfr_video,
        detector=FixedPlayerDetector(),
        output_dir=out_dir,
        processing_fps=10,
        batch_size=4,
        encode=EncodeConfig(crf=23, preset="ultrafast"),
        metadata={"model": "falso.pt"},
    )

    data = json.loads((out_dir / "detections.json").read_text())
    assert summary.frames_processed == 20
    assert summary.detections_per_class == {"player": 20}
    assert summary.frames_with_ball == 0
    assert (data["width"], data["height"], data["fps_processed"]) == (64, 48, 10)
    assert data["model"] == "falso.pt"
    assert [f["frame"] for f in data["frames"]] == list(range(0, 60, 3))
    assert data["summary"]["frames_processed"] == 20

    cap = cv2.VideoCapture(str(out_dir / "annotated.mp4"))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 20
    cap.release()


def test_run_detection_respects_max_seconds(tmp_path: Path, cfr_video: Path) -> None:
    summary = run_detection(
        video_path=cfr_video,
        detector=FixedPlayerDetector(),
        output_dir=tmp_path,
        processing_fps=10,
        batch_size=8,
        encode=EncodeConfig(crf=23, preset="ultrafast"),
        max_seconds=0.5,
    )
    assert summary.frames_processed == 5
