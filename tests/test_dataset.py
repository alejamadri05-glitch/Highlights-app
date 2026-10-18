import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from footy import dataset
from footy.dataset import (
    RoboflowDataset,
    class_names,
    download_roboflow_export,
    extract_dataset,
    iter_split,
    normalize_data_yaml,
    read_yolo_labels,
)

ROBOFLOW_YAML = (
    "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
    "nc: 4\nnames: ['ball', 'goalkeeper', 'player', 'referee']\n"
    "roboflow:\n  license: CC BY 4.0\n"
)


def make_roboflow_export(root: Path, splits: tuple[str, ...] = ("train", "valid")) -> Path:
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(ROBOFLOW_YAML)
    return root


def test_read_yolo_labels_converts_to_pixels(tmp_path: Path) -> None:
    labels = tmp_path / "img.txt"
    labels.write_text("0 0.5 0.5 0.1 0.2\n2 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n\n")

    class_ids, boxes = read_yolo_labels(labels, width=200, height=100)

    assert class_ids.tolist() == [0, 2]
    np.testing.assert_allclose(boxes, [[90, 40, 110, 60], [20, 10, 60, 40]])


def test_missing_label_file_means_no_objects(tmp_path: Path) -> None:
    class_ids, boxes = read_yolo_labels(tmp_path / "no-existe.txt", 200, 100)
    assert class_ids.shape == (0,) and boxes.shape == (0, 4)


def test_normalize_fixes_roboflow_paths(tmp_path: Path) -> None:
    root = make_roboflow_export(tmp_path / "ds")

    data = yaml.safe_load(normalize_data_yaml(root).read_text())

    assert data["path"] == str(root.resolve())
    assert (data["train"], data["val"]) == ("train/images", "valid/images")
    assert "test" not in data  # el export no trae test/
    assert data["names"] == ["ball", "goalkeeper", "player", "referee"]
    assert data["roboflow"]["license"] == "CC BY 4.0"


def test_normalize_requires_validation_split(tmp_path: Path) -> None:
    root = make_roboflow_export(tmp_path / "ds", splits=("train",))
    with pytest.raises(ValueError):
        normalize_data_yaml(root)


def test_class_names_accepts_list_or_dict(tmp_path: Path) -> None:
    as_list = tmp_path / "a.yaml"
    as_list.write_text("names: [ball, player]\n")
    as_dict = tmp_path / "b.yaml"
    as_dict.write_text("names: {1: player, 0: ball}\n")
    assert class_names(as_list) == class_names(as_dict) == ["ball", "player"]


def test_extract_dataset(tmp_path: Path) -> None:
    source = make_roboflow_export(tmp_path / "export")
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(source / "data.yaml", "data.yaml")
        archive.writestr("train/images/a.jpg", b"jpg")

    target = extract_dataset(zip_path, tmp_path / "ds")

    assert (target / "data.yaml").is_file() and (target / "train/images/a.jpg").is_file()


def test_extract_rejects_zip_without_data_yaml(tmp_path: Path) -> None:
    zip_path = tmp_path / "otro.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("leeme.txt", "hola")
    with pytest.raises(ValueError):
        extract_dataset(zip_path, tmp_path / "ds")


def test_iter_split_pairs_images_with_labels(tmp_path: Path) -> None:
    root = make_roboflow_export(tmp_path / "ds")
    for name in ("b.jpg", "a.png", "notas.txt"):
        (root / "valid/images" / name).write_bytes(b"x")

    pairs = list(iter_split(root, "val"))

    assert [(image.name, label.name) for image, label in pairs] == [
        ("a.png", "a.txt"),
        ("b.jpg", "b.txt"),
    ]


ROBOFLOW = RoboflowDataset("ws", "football", 10, "yolov8")


def fake_api(responses: list[tuple[int, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch) -> list:
    """Reemplaza la red: cada consulta a la API devuelve la siguiente respuesta."""
    calls: list = []

    def get_json(url: str) -> tuple[int, dict[str, Any]]:
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(dataset, "_get_json", get_json)
    monkeypatch.setattr(dataset, "_download_file", lambda url, dest: dest.write_bytes(url.encode()))
    return calls


def test_download_waits_while_export_is_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = "https://example.com/export.zip"
    calls = fake_api([(202, {"progress": 0.5}), (200, {"export": {"link": link}})], monkeypatch)

    zip_path = download_roboflow_export(ROBOFLOW, "clave", tmp_path / "d.zip", poll_interval_s=0)

    assert zip_path.read_bytes() == link.encode()
    assert len(calls) == 2
    assert calls[0].startswith("https://api.roboflow.com/ws/football/10/yolov8?")


def test_download_reports_api_errors_without_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_api([(401, {"error": "Unauthorized"})], monkeypatch)
    with pytest.raises(RuntimeError, match="401") as error:
        download_roboflow_export(ROBOFLOW, "clave-secreta", tmp_path / "d.zip")
    assert "clave-secreta" not in str(error.value)


def test_download_gives_up_after_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api([(202, {"progress": 0.1})], monkeypatch)
    with pytest.raises(RuntimeError, match="202"):
        download_roboflow_export(ROBOFLOW, "clave", tmp_path / "d.zip", timeout_s=0)
