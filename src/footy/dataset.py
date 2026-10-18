"""Descarga y preparación del dataset de fútbol en formato YOLO (export de Roboflow)."""

import json
import logging
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml

logger = logging.getLogger(__name__)

ROBOFLOW_API_URL = "https://api.roboflow.com"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Nombre de la carpeta de cada split según el exportador (Roboflow usa "valid").
SPLIT_FOLDERS = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}


@dataclass(frozen=True)
class RoboflowDataset:
    """Identifica una versión de un dataset de Roboflow Universe."""

    workspace: str
    project: str
    version: int
    format: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Construye la referencia desde ``dataset.roboflow`` de config.yaml."""
        return cls(
            workspace=str(data["workspace"]),
            project=str(data["project"]),
            version=int(data["version"]),
            format=str(data["format"]),
        )

    @property
    def export_url(self) -> str:
        """Endpoint REST que devuelve el enlace de descarga del export."""
        return f"{ROBOFLOW_API_URL}/{self.workspace}/{self.project}/{self.version}/{self.format}"


def _get_json(url: str) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:
            return error.code, {"error": body[:500]}


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as f:
        shutil.copyfileobj(response, f)


def download_roboflow_export(
    dataset: RoboflowDataset,
    api_key: str,
    destination: Path,
    poll_interval_s: float = 2.0,
    timeout_s: float = 900.0,
) -> Path:
    """Descarga el .zip de un dataset de Roboflow usando su API REST.

    Roboflow responde 202 mientras genera el export y 200 con el enlace cuando está
    listo. Se usa la API directamente en lugar del paquete ``roboflow`` porque este
    instala ``opencv-python-headless``, que choca con ``opencv-python``.

    Raises:
        RuntimeError: si Roboflow responde con un error o el export no termina a tiempo.
    """
    query = urllib.parse.urlencode({"api_key": api_key})
    url = f"{dataset.export_url}?{query}"  # contiene la API key: nunca registrarla
    deadline = time.monotonic() + timeout_s
    while True:
        status, payload = _get_json(url)
        if status == 200:
            break
        if status == 202 and time.monotonic() < deadline:
            logger.info("Roboflow está generando el export (%s)...", payload.get("progress", "?"))
            time.sleep(poll_interval_s)
            continue
        raise RuntimeError(f"Roboflow respondió {status}: {payload}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Descargando %s versión %d...", dataset.project, dataset.version)
    _download_file(payload["export"]["link"], destination)
    return destination


def extract_dataset(zip_path: Path, dataset_dir: Path) -> Path:
    """Descomprime un export de Roboflow (data.yaml, train/, valid/, test/) en ``dataset_dir``."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir)
    if not (dataset_dir / "data.yaml").is_file():
        raise ValueError(f"{zip_path} no parece un export YOLO: falta data.yaml")
    return dataset_dir


def _split_images_dir(dataset_dir: Path, split: str) -> Path | None:
    for folder in SPLIT_FOLDERS[split]:
        images_dir = dataset_dir / folder / "images"
        if images_dir.is_dir():
            return images_dir
    return None


def normalize_data_yaml(dataset_dir: Path) -> Path:
    """Reescribe data.yaml con rutas que Ultralytics resuelve desde cualquier carpeta.

    Los exports de Roboflow traen rutas como ``../train/images``, que se rompen según
    dónde se ejecute el entrenamiento (el notebook oficial las corrige con ``sed``).
    """
    yaml_path = dataset_dir / "data.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["path"] = str(dataset_dir.resolve())
    for split in SPLIT_FOLDERS:
        images_dir = _split_images_dir(dataset_dir, split)
        if images_dir is None:
            data.pop(split, None)
        else:
            data[split] = images_dir.relative_to(dataset_dir).as_posix()
    if "train" not in data or "val" not in data:
        raise ValueError(f"{dataset_dir} necesita al menos train/images y valid/images")
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), "utf-8")
    return yaml_path


def class_names(data_yaml: Path) -> list[str]:
    """Nombres de clase de un data.yaml (acepta lista o diccionario ``id: nombre``)."""
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def read_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Lee un archivo de etiquetas YOLO y devuelve ``(class_ids, cajas xyxy en píxeles)``.

    Cada línea es ``clase cx cy w h`` normalizado a [0, 1]. Si la línea trae un polígono
    (``clase x1 y1 x2 y2 ...``) se usa su caja envolvente. Un archivo inexistente
    significa que la imagen no tiene objetos.
    """
    class_ids: list[int] = []
    boxes: list[list[float]] = []
    lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.is_file() else []
    for line in lines:
        values = [float(v) for v in line.split()]
        if not values:
            continue
        coords = values[1:]
        if len(coords) == 4:
            cx, cy, w, h = coords
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        class_ids.append(int(values[0]))
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
    return np.array(class_ids, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def iter_split(dataset_dir: Path, split: str) -> Iterator[tuple[Path, Path]]:
    """Recorre las imágenes de un split, entregando ``(imagen, archivo de etiquetas)``."""
    images_dir = _split_images_dir(dataset_dir, split)
    if images_dir is None:
        raise FileNotFoundError(f"No hay imágenes del split '{split}' en {dataset_dir}")
    labels_dir = images_dir.parent / "labels"
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            yield image_path, labels_dir / f"{image_path.stem}.txt"
