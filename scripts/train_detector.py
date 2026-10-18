"""Fase 1: fine-tuning de YOLO con el dataset de fútbol de Roboflow Universe.

Uso (en una máquina con GPU):
    export ROBOFLOW_API_KEY=...        # cuenta gratuita en roboflow.com
    python scripts/train_detector.py

    # con un export YOLOv8 descargado a mano desde la página del dataset:
    python scripts/train_detector.py --zip football-players-detection.zip

    # prueba rápida del flujo en CPU (resultados sin valor):
    python scripts/train_detector.py --epochs 1 --imgsz 320 --device cpu

Al terminar copia los mejores pesos a training.output_weights de config.yaml.
"""

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from footy.config import load_config, setup_logging
from footy.dataset import (
    RoboflowDataset,
    class_names,
    download_roboflow_export,
    extract_dataset,
    normalize_data_yaml,
)

logger = logging.getLogger("train_detector")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--zip", type=Path, help="export YOLO ya descargado (en vez de la API)")
    parser.add_argument("--epochs", type=int, help="reemplaza training.epochs")
    parser.add_argument("--batch", type=int, help="reemplaza training.batch")
    parser.add_argument("--imgsz", type=int, help="reemplaza training.imgsz")
    parser.add_argument("--device", help='reemplaza training.device (ej. "cpu", "0")')
    parser.add_argument("--base-weights", help="reemplaza training.base_weights")
    return parser.parse_args(argv)


def prepare_dataset(dataset_config: dict[str, Any], zip_path: Path | None) -> Path:
    """Deja el dataset en ``dataset.dir`` (descargándolo si hace falta) y devuelve su data.yaml."""
    dataset_dir = Path(dataset_config["dir"])
    if (dataset_dir / "data.yaml").is_file():
        logger.info("Usando el dataset que ya está en %s", dataset_dir)
    elif zip_path is not None:
        extract_dataset(zip_path, dataset_dir)
    else:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise SystemExit("Falta ROBOFLOW_API_KEY (o usar --zip con un export descargado)")
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dataset_dir.parent) as tmp:
            roboflow = RoboflowDataset.from_dict(dataset_config["roboflow"])
            zip_file = download_roboflow_export(roboflow, api_key, Path(tmp) / "dataset.zip")
            extract_dataset(zip_file, dataset_dir)

    data_yaml = normalize_data_yaml(dataset_dir)
    license_name = yaml.safe_load(data_yaml.read_text()).get("roboflow", {}).get("license")
    logger.info("Clases del dataset: %s", ", ".join(class_names(data_yaml)))
    logger.info("Licencia del dataset: %s", license_name or "no indicada; revisar en Roboflow")
    return data_yaml


def train(data_yaml: Path, training: dict[str, Any], args: argparse.Namespace) -> Path:
    """Entrena con Ultralytics y copia ``best.pt`` a ``training.output_weights``."""
    from ultralytics import YOLO  # importación diferida: es pesada

    model = YOLO(args.base_weights or training["base_weights"])
    model.train(
        data=str(data_yaml),
        epochs=args.epochs or training["epochs"],
        imgsz=args.imgsz or training["imgsz"],
        batch=args.batch or training["batch"],
        patience=training["patience"],
        device=args.device or training["device"],
        seed=training["seed"],
        plots=training["plots"],
        project=str(Path(training["runs_dir"]).resolve()),
        name=training["run_name"],
    )
    output = Path(training["output_weights"])
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model.trainer.best, output)
    logger.info("Entrenamiento en %s", model.trainer.save_dir)
    return output


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    data_yaml = prepare_dataset(config["dataset"], args.zip)
    weights = train(data_yaml, config["training"], args)
    logger.info("Mejores pesos copiados a %s", weights)
    logger.info("Siguiente paso: python scripts/eval_detector.py --weights %s", weights)


if __name__ == "__main__":
    main()
