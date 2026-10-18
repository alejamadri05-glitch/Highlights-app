"""Fase 1: reporte del detector sobre el set de validación del dataset de fútbol.

Genera <evaluation.output_dir>/detector_report.md y .json con:
  1. mAP por clase (validación de Ultralytics).
  2. Recall y precisión del balón con y sin SAHI, tiempo por imagen y el estimado
     para procesar un partido completo.

Uso:
    python scripts/eval_detector.py                    # pesos de training.output_weights
    python scripts/eval_detector.py --weights data/models/footy-detector.pt --device 0
"""

import argparse
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from footy.config import load_config, setup_logging
from footy.dataset import class_names
from footy.detection import BALL_CLASS, DetectorConfig, YoloDetector
from footy.detection_eval import BallEvaluation, evaluate_ball_detection

logger = logging.getLogger("eval_detector")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--weights", help="por defecto training.output_weights")
    parser.add_argument("--device", help='reemplaza detection.device (ej. "cpu", "0")')
    parser.add_argument("--split", help="reemplaza evaluation.split")
    parser.add_argument("--max-images", type=int, help="evalúa el balón solo en N imágenes")
    return parser.parse_args(argv)


def device_name(device: str | None) -> str:
    """Nombre legible del dispositivo, para el reporte."""
    import torch

    if device != "cpu" and torch.cuda.is_available():
        return torch.cuda.get_device_name(int(device) if device and device.isdigit() else 0)
    return "CPU"


def evaluate_map(
    model: Any, data_yaml: Path, split: str, config: DetectorConfig, runs_dir: Path
) -> dict[str, Any]:
    """mAP general y por clase con el validador de Ultralytics."""
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=config.imgsz,
        batch=config.batch_size,
        device=config.device,
        plots=False,
        project=str(runs_dir.resolve()),
        name="val",
        exist_ok=True,
        verbose=False,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": metrics.summary(),
    }


def ball_detectors(config: DetectorConfig, model: Any) -> dict[str, YoloDetector]:
    """El mismo modelo con y sin SAHI para el balón."""
    slicing = config.ball_slicing
    with_sahi = (
        f"Con SAHI (recortes {slicing.slice_wh[0]}×{slicing.slice_wh[1]}, "
        f"traslape {slicing.overlap_wh[0]} px)"
    )
    return {
        name: YoloDetector(
            dataclasses.replace(config, ball_slicing=dataclasses.replace(slicing, enabled=enabled)),
            model=model,
        )
        for name, enabled in (("Sin SAHI", False), (with_sahi, True))
    }


def match_estimate_minutes(ms_per_image: float, match_minutes: float, fps: float) -> float:
    """Minutos de cómputo para un partido completo a ``fps`` cuadros por segundo."""
    return match_minutes * 60 * fps * ms_per_image / 1000 / 60


def ball_report(evaluation: BallEvaluation, match_minutes: float, fps: float) -> dict[str, Any]:
    """Resultados del balón listos para JSON, con el estimado por partido."""
    return {
        "images": evaluation.images,
        "real_balls": evaluation.real_balls,
        "modes": {
            name: {
                **dataclasses.asdict(result.counts),
                "recall": result.counts.recall,
                "precision": result.counts.precision,
                "ms_per_image": result.ms_per_image,
                "match_estimate_min": match_estimate_minutes(
                    result.ms_per_image, match_minutes, fps
                ),
            }
            for name, result in evaluation.results.items()
        },
    }


def json_default(value: Any) -> Any:
    """Convierte tipos de numpy (Ultralytics los devuelve en las métricas) a tipos de Python."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} no se puede guardar en JSON")


def format_markdown(report: dict[str, Any]) -> str:
    """Reporte legible en Markdown."""
    info, map_report, ball = report["info"], report["map"], report["ball"]
    lines = [
        "# Reporte del detector (Fase 1)",
        "",
        f"- Pesos: `{info['weights']}`",
        f"- Dataset: `{info['dataset']}` (split `{info['split']}`)",
        f"- Tamaño de inferencia: {info['imgsz']} · confianza mínima: {info['confidence']}",
        f"- Dispositivo: {info['device']} · fecha: {info['date']}",
        "",
        "## mAP por clase",
        "",
        "| Clase | Imágenes | Instancias | Precisión | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in map_report["per_class"]:
        lines.append(
            f"| {row['Class']} | {row['Images']} | {row['Instances']} | {row['Box-P']:.3f} "
            f"| {row['Box-R']:.3f} | {row['mAP50']:.3f} | {row['mAP50-95']:.3f} |"
        )
    lines += [
        f"| **todas** | | | | | **{map_report['map50']:.3f}** | **{map_report['map50_95']:.3f}** |",
        "",
        "## Balón con y sin SAHI",
        "",
        f"{ball['real_balls']} balones reales en {ball['images']} imágenes. Un balón cuenta como "
        f"encontrado si su IoU con el real es ≥ {info['ball_match_iou']}. El estimado de "
        f"{info['match_minutes']} min a {info['fps']:g} fps procesa una imagen a la vez "
        "(cota superior: detect.py procesa por lotes).",
        "",
        "| Modo | Encontrados | Falsos positivos | Recall | Precisión | ms por imagen "
        f"| Partido de {info['match_minutes']} min |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, mode in ball["modes"].items():
        lines.append(
            f"| {name} | {mode['true_positives']} | {mode['false_positives']} "
            f"| {mode['recall']:.3f} | {mode['precision']:.3f} | {mode['ms_per_image']:.0f} "
            f"| {mode['match_estimate_min']:.0f} min |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    evaluation, training = config["evaluation"], config["training"]
    detector_config = DetectorConfig.from_dict(config["detection"])
    detector_config = dataclasses.replace(
        detector_config,
        weights=args.weights or training["output_weights"],
        device=args.device or detector_config.device,
    )
    split = args.split or evaluation["split"]
    dataset_dir = Path(config["dataset"]["dir"])
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"No está el dataset en {dataset_dir}: correr train_detector.py primero")

    from ultralytics import YOLO  # importación diferida: es pesada

    model = YOLO(detector_config.weights)
    dataset_classes = class_names(data_yaml)
    if set(model.names.values()) != set(dataset_classes):
        raise SystemExit(
            f"{detector_config.weights} tiene {len(model.names)} clases y no son las del dataset "
            f"({', '.join(dataset_classes)}): ¿son los pesos del fine-tuning?"
        )

    logger.info("Calculando mAP en el split %s...", split)
    map_report = evaluate_map(model, data_yaml, split, detector_config, Path(training["runs_dir"]))
    logger.info("Comparando el balón con y sin SAHI...")
    ball_evaluation = evaluate_ball_detection(
        ball_detectors(detector_config, model),
        dataset_dir,
        split,
        ball_label_id=dataset_classes.index(BALL_CLASS),
        iou_threshold=float(evaluation["ball_match_iou"]),
        max_images=args.max_images,
    )

    fps = float(config["video"]["processing_fps"])
    match_minutes = float(evaluation["match_minutes"])
    report = {
        "info": {
            "weights": detector_config.weights,
            "dataset": str(dataset_dir),
            "split": split,
            "imgsz": detector_config.imgsz,
            "confidence": detector_config.confidence,
            "device": device_name(detector_config.device),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ball_match_iou": evaluation["ball_match_iou"],
            "match_minutes": evaluation["match_minutes"],
            "fps": fps,
        },
        "map": map_report,
        "ball": ball_report(ball_evaluation, match_minutes, fps),
    }
    output_dir = Path(evaluation["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detector_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8"
    )
    markdown = format_markdown(report)
    (output_dir / "detector_report.md").write_text(markdown, encoding="utf-8")
    logger.info("Reporte en %s\n%s", output_dir / "detector_report.md", markdown)


if __name__ == "__main__":
    main()
