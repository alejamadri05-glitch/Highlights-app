"""Fase 1: corre el detector sobre un video y guarda annotated.mp4 y detections.json.

Uso:
    python scripts/detect.py --video data/raw/partido1.mp4
    python scripts/detect.py --video data/raw/partido1.mp4 --ball-slicing --max-seconds 60

Los resultados quedan en ``<output_dir>/<nombre del video>/`` (por defecto outputs/partido1/).
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from footy.config import load_config, setup_logging
from footy.detection import DetectorConfig, YoloDetector, run_detection
from footy.video_io import EncodeConfig, VideoReadError

logger = logging.getLogger("detect")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True, help="video de entrada")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, help="por defecto <output_dir>/<video>/")
    parser.add_argument("--fps", type=float, help="reemplaza video.processing_fps")
    parser.add_argument("--weights", help="reemplaza detection.weights")
    parser.add_argument("--device", help='reemplaza detection.device (ej. "cpu", "0")')
    parser.add_argument(
        "--ball-slicing",
        action=argparse.BooleanOptionalAction,
        help="activa/desactiva SAHI para el balón (reemplaza detection.ball_slicing.enabled)",
    )
    parser.add_argument("--max-seconds", type=float, help="procesa solo los primeros N segundos")
    return parser.parse_args(argv)


def detector_config_from_args(config: dict, args: argparse.Namespace) -> DetectorConfig:
    """Lee la sección detection de config.yaml y aplica lo que se pasó por línea de comandos."""
    detector_config = DetectorConfig.from_dict(config["detection"])
    if args.weights:
        detector_config = dataclasses.replace(detector_config, weights=args.weights)
    if args.device:
        detector_config = dataclasses.replace(detector_config, device=args.device)
    if args.ball_slicing is not None:
        slicing = dataclasses.replace(detector_config.ball_slicing, enabled=args.ball_slicing)
        detector_config = dataclasses.replace(detector_config, ball_slicing=slicing)
    return detector_config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    detector_config = detector_config_from_args(config, args)
    processing_fps = args.fps or float(config["video"]["processing_fps"])
    output_dir = args.output_dir or Path(config["output_dir"]) / args.video.stem

    logger.info(
        "Modelo %s, %g fps, SAHI para el balón: %s",
        detector_config.weights,
        processing_fps,
        "sí" if detector_config.ball_slicing.enabled else "no",
    )
    detector = YoloDetector(detector_config)
    try:
        summary = run_detection(
            video_path=args.video,
            detector=detector,
            output_dir=output_dir,
            processing_fps=processing_fps,
            batch_size=detector_config.batch_size,
            encode=EncodeConfig.from_dict(config["video"]["encode"]),
            max_seconds=args.max_seconds,
            metadata={
                "model": detector_config.weights,
                "imgsz": detector_config.imgsz,
                "confidence": detector_config.confidence,
                "ball_slicing": detector_config.ball_slicing.enabled,
            },
        )
    except VideoReadError as error:
        logger.error("%s", error)
        raise SystemExit(1) from error

    logger.info(
        "Listo: %d cuadros en %.1f s (%.0f ms por cuadro). Detecciones: %s. Cuadros con balón: %d",
        summary.frames_processed,
        summary.processing_time_s,
        summary.ms_per_frame,
        summary.detections_per_class,
        summary.frames_with_ball,
    )
    logger.info("Resultados en %s", output_dir)


if __name__ == "__main__":
    main()
