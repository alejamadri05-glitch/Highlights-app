"""Carga de config.yaml y configuración del logging de los scripts."""

import logging
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    """Lee el archivo YAML de configuración.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el contenido no es un diccionario YAML.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path} debe contener un diccionario YAML")
    return config


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el formato de logging común a todos los scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
