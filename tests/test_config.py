"""El config.yaml del repositorio debe poder leerse con las clases de configuración."""

from pathlib import Path

from footy.config import load_config
from footy.detection import DetectorConfig
from footy.video_io import EncodeConfig

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_repo_config_matches_code() -> None:
    config = load_config(REPO_CONFIG)

    detection = DetectorConfig.from_dict(config["detection"])
    encode = EncodeConfig.from_dict(config["video"]["encode"])

    assert set(detection.classes) == {"player", "goalkeeper", "referee", "ball"}
    assert detection.ball_slicing.slice_wh == (640, 640)
    assert config["video"]["processing_fps"] == 10
    assert encode.crf > 0
