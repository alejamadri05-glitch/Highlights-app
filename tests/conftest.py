"""Fixtures compartidos: videos sintéticos pequeños."""

import shutil
from pathlib import Path

import pytest

from synthetic import write_test_video, write_vfr_video


@pytest.fixture
def cfr_video(tmp_path: Path) -> Path:
    """Video de 2 s a 30 fps constantes (60 cuadros)."""
    return write_test_video(tmp_path / "cfr.avi", n_frames=60, fps=30)


@pytest.fixture
def vfr_video(tmp_path: Path) -> Path:
    """Video de 3 s con fps variable (30 fps el primer segundo, luego 15 fps)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg no está instalado")
    return write_vfr_video(tmp_path / "vfr.mp4")
