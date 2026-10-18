"""Generadores de datos sintéticos para los tests (nunca videos reales)."""

import subprocess
from pathlib import Path

import cv2
import numpy as np

BRIGHTNESS_STEP = 4


def write_test_video(
    path: Path, n_frames: int, fps: float, size: tuple[int, int] = (64, 48)
) -> Path:
    """Escribe un video MJPG (.avi) donde el cuadro ``i`` tiene brillo uniforme ``4 * i``.

    Así cada cuadro leído se puede reconocer por su brillo. MJPG comprime cada cuadro
    por separado; con mp4v el brillo decodificado se atrasa respecto al real.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (BRIGHTNESS_STEP * i) % 256, np.uint8))
    writer.release()
    return path


def write_vfr_video(path: Path) -> Path:
    """Video de fps variable como los de celular: 1 s a 30 fps y luego 2 s a 15 fps."""
    command = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-t", "3",
        "-vf", "setpts='if(lt(N,30),N/30,1+(N-30)/15)/TB'", "-fps_mode", "vfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True)
    return path
