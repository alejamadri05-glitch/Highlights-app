"""Lectura de video con OpenCV, muestreo de cuadros por tiempo y escritura en H.264 con FFmpeg."""

import logging
import math
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Tolerancia (en fracciones de intervalo) para que el redondeo de las marcas de tiempo
# no mande un cuadro al intervalo equivocado: 99.9999 ms debe contar como 100 ms.
_SLOT_EPSILON = 1e-3


class VideoReadError(RuntimeError):
    """El video no existe, está corrupto o no se puede decodificar."""


@dataclass(frozen=True)
class VideoInfo:
    """Metadatos básicos de un video según OpenCV."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        """Duración aproximada en segundos (exacta solo en videos de fps constante)."""
        return self.frame_count / self.fps


@dataclass(frozen=True)
class Frame:
    """Un cuadro decodificado del video original."""

    index: int  # posición en el video original, empezando en 0
    timestamp_s: float
    image: np.ndarray  # BGR, alto x ancho x 3


@dataclass(frozen=True)
class EncodeConfig:
    """Parámetros de libx264 para los videos de salida."""

    crf: int
    preset: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Construye la configuración desde la sección ``video.encode`` de config.yaml."""
        return cls(crf=int(data["crf"]), preset=str(data["preset"]))


def _open_capture(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise VideoReadError(f"No existe el video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"OpenCV no pudo abrir el video (¿corrupto?): {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or math.isnan(fps):
        cap.release()
        raise VideoReadError(f"El video no reporta fps válidos: {path}")
    return cap


def probe_video(path: Path) -> VideoInfo:
    """Lee los metadatos del video sin decodificarlo."""
    cap = _open_capture(path)
    try:
        return VideoInfo(
            path=path,
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def frame_timestamp(pos_msec: float, index: int, fps: float) -> float:
    """Marca de tiempo en segundos del cuadro que se acaba de leer.

    Usa la posición que reporta el decodificador, que respeta los videos de fps
    variable que graban los celulares. Si el backend no la reporta (devuelve 0 después
    del primer cuadro), cae a ``index / fps``.
    """
    if pos_msec > 0 or index == 0:
        return pos_msec / 1000.0
    return index / fps


class FrameSampler:
    """Elige como máximo un cuadro por cada intervalo de ``1 / target_fps`` segundos.

    El primer cuadro de cada intervalo es el que se procesa. Con ``target_fps=None``
    se aceptan todos los cuadros.
    """

    def __init__(self, target_fps: float | None) -> None:
        if target_fps is not None and target_fps <= 0:
            raise ValueError(f"target_fps debe ser positivo, no {target_fps}")
        self._target_fps = target_fps
        self._last_slot = -1

    def accept(self, timestamp_s: float) -> bool:
        """Indica si el cuadro con esta marca de tiempo debe procesarse."""
        if self._target_fps is None:
            return True
        slot = math.floor(timestamp_s * self._target_fps + _SLOT_EPSILON)
        if slot <= self._last_slot:
            return False
        self._last_slot = slot
        return True


def iter_frames(
    path: Path, target_fps: float | None = None, max_seconds: float | None = None
) -> Iterator[Frame]:
    """Recorre el video entregando los cuadros muestreados a ``target_fps``.

    Args:
        path: video de entrada.
        target_fps: cuadros por segundo a entregar; ``None`` entrega todos.
        max_seconds: si se indica, se detiene al pasar esa marca de tiempo.

    Raises:
        VideoReadError: si el video no se puede abrir.
    """
    cap = _open_capture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sampler = FrameSampler(target_fps)
    index = -1
    try:
        while cap.grab():
            index += 1
            timestamp_s = frame_timestamp(cap.get(cv2.CAP_PROP_POS_MSEC), index, fps)
            if max_seconds is not None and timestamp_s >= max_seconds:
                break
            if not sampler.accept(timestamp_s):
                continue
            ok, image = cap.retrieve()
            if not ok:
                logger.warning("No se pudo decodificar el cuadro %d de %s; se omite", index, path)
                continue
            yield Frame(index=index, timestamp_s=timestamp_s, image=image)
    finally:
        cap.release()


def ffmpeg_h264_command(
    path: Path, width: int, height: int, fps: float, encode: EncodeConfig
) -> list[str]:
    """Comando de FFmpeg que recibe cuadros BGR por stdin y escribe un MP4 H.264."""
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
        "-i", "-",
        "-an",
        # yuv420p (necesario para que se reproduzca en celulares) exige ancho y alto pares
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", encode.preset, "-crf", str(encode.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]  # fmt: skip


class VideoWriter:
    """Escribe cuadros BGR a un MP4 H.264 que se reproduce en navegadores y celulares.

    El tamaño del video se toma del primer cuadro. Si FFmpeg no está instalado, usa el
    códec mp4v de OpenCV, que quizás no se reproduzca en el navegador ni en WhatsApp.
    """

    def __init__(self, path: Path, fps: float, encode: EncodeConfig) -> None:
        self.path = path
        self.fps = fps
        self.frames_written = 0
        self._encode = encode
        self._size: tuple[int, int] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._cv_writer: cv2.VideoWriter | None = None

    def write(self, image: np.ndarray) -> None:
        """Agrega un cuadro BGR; todos deben tener el mismo tamaño."""
        height, width = image.shape[:2]
        if self._size is None:
            self._open(width, height)
        elif (width, height) != self._size:
            raise ValueError(f"Cuadro de {width}x{height}; el video es de {self._size}")
        if self._process is not None:
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(image.tobytes())
            except BrokenPipeError as error:
                raise RuntimeError(f"FFmpeg falló: {self._finish_ffmpeg()}") from error
        else:
            assert self._cv_writer is not None
            self._cv_writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        """Termina el archivo; debe llamarse siempre (o usar ``with``)."""
        if self._process is not None:
            stderr = self._finish_ffmpeg()
            if self._process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg terminó con código {self._process.returncode}: {stderr}"
                )
            self._process = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _open(self, width: int, height: int) -> None:
        self._size = (width, height)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg"):
            command = ffmpeg_h264_command(self.path, width, height, self.fps, self._encode)
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            return
        logger.warning("FFmpeg no está instalado: %s se escribe con mp4v de OpenCV", self.path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv_writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
        if not self._cv_writer.isOpened():
            raise RuntimeError(f"OpenCV no pudo crear {self.path}")

    def _finish_ffmpeg(self) -> str:
        assert self._process is not None
        _, stderr = self._process.communicate()
        return stderr.decode("utf-8", errors="replace").strip()
