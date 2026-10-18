from pathlib import Path

import cv2
import numpy as np
import pytest

from footy import video_io
from footy.video_io import (
    EncodeConfig,
    FrameSampler,
    VideoReadError,
    VideoWriter,
    frame_timestamp,
    iter_frames,
    probe_video,
)
from synthetic import BRIGHTNESS_STEP

ENCODE = EncodeConfig(crf=23, preset="ultrafast")


def accepted_indices(sampler: FrameSampler, timestamps: list[float]) -> list[int]:
    return [i for i, t in enumerate(timestamps) if sampler.accept(t)]


def test_sampler_keeps_first_frame_of_each_interval() -> None:
    timestamps = [i / 30 for i in range(30)]
    assert accepted_indices(FrameSampler(10), timestamps) == list(range(0, 30, 3))


def test_sampler_tolerates_rounding_in_timestamps() -> None:
    # 0.0999999 s pertenece al intervalo de 0.1 s, no al de 0.0 s
    assert accepted_indices(FrameSampler(10), [0.0, 0.0999999, 0.1333]) == [0, 1]


def test_sampler_keeps_every_frame_when_target_exceeds_source() -> None:
    timestamps = [i / 25 for i in range(10)]
    assert accepted_indices(FrameSampler(60), timestamps) == list(range(10))
    assert accepted_indices(FrameSampler(None), timestamps) == list(range(10))


def test_sampler_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError):
        FrameSampler(0)


def test_frame_timestamp_prefers_decoder_position() -> None:
    assert frame_timestamp(1500.0, index=10, fps=30) == pytest.approx(1.5)
    assert frame_timestamp(0.0, index=0, fps=30) == 0.0
    # backend sin posición: se calcula con el índice
    assert frame_timestamp(0.0, index=5, fps=25) == pytest.approx(0.2)


def test_probe_video(cfr_video: Path) -> None:
    info = probe_video(cfr_video)
    assert info.fps == pytest.approx(30)
    assert info.frame_count == 60
    assert (info.width, info.height) == (64, 48)
    assert info.duration_s == pytest.approx(2.0)


def test_iter_frames_samples_at_target_fps(cfr_video: Path) -> None:
    frames = list(iter_frames(cfr_video, target_fps=10))

    assert [f.index for f in frames] == list(range(0, 60, 3))
    assert frames[1].timestamp_s == pytest.approx(0.1, abs=1e-3)
    for frame in frames:  # cada imagen corresponde a su índice
        assert frame.image.mean() == pytest.approx(BRIGHTNESS_STEP * frame.index, abs=3)


def test_iter_frames_stops_at_max_seconds(cfr_video: Path) -> None:
    frames = list(iter_frames(cfr_video, target_fps=10, max_seconds=1.0))
    assert len(frames) == 10
    assert frames[-1].timestamp_s < 1.0


def test_iter_frames_uses_real_timestamps_on_variable_fps(vfr_video: Path) -> None:
    frames = list(iter_frames(vfr_video, target_fps=10))
    timestamps = [f.timestamp_s for f in frames]

    # 10 cuadros por segundo durante los 3 s, aunque el fps promedio sea ~20
    assert len(frames) == 30
    assert frames[10].index == 30 and frames[10].timestamp_s == pytest.approx(1.0, abs=1e-3)
    assert timestamps[-1] == pytest.approx(2.9333, abs=1e-3)
    assert all(0.066 < b - a < 0.134 for a, b in zip(timestamps, timestamps[1:], strict=False))


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoReadError):
        probe_video(tmp_path / "no-existe.mp4")


def test_corrupt_video_raises(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupto.mp4"
    corrupt.write_bytes(np.random.default_rng(0).bytes(4096))
    with pytest.raises(VideoReadError):
        list(iter_frames(corrupt))


def read_back(path: Path) -> tuple[int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.grab():
        count += 1
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return count, size


def test_writer_produces_h264_with_even_dimensions(tmp_path: Path) -> None:
    if video_io.shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg no está instalado")
    path = tmp_path / "out" / "annotated.mp4"
    with VideoWriter(path, fps=10, encode=ENCODE) as writer:
        for _ in range(12):
            writer.write(np.zeros((49, 65, 3), np.uint8))

    assert writer.frames_written == 12
    assert read_back(path) == (12, (64, 48))
    cap = cv2.VideoCapture(str(path))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    cap.release()
    assert fourcc in {"avc1", "h264"}


def test_writer_falls_back_to_opencv_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video_io.shutil, "which", lambda _: None)
    path = tmp_path / "fallback.mp4"
    with VideoWriter(path, fps=10, encode=ENCODE) as writer:
        for _ in range(5):
            writer.write(np.zeros((48, 64, 3), np.uint8))
    assert read_back(path) == (5, (64, 48))


def test_writer_rejects_frames_of_different_size(tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "x.mp4", fps=10, encode=ENCODE) as writer:
        writer.write(np.zeros((48, 64, 3), np.uint8))
        with pytest.raises(ValueError):
            writer.write(np.zeros((50, 64, 3), np.uint8))
