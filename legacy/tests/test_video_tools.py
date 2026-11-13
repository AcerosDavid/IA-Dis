"""
Tests de las utilidades de video (no-IA) en src/core/video_tools.py.
Usan un video sintético pequeño creado con OpenCV.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from src.core.video_tools import (
    dir_usage, encode_jpeg, extract_clip, frame_at, probe_video, thumbnail_jpeg,
)


@pytest.fixture()
def video_sintetico(tmp_path):
    ruta = tmp_path / "sintetico.avi"
    writer = cv2.VideoWriter(
        str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120),
    )
    assert writer.isOpened()
    for i in range(20):
        frame = np.full((120, 160, 3), (i * 10) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(ruta)


def test_probe_video_metadatos(video_sintetico):
    info = probe_video(video_sintetico)
    assert info is not None
    assert info.readable
    assert info.width == 160
    assert info.height == 120
    assert info.fps == pytest.approx(10.0, abs=0.1)
    assert info.frame_count == 20
    assert info.duration_seconds == pytest.approx(2.0, abs=0.1)
    assert info.size_bytes > 0


def test_probe_video_inexistente_o_invalido(tmp_path):
    assert probe_video("no/existe.mp4") is None
    basura = tmp_path / "basura.mp4"
    basura.write_bytes(b"no es un video")
    info = probe_video(str(basura))
    assert info is None or not info.readable


def test_frame_at_devuelve_frame(video_sintetico):
    frame = frame_at(video_sintetico, 0.5)
    assert frame is not None
    assert frame.shape == (120, 160, 3)


def test_encode_jpeg_y_thumbnail(video_sintetico):
    frame = frame_at(video_sintetico, 0.0)
    jpeg = encode_jpeg(frame, width=80, quality=70)
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"

    thumb = thumbnail_jpeg(video_sintetico, seconds=0.5, width=64)
    assert thumb is not None and thumb[:2] == b"\xff\xd8"


def test_extract_clip_genera_archivo(video_sintetico, tmp_path):
    destino = tmp_path / "clip.mp4"
    frames = extract_clip(video_sintetico, 0.5, 1.5, str(destino))
    assert frames == 10
    assert destino.is_file() and destino.stat().st_size > 0


def test_extract_clip_rango_invalido(video_sintetico, tmp_path):
    with pytest.raises(ValueError):
        extract_clip(video_sintetico, 1.5, 0.5, str(tmp_path / "x.mp4"))
    with pytest.raises(ValueError):
        extract_clip(video_sintetico, 0.0, 9999.0, str(tmp_path / "x.mp4"))


def test_dir_usage(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * (1024 * 1024))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * (512 * 1024))
    uso = dir_usage(str(tmp_path))
    assert uso["files"] == 2
    assert uso["bytes"] == 1024 * 1024 + 512 * 1024
    assert uso["mb"] == pytest.approx(1.5, rel=1e-3)
    assert os.path.isdir(tmp_path)
