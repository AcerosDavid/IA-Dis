"""
video_tools.py — Utilidades de video independientes de la IA.

Provee metadatos, miniaturas, frames y recortes usando OpenCV. No importa
YOLO ni PyTorch, por lo que puede usarse en endpoints ligeros sin cargar
modelos ni consumir CPU de inferencia.

Funciones principales:
    probe_video(path)                  → VideoInfo | None
    frame_at(path, seconds)            → np.ndarray | None
    encode_jpeg(frame, width, quality) → bytes
    thumbnail_jpeg(path, seconds, ...) → bytes | None
    extract_clip(path, start, end, dst)→ int (frames escritos)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Metadatos técnicos de un archivo de video."""
    path:             str
    width:            int
    height:           int
    fps:              float
    frame_count:      int
    duration_seconds: float
    fourcc:           str
    size_bytes:       int
    readable:         bool = True

    def as_dict(self) -> dict:
        return {
            "path":             self.path,
            "width":            self.width,
            "height":           self.height,
            "fps":              round(self.fps, 3),
            "frame_count":      self.frame_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "fourcc":           self.fourcc,
            "size_bytes":       self.size_bytes,
            "readable":         self.readable,
        }


def _fourcc_to_str(code: int) -> str:
    """Convierte el entero de CAP_PROP_FOURCC a un string legible ('avc1')."""
    try:
        chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
        return "".join(c for c in chars if c.isprintable()).strip() or "?"
    except Exception:
        return "?"


def probe_video(path: str) -> VideoInfo | None:
    """
    Lee metadatos del video sin decodificar todos los frames.

    Retorna None si el archivo no existe o OpenCV no puede abrirlo.
    """
    if not path or not os.path.isfile(path):
        return None

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None

        fps         = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc      = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC) or 0))

        if fps <= 0:
            fps = 0.0
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0

        return VideoInfo(
            path             = os.path.abspath(path),
            width            = width,
            height           = height,
            fps              = fps,
            frame_count      = frame_count,
            duration_seconds = duration,
            fourcc           = fourcc,
            size_bytes       = os.path.getsize(path),
            readable         = width > 0 and height > 0,
        )
    finally:
        cap.release()


def frame_at(path: str, seconds: float) -> np.ndarray | None:
    """
    Extrae el frame más cercano al segundo indicado.

    Si el seek falla o se sale del rango, intenta el primer frame.
    """
    if not path or not os.path.isfile(path):
        return None

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None

        fps   = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx   = max(0, int(round(seconds * fps)))
        if total > 0:
            idx = min(idx, total - 1)

        if idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ok, frame = cap.read()
        if not ok and idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def encode_jpeg(frame: np.ndarray, width: int = 0, quality: int = 85) -> bytes | None:
    """Codifica un frame BGR como JPEG. Si width>0 redimensiona manteniendo aspecto."""
    if frame is None or frame.size == 0:
        return None

    if width and width > 0 and frame.shape[1] > width:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)

    quality = max(1, min(100, int(quality)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def thumbnail_jpeg(
    path: str,
    seconds: float = 1.0,
    width: int = 320,
    quality: int = 80,
) -> bytes | None:
    """Atajo: frame_at + encode_jpeg."""
    return encode_jpeg(frame_at(path, seconds), width=width, quality=quality)


def extract_clip(
    path: str,
    start_seconds: float,
    end_seconds: float,
    dest: str,
    max_seconds: float = 600.0,
) -> int:
    """
    Recorta un fragmento [start, end] y lo guarda como MP4 (códec mp4v).

    Returns:
        Número de frames escritos. 0 si no se pudo generar.
    Raises:
        ValueError: si el rango es inválido o excede max_seconds.
    """
    if not path or not os.path.isfile(path):
        raise ValueError("Archivo de video no encontrado")
    if end_seconds <= start_seconds:
        raise ValueError("end debe ser mayor que start")
    if (end_seconds - start_seconds) > max_seconds:
        raise ValueError(f"El clip supera el máximo de {max_seconds:.0f}s")

    cap = cv2.VideoCapture(path)
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV no pudo abrir el video")

        fps   = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            raise ValueError("Dimensiones de video inválidas")

        start_frame = max(0, int(round(start_seconds * fps)))
        end_frame   = int(round(end_seconds * fps))
        if total > 0:
            end_frame = min(end_frame, total)
        if end_frame <= start_frame:
            raise ValueError("Rango de frames vacío")

        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        writer = cv2.VideoWriter(
            dest, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
        )
        if not writer.isOpened():
            raise ValueError("No se pudo crear el archivo de salida")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        escrito = 0
        for _ in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            escrito += 1
        return escrito
    finally:
        if writer is not None:
            writer.release()
        cap.release()


def dir_usage(path: str) -> dict:
    """Retorna cantidad de archivos y bytes totales de un directorio."""
    total_files = 0
    total_bytes = 0
    if not path or not os.path.isdir(path):
        return {"path": path, "files": 0, "bytes": 0, "mb": 0.0}
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, name))
                total_files += 1
            except OSError:
                continue
    return {
        "path":  path,
        "files": total_files,
        "bytes": total_bytes,
        "mb":    round(total_bytes / (1024 * 1024), 2),
    }
