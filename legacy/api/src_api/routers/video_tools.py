"""
Router de herramientas de video (no-IA).

Endpoints ligeros para operar con los archivos de video ya subidos:
  GET  /videos/storage/usage         — uso de disco de cada carpeta
  GET  /videos/{id}/probe            — metadatos técnicos (resolución, fps, códec)
  GET  /videos/{id}/thumbnail        — miniatura JPEG (cacheada en disco)
  GET  /videos/{id}/frame            — frame JPEG en un segundo concreto
  POST /videos/{id}/clip             — recorta un fragmento y devuelve su ruta
  GET  /videos/{id}/download         — descarga el archivo original

Nada de esto carga YOLO/PyTorch: son operaciones de OpenCV y disco.
"""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from src.config import load_config
from src.core.video_tools import (
    dir_usage, encode_jpeg, extract_clip, frame_at, probe_video, thumbnail_jpeg,
)
from src.database.db import get_session_dep
from src.database.models import Video

router = APIRouter(prefix="/videos", tags=["Herramientas de video"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _video_o_404(db: Session, video_id: int) -> Video:
    v = db.get(Video, video_id)
    if v is None or v.retention_status == "deleted":
        raise HTTPException(404, "Video no encontrado")
    return v


def _archivo_o_404(v: Video) -> str:
    if not v.storage_key or not os.path.isfile(v.storage_key):
        raise HTTPException(409, "El archivo del video no está disponible en disco")
    return v.storage_key


# ── Uso de disco ──────────────────────────────────────────────────────────────

@router.get("/storage/usage")
def uso_de_disco():
    """Retorna cuánto ocupan las carpetas de videos, clips y miniaturas."""
    cfg = load_config()
    carpetas = {
        "videos":      cfg.videos_dir,
        "clips":       cfg.clips_dir,
        "thumbnails":  cfg.thumbnails_dir,
        "reports":     "data/reports",
    }
    detalle = {nombre: dir_usage(ruta) for nombre, ruta in carpetas.items()}
    total_mb = round(sum(d["mb"] for d in detalle.values()), 2)
    return {"total_mb": total_mb, "carpetas": detalle}


# ── Metadatos ─────────────────────────────────────────────────────────────────

@router.get("/{video_id}/probe")
def metadatos_video(video_id: int, db: Session = Depends(get_session_dep)):
    """Lee el archivo con OpenCV y devuelve resolución, fps, frames y códec."""
    v = _video_o_404(db, video_id)
    path = _archivo_o_404(v)
    info = probe_video(path)
    if info is None:
        raise HTTPException(422, "El archivo no se puede decodificar como video")
    return {
        "video_id":  v.id,
        "filename":  v.original_filename,
        "probe":     info.as_dict(),
        "db": {
            "duration_seconds": v.duration_seconds,
            "file_size_bytes":  v.file_size_bytes,
            "checksum":         v.checksum,
        },
    }


# ── Miniaturas / frames ───────────────────────────────────────────────────────

@router.get("/{video_id}/thumbnail")
def miniatura(
    video_id: int,
    at:      float = Query(1.0, ge=0, description="Segundo del video"),
    width:   int   = Query(320, ge=32, le=1280),
    quality: int   = Query(80, ge=10, le=100),
    db: Session = Depends(get_session_dep),
):
    """
    Devuelve un JPEG pequeño. Se cachea en disco para no decodificar el video
    en cada petición del dashboard.
    """
    v = _video_o_404(db, video_id)
    path = _archivo_o_404(v)
    cfg = load_config()

    cache_name = f"v{video_id}_{int(round(at * 1000))}_{width}_{quality}.jpg"
    cache_path = os.path.join(cfg.thumbnails_dir, cache_name)
    if os.path.isfile(cache_path):
        return FileResponse(cache_path, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

    data = thumbnail_jpeg(path, seconds=at, width=width, quality=quality)
    if data is None:
        raise HTTPException(422, "No se pudo generar la miniatura")

    os.makedirs(cfg.thumbnails_dir, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            f.write(data)
    except OSError:
        pass   # si falla el cache, igual devolvemos la imagen

    return Response(data, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/{video_id}/frame")
def frame_completo(
    video_id: int,
    at:       float = Query(0.0, ge=0, description="Segundo del video"),
    quality:  int   = Query(90, ge=10, le=100),
    db: Session = Depends(get_session_dep),
):
    """Devuelve un frame JPEG a resolución original (para inspección manual)."""
    v = _video_o_404(db, video_id)
    path = _archivo_o_404(v)
    frame = frame_at(path, seconds=at)
    data = encode_jpeg(frame, width=0, quality=quality)
    if data is None:
        raise HTTPException(422, "No se pudo extraer el frame")
    return Response(data, media_type="image/jpeg")


# ── Recorte de clips ──────────────────────────────────────────────────────────

@router.post("/{video_id}/clip", status_code=201)
def recortar_clip(
    video_id: int,
    start:    float = Query(0.0, ge=0, description="Segundo inicial"),
    end:      float = Query(..., gt=0, description="Segundo final"),
    db: Session = Depends(get_session_dep),
):
    """
    Recorta [start, end] del video original y guarda un MP4 en `data/clips`.

    Útil para evidencia de eventos: el operador revisa un fragmento corto
    en lugar de descargar el video completo.
    """
    v = _video_o_404(db, video_id)
    path = _archivo_o_404(v)
    cfg = load_config()

    os.makedirs(cfg.clips_dir, exist_ok=True)
    nombre = f"clip_v{video_id}_{uuid.uuid4().hex[:10]}.mp4"
    destino = os.path.join(cfg.clips_dir, nombre)

    try:
        frames = extract_clip(path, start, end, destino, max_seconds=600.0)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    if frames <= 0 or not os.path.isfile(destino):
        raise HTTPException(422, "No se pudo generar el clip (rango sin frames)")

    return {
        "video_id":   video_id,
        "filename":   nombre,
        "path":       os.path.abspath(destino),
        "start":      start,
        "end":        end,
        "frames":     frames,
        "size_bytes": os.path.getsize(destino),
    }


# ── Descarga ──────────────────────────────────────────────────────────────────

@router.get("/{video_id}/download")
def descargar_video(video_id: int, db: Session = Depends(get_session_dep)):
    """Descarga el archivo original de video."""
    v = _video_o_404(db, video_id)
    path = _archivo_o_404(v)
    return FileResponse(
        path,
        media_type=v.content_type or "video/mp4",
        filename=v.original_filename or os.path.basename(path),
    )
