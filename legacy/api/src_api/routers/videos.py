"""
Router de Videos — gestión de archivos de video como entidades independientes.

Paso 3: Los videos son entidades de primera clase separadas de sus análisis.
Un video puede analizarse N veces con distintas configuraciones.

Endpoints:
  GET    /videos                          — lista + búsqueda paginada
  GET    /videos/{id}                     — metadatos del video
  PATCH  /videos/{id}                     — actualizar metadatos operacionales
  DELETE /videos/{id}                     — soft-delete (marca como deleted)
  GET    /videos/{id}/analysis-jobs       — historial de análisis de ese video
  POST   /videos/{id}/analysis-jobs       — crear nuevo análisis sobre el video
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.schemas import (
    VideoOut, VideoCreate, AnalysisJobOut, AnalysisJobCreate, PaginacionMeta,
)
from src.database.db import get_session_dep, RepositorioAuditoria
from src.database.models import Video, AnalysisJob

router = APIRouter(prefix="/videos", tags=["Videos"])


# ── Schemas locales ────────────────────────────────────────────────────────────

class VideoUpdate(BaseModel):
    """Campos actualizables de un video."""
    project_id:   str | None = None
    equipment_id: str | None = None
    shift_id:     str | None = None
    recorded_at:  str | None = None


class VideoListPaginado(BaseModel):
    items:      list[VideoOut]
    paginacion: PaginacionMeta


class JobsVideoOut(BaseModel):
    """Lista de análisis de un video."""
    items: list[AnalysisJobOut]
    total: int


# ── Helper ────────────────────────────────────────────────────────────────────

def _video_to_out(v: Video) -> VideoOut:
    return VideoOut(
        id               = v.id,
        original_filename = v.original_filename,
        storage_key      = v.storage_key,
        content_type     = v.content_type,
        file_size_bytes  = v.file_size_bytes,
        duration_seconds = v.duration_seconds,
        checksum         = v.checksum,
        uploaded_by      = v.uploaded_by,
        uploaded_at      = v.uploaded_at.isoformat() if v.uploaded_at else "",
        project_id       = v.project_id,
        equipment_id     = v.equipment_id,
        shift_id         = v.shift_id,
        recorded_at      = v.recorded_at.isoformat() if v.recorded_at else None,
        retention_status = v.retention_status,
    )


def _job_to_out(j: AnalysisJob) -> AnalysisJobOut:
    return AnalysisJobOut(
        id               = j.id,
        video_id         = j.video_id,
        status           = j.status,
        stage            = j.stage,
        progress_percent = j.progress_percent,
        analysis_options = j.analysis_options,
        model_id         = j.model_id,
        model_version    = j.model_version,
        created_by       = j.created_by,
        created_at       = j.created_at.isoformat() if j.created_at else "",
        started_at       = j.started_at.isoformat() if j.started_at else None,
        finished_at      = j.finished_at.isoformat() if j.finished_at else None,
        error_message    = j.error_message,
        retry_count      = j.retry_count,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=VideoListPaginado)
def listar_videos(
    project_id:   str | None = Query(None, description="Filtrar por proyecto/mina"),
    equipment_id: str | None = Query(None, description="Filtrar por equipo/cámara"),
    status:       str | None = Query(None, description="Filtrar por retention_status"),
    page:         int        = Query(1, ge=1),
    page_size:    int        = Query(20, ge=1, le=100),
    db: Session = Depends(get_session_dep),
):
    """
    Lista videos con filtros opcionales y paginación.
    Excluye videos eliminados (deleted) por defecto.
    """
    q = db.query(Video)

    # Por defecto no mostrar los eliminados
    if status:
        q = q.filter(Video.retention_status == status)
    else:
        q = q.filter(Video.retention_status != "deleted")

    if project_id:
        q = q.filter(Video.project_id == project_id)
    if equipment_id:
        q = q.filter(Video.equipment_id == equipment_id)

    total       = q.count()
    total_pages = max(1, (total + page_size - 1) // page_size)
    videos      = (
        q.order_by(Video.uploaded_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return VideoListPaginado(
        items      = [_video_to_out(v) for v in videos],
        paginacion = PaginacionMeta(
            page=page, page_size=page_size,
            total=total, total_pages=total_pages,
        ),
    )


@router.get("/{video_id}", response_model=VideoOut)
def obtener_video(video_id: int, db: Session = Depends(get_session_dep)):
    """Retorna los metadatos completos de un video."""
    v = db.get(Video, video_id)
    if v is None or v.retention_status == "deleted":
        raise HTTPException(404, "Video no encontrado")
    return _video_to_out(v)


@router.patch("/{video_id}", response_model=VideoOut)
def actualizar_video(
    video_id: int,
    datos:    VideoUpdate,
    db: Session = Depends(get_session_dep),
):
    """
    Actualiza los metadatos operacionales de un video
    (proyecto, equipo, turno, fecha de grabación).
    """
    v = db.get(Video, video_id)
    if v is None or v.retention_status == "deleted":
        raise HTTPException(404, "Video no encontrado")

    if datos.project_id   is not None: v.project_id   = datos.project_id
    if datos.equipment_id is not None: v.equipment_id = datos.equipment_id
    if datos.shift_id     is not None: v.shift_id     = datos.shift_id
    if datos.recorded_at  is not None:
        # Parsear ISO 8601 si viene como string
        try:
            v.recorded_at = datetime.fromisoformat(datos.recorded_at)
        except ValueError:
            raise HTTPException(422, f"Formato de fecha inválido: {datos.recorded_at}. Usa ISO 8601.")

    db.flush()

    # Auditoría
    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action        = "update_video",
        resource_type = "video",
        resource_id   = str(video_id),
        result        = "success",
    )

    return _video_to_out(v)


@router.delete("/{video_id}", status_code=204)
def eliminar_video(video_id: int, db: Session = Depends(get_session_dep)):
    """
    Soft-delete: marca el video como 'deleted'.
    No elimina el archivo ni los análisis asociados.
    Solo oculta el video de los listados normales.
    """
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(404, "Video no encontrado")
    if v.retention_status == "deleted":
        raise HTTPException(409, "El video ya está eliminado")

    # No permitir eliminar si hay análisis en curso
    activos = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.video_id == video_id,
            AnalysisJob.status.in_(["queued", "processing"]),
        )
        .count()
    )
    if activos > 0:
        raise HTTPException(
            409,
            f"El video tiene {activos} análisis en curso. Cancélalos antes de eliminar.",
        )

    v.retention_status = "deleted"
    db.flush()

    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action="delete_video", resource_type="video",
        resource_id=str(video_id), result="success",
    )


@router.get("/{video_id}/analysis-jobs", response_model=JobsVideoOut)
def listar_analisis_del_video(
    video_id: int,
    db: Session = Depends(get_session_dep),
):
    """
    Retorna el historial completo de análisis realizados sobre un video.
    Permite ver qué modelos y configuraciones se usaron en cada ejecución.
    """
    v = db.get(Video, video_id)
    if v is None or v.retention_status == "deleted":
        raise HTTPException(404, "Video no encontrado")

    jobs = (
        db.query(AnalysisJob)
        .filter(AnalysisJob.video_id == video_id)
        .order_by(AnalysisJob.created_at.desc())
        .all()
    )

    return JobsVideoOut(
        items = [_job_to_out(j) for j in jobs],
        total = len(jobs),
    )


@router.post("/{video_id}/analysis-jobs", response_model=AnalysisJobOut, status_code=202)
def crear_analisis(
    video_id: int,
    payload:  AnalysisJobCreate,
    db: Session = Depends(get_session_dep),
):
    """
    Crea una nueva ejecución de análisis sobre un video ya subido.
    Permite reprocesar el mismo video con otra configuración o modelo.
    El análisis corre en background; sondear GET /video/estado/{id}.
    """
    import threading
    from src.api.routers.video import _run_pipeline, _RUNTIME, _RUNTIME_LOCK, JobRuntime

    v = db.get(Video, video_id)
    if v is None or v.retention_status == "deleted":
        raise HTTPException(404, "Video no encontrado")

    if not os.path.exists(v.storage_key):
        raise HTTPException(
            409,
            f"El archivo de video no está disponible en disco: {v.storage_key}",
        )

    # Verificar concurrencia
    activos = (
        db.query(AnalysisJob)
        .filter(AnalysisJob.status.in_(["queued", "processing"]))
        .count()
    )
    from src.config import load_config
    config = load_config()
    if activos >= config.analysis_max_activos:
        raise HTTPException(429, f"Hay {activos} análisis en curso. Espera.")

    job = AnalysisJob(
        video_id         = video_id,
        status           = "queued",
        model_id         = payload.model_id or os.path.basename(config.model_path),
        model_version    = payload.model_version,
        analysis_options = payload.options or {},
    )
    db.add(job)
    db.flush()

    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action="create_analysis", resource_type="job",
        resource_id=str(job.id), result="success",
        context_json={"video_id": video_id, "model_id": job.model_id},
    )

    # COMMIT antes de arrancar el hilo: si no, el worker puede leer el job
    # antes de que exista y fallar con "Job no encontrado".
    db.commit()
    db.refresh(job)

    rt = JobRuntime(job_db_id=job.id, _video_path=v.storage_key)
    with _RUNTIME_LOCK:
        _RUNTIME[job.id] = rt

    t = threading.Thread(target=_run_pipeline, args=(job.id,), daemon=True)
    rt._thread = t
    t.start()

    return _job_to_out(job)
