"""
Router de Eventos Detectados — con revisión humana.

Paso 4: Los eventos detectados ahora tienen ciclo de revisión.
Un operador puede confirmar, rechazar, comentar o marcar para seguimiento.

Endpoints:
  GET    /events                         — lista paginada con filtros
  GET    /events/{id}                    — detalle de un evento
  PATCH  /events/{id}/review             — confirmar / rechazar / comentar
  GET    /analysis-jobs/{job_id}/events  — eventos de una ejecución
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.api.schemas import DetectedEventOut, EventReviewIn, PaginacionMeta
from src.database.db import get_session_dep, RepositorioAuditoria
from src.database.models import DetectedEvent, AnalysisJob, Video

router = APIRouter(tags=["Eventos Detectados"])


# ── Schemas locales ───────────────────────────────────────────────────────────

class EventListPaginado(BaseModel):
    items:      list[DetectedEventOut]
    paginacion: PaginacionMeta
    pendientes: int    # total de eventos pendientes de revisión (útil para dashboard)


# ── Helper ────────────────────────────────────────────────────────────────────

def _ev_to_out(e: DetectedEvent) -> DetectedEventOut:
    return DetectedEventOut(
        id                  = e.id,
        analysis_job_id     = e.analysis_job_id,
        category            = e.category,
        description         = e.description,
        start_time_seconds  = e.start_time_seconds,
        end_time_seconds    = e.end_time_seconds,
        confidence          = e.confidence,
        severity            = e.severity,
        evidence_reference  = e.evidence_reference,
        review_status       = e.review_status,
        reviewed_by         = e.reviewed_by,
        reviewed_at         = e.reviewed_at.isoformat() if e.reviewed_at else None,
        review_comment      = e.review_comment,
        created_at          = e.created_at.isoformat() if e.created_at else "",
    )


# ── GET /events  ──────────────────────────────────────────────────────────────

@router.get("/events", response_model=EventListPaginado)
def listar_eventos_detectados(
    job_id:        int | None = Query(None, description="Filtrar por job"),
    category:      str | None = Query(None, description="Filtrar por categoría"),
    review_status: str | None = Query(None, description="Filtrar por estado de revisión"),
    severity:      str | None = Query(None, description="Filtrar por severidad"),
    page:          int        = Query(1, ge=1),
    page_size:     int        = Query(20, ge=1, le=100),
    db: Session = Depends(get_session_dep),
):
    """
    Lista eventos detectados con filtros y paginación.
    Incluye el total de pendientes para el badge del dashboard.
    """
    q = db.query(DetectedEvent)

    if job_id:
        q = q.filter(DetectedEvent.analysis_job_id == job_id)
    if category:
        q = q.filter(DetectedEvent.category == category)
    if review_status:
        q = q.filter(DetectedEvent.review_status == review_status)
    if severity:
        q = q.filter(DetectedEvent.severity == severity)

    total       = q.count()
    total_pages = max(1, (total + page_size - 1) // page_size)
    eventos     = (
        q.order_by(DetectedEvent.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    # Contar pendientes globales para el dashboard
    pendientes = (
        db.query(func.count(DetectedEvent.id))
        .filter(DetectedEvent.review_status == "pending")
        .scalar() or 0
    )

    return EventListPaginado(
        items      = [_ev_to_out(e) for e in eventos],
        paginacion = PaginacionMeta(
            page=page, page_size=page_size,
            total=total, total_pages=total_pages,
        ),
        pendientes = pendientes,
    )


# ── GET /events/{id}  ─────────────────────────────────────────────────────────

@router.get("/events/{event_id}", response_model=DetectedEventOut)
def obtener_evento_detectado(event_id: int, db: Session = Depends(get_session_dep)):
    """Retorna el detalle completo de un evento detectado."""
    e = db.get(DetectedEvent, event_id)
    if e is None:
        raise HTTPException(404, "Evento no encontrado")
    return _ev_to_out(e)


# ── PATCH /events/{id}/review  ────────────────────────────────────────────────

@router.patch("/events/{event_id}/review", response_model=DetectedEventOut)
def revisar_evento(
    event_id: int,
    datos:    EventReviewIn,
    db: Session = Depends(get_session_dep),
):
    """
    Registra la revisión humana de un evento detectado.

    Estados válidos:
      confirmed  — el evento es correcto y real
      rejected   — falso positivo, el modelo se equivocó
      corrected  — era correcto pero con categoría o datos erróneos
      follow_up  — requiere seguimiento posterior

    La acción queda registrada en audit_log con fecha y revisor.
    """
    e = db.get(DetectedEvent, event_id)
    if e is None:
        raise HTTPException(404, "Evento no encontrado")

    estados_validos = {"confirmed", "rejected", "corrected", "follow_up"}
    if datos.review_status not in estados_validos:
        raise HTTPException(
            422,
            f"Estado inválido: '{datos.review_status}'. "
            f"Opciones: {sorted(estados_validos)}",
        )

    e.review_status  = datos.review_status
    e.reviewed_at    = datetime.now(timezone.utc)
    e.reviewed_by    = getattr(datos, "reviewed_by", None)  # futuro: del token JWT
    if datos.review_comment is not None:
        e.review_comment = datos.review_comment
    db.flush()

    # Auditoría inmutable
    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action        = "review_event",
        resource_type = "event",
        resource_id   = str(event_id),
        result        = "success",
        context_json  = {
            "new_status":    datos.review_status,
            "has_comment":   datos.review_comment is not None,
        },
    )

    return _ev_to_out(e)


# ── GET /analysis-jobs/{job_id}/events  ───────────────────────────────────────

@router.get("/analysis-jobs/{job_id}/events", response_model=EventListPaginado)
def eventos_del_job(
    job_id:        int,
    category:      str | None = Query(None),
    review_status: str | None = Query(None),
    page:          int        = Query(1, ge=1),
    page_size:     int        = Query(50, ge=1, le=200),
    db: Session = Depends(get_session_dep),
):
    """
    Retorna los eventos detectados de una ejecución específica,
    ordenados por su segundo de aparición en el video.
    """
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    q = db.query(DetectedEvent).filter(DetectedEvent.analysis_job_id == job_id)
    if category:
        q = q.filter(DetectedEvent.category == category)
    if review_status:
        q = q.filter(DetectedEvent.review_status == review_status)

    total       = q.count()
    total_pages = max(1, (total + page_size - 1) // page_size)
    eventos     = (
        q.order_by(DetectedEvent.start_time_seconds.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    pendientes = (
        db.query(func.count(DetectedEvent.id))
        .filter(
            DetectedEvent.analysis_job_id == job_id,
            DetectedEvent.review_status   == "pending",
        )
        .scalar() or 0
    )

    return EventListPaginado(
        items      = [_ev_to_out(e) for e in eventos],
        paginacion = PaginacionMeta(
            page=page, page_size=page_size,
            total=total, total_pages=total_pages,
        ),
        pendientes = pendientes,
    )
