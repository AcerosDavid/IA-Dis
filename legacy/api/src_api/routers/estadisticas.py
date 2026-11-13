"""
Router de estadísticas: resumen y métricas del sistema.

Endpoints heredados:
  GET /estadisticas/resumen        — totales de eventos y ciclos (legacy)
  GET /estadisticas/ciclos         — stats de ciclos por tipo de máquina
  GET /estadisticas/productividad  — ciclos por hora por tipo

Endpoints nuevos (Paso 6):
  GET /dashboard/summary           — indicadores completos para el dashboard
  GET /estadisticas/jobs           — estado de los jobs de análisis
  GET /estadisticas/videos         — métricas de videos subidos
  GET /estadisticas/eventos-detectados — eventos detectados y pendientes de revisión
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.api.schemas import (
    ResumenSchema,
    EstadisticasTipoSchema,
    EstadisticasZonaSchema,
    EstadisticasCiclosSchema,
)
from src.database.db import RepositorioCiclos, get_session_dep

router = APIRouter(tags=["Estadísticas"])


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS HEREDADOS (sin cambios)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/estadisticas/resumen", response_model=ResumenSchema)
def resumen(db: Session = Depends(get_session_dep)):
    """Resumen general del sistema (legacy)."""
    from src.database.models import Evento, Ciclo

    try:
        total_eventos = db.query(func.count(Evento.id)).scalar() or 0
        total_ciclos  = db.query(func.count(Ciclo.id)).scalar() or 0
        ciclos_completados = (
            db.query(func.count(Ciclo.id))
            .filter(Ciclo.completado == True)
            .scalar() or 0
        )

        por_tipo_rows = (
            db.query(Evento.tipo_maquina, func.count(Evento.id).label("total"))
            .group_by(Evento.tipo_maquina)
            .all()
        )
        por_tipo = [
            EstadisticasTipoSchema(tipo_maquina=r.tipo_maquina, total=r.total)
            for r in por_tipo_rows
        ]

        por_zona_rows = (
            db.query(Evento.zona_nombre, func.count(Evento.id).label("total"))
            .filter(Evento.zona_nombre != None)
            .group_by(Evento.zona_nombre)
            .all()
        )
        por_zona = [
            EstadisticasZonaSchema(zona_nombre=r.zona_nombre, total=r.total)
            for r in por_zona_rows
        ]

        return ResumenSchema(
            total_eventos      = total_eventos,
            total_ciclos       = total_ciclos,
            ciclos_completados = ciclos_completados,
            por_tipo           = por_tipo,
            por_zona           = por_zona,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/estadisticas/ciclos", response_model=list[EstadisticasCiclosSchema])
def estadisticas_ciclos(db: Session = Depends(get_session_dep)):
    """Estadísticas de ciclos completados por tipo de máquina."""
    try:
        repo = RepositorioCiclos(db)
        return [EstadisticasCiclosSchema(**row) for row in repo.estadisticas()]
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/estadisticas/productividad")
def productividad(db: Session = Depends(get_session_dep)):
    """Ciclos completados por hora para cada tipo de máquina."""
    from src.database.models import Ciclo

    try:
        rows = (
            db.query(
                Ciclo.tipo_maquina,
                func.count(Ciclo.id).label("total_ciclos"),
                func.avg(Ciclo.duracion_segundos).label("duracion_promedio"),
            )
            .filter(Ciclo.completado == True, Ciclo.duracion_segundos > 0)
            .group_by(Ciclo.tipo_maquina)
            .all()
        )

        resultado = []
        for r in rows:
            ciclos_por_hora = (
                round(3600 / r.duracion_promedio, 2)
                if r.duracion_promedio and r.duracion_promedio > 0
                else 0
            )
            resultado.append({
                "tipo_maquina":        r.tipo_maquina,
                "total_ciclos":        r.total_ciclos,
                "duracion_promedio_s": round(r.duracion_promedio or 0, 1),
                "ciclos_por_hora":     ciclos_por_hora,
            })

        return resultado
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS DEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

class TarjetaConteo(BaseModel):
    """Indicador numérico simple para una tarjeta del dashboard."""
    valor:     int | float
    etiqueta:  str
    tendencia: str | None = None   # "up" | "down" | "neutral"
    detalle:   str | None = None


class DistribucionItem(BaseModel):
    nombre: str
    valor:  int
    pct:    float   # 0-100


class JobsResumen(BaseModel):
    pendientes:  int
    procesando:  int
    completados: int
    fallidos:    int
    cancelados:  int
    total:       int


class VideosResumen(BaseModel):
    total:          int
    horas_totales:  float    # horas de video analizadas (completados)
    tamano_total_mb: float


class EventosDetectadosResumen(BaseModel):
    total:              int
    pendientes_revision: int
    confirmados:        int
    rechazados:         int
    por_categoria:      list[DistribucionItem]


class ServicioEstado(BaseModel):
    nombre:     str
    disponible: bool
    detalle:    str | None = None


class DashboardSummary(BaseModel):
    """
    Respuesta completa del dashboard — todos los indicadores en una sola llamada.
    Calculado en tiempo real desde la DB.
    """
    periodo:              str                         # "hoy" | "semana" | "mes" | "todo"
    videos:               VideosResumen
    jobs:                 JobsResumen
    eventos_detectados:   EventosDetectadosResumen
    ciclos:               list[dict]                  # estadísticas de ciclos
    productividad:        list[dict]                  # ciclos por hora
    top_equipos:          list[DistribucionItem]      # equipos con más análisis
    trabajos_recientes:   list[dict]                  # últimos 5 jobs
    reportes_recientes:   list[dict]                  # últimos 5 reportes
    servicios:            list[ServicioEstado]        # estado de servicios
    generado_en:          str                         # ISO 8601


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS NUEVOS
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(
    periodo: str = Query(
        "semana",
        description="Rango temporal: hoy | semana | mes | todo",
        pattern="^(hoy|semana|mes|todo)$",
    ),
    project_id:   str | None = Query(None, description="Filtrar por proyecto/mina"),
    equipment_id: str | None = Query(None, description="Filtrar por equipo/cámara"),
    db: Session = Depends(get_session_dep),
):
    """
    Indicadores completos para el dashboard operativo.
    Una sola llamada devuelve todo lo que necesita la pantalla principal.
    """
    from src.database.models import (
        Video, AnalysisJob, DetectedEvent, Report, Ciclo, Evento
    )

    # ── Calcular rango de fechas ───────────────────────────────────────────────
    ahora = datetime.now(timezone.utc)
    if periodo == "hoy":
        desde = ahora.replace(hour=0, minute=0, second=0, microsecond=0)
    elif periodo == "semana":
        desde = ahora - timedelta(days=7)
    elif periodo == "mes":
        desde = ahora - timedelta(days=30)
    else:   # "todo"
        desde = None

    def _filtro_fecha(q, col):
        if desde:
            return q.filter(col >= desde)
        return q

    # ── Videos ────────────────────────────────────────────────────────────────
    q_videos = db.query(Video).filter(Video.retention_status != "deleted")
    if project_id:
        q_videos = q_videos.filter(Video.project_id == project_id)
    if equipment_id:
        q_videos = q_videos.filter(Video.equipment_id == equipment_id)
    q_videos = _filtro_fecha(q_videos, Video.uploaded_at)

    total_videos   = q_videos.count()
    horas_video    = db.query(func.sum(Video.duration_seconds)).filter(
        Video.retention_status != "deleted"
    ).scalar() or 0
    tamano_total   = db.query(func.sum(Video.file_size_bytes)).filter(
        Video.retention_status != "deleted"
    ).scalar() or 0

    videos_resumen = VideosResumen(
        total           = total_videos,
        horas_totales   = round(horas_video / 3600, 2),
        tamano_total_mb = round(tamano_total / (1024 * 1024), 1),
    )

    # ── Jobs ──────────────────────────────────────────────────────────────────
    def _contar_jobs(status):
        q = db.query(func.count(AnalysisJob.id)).filter(AnalysisJob.status == status)
        q = _filtro_fecha(q, AnalysisJob.created_at)
        return q.scalar() or 0

    jobs_resumen = JobsResumen(
        pendientes  = _contar_jobs("queued"),
        procesando  = _contar_jobs("processing"),
        completados = _contar_jobs("completed"),
        fallidos    = _contar_jobs("failed"),
        cancelados  = _contar_jobs("cancelled"),
        total       = sum([
            _contar_jobs(s)
            for s in ("queued", "processing", "completed", "failed", "cancelled")
        ]),
    )

    # ── Eventos detectados ────────────────────────────────────────────────────
    q_ev = db.query(DetectedEvent)
    if desde:
        q_ev = q_ev.filter(DetectedEvent.created_at >= desde)

    total_ev        = q_ev.count()
    pendientes_rev  = q_ev.filter(DetectedEvent.review_status == "pending").count()
    confirmados     = q_ev.filter(DetectedEvent.review_status == "confirmed").count()
    rechazados      = q_ev.filter(DetectedEvent.review_status == "rejected").count()

    cat_rows = (
        db.query(DetectedEvent.category, func.count(DetectedEvent.id).label("n"))
        .group_by(DetectedEvent.category)
        .order_by(func.count(DetectedEvent.id).desc())
        .limit(10)
        .all()
    )
    total_cat = sum(r.n for r in cat_rows) or 1
    por_cat = [
        DistribucionItem(nombre=r.category, valor=r.n, pct=round(r.n / total_cat * 100, 1))
        for r in cat_rows
    ]

    eventos_resumen = EventosDetectadosResumen(
        total               = total_ev,
        pendientes_revision = pendientes_rev,
        confirmados         = confirmados,
        rechazados          = rechazados,
        por_categoria       = por_cat,
    )

    # ── Estadísticas de ciclos ────────────────────────────────────────────────
    repo_ciclos  = RepositorioCiclos(db)
    ciclos_stats = repo_ciclos.estadisticas()

    prod_rows = (
        db.query(
            Ciclo.tipo_maquina,
            func.count(Ciclo.id).label("total_ciclos"),
            func.avg(Ciclo.duracion_segundos).label("duracion_promedio"),
        )
        .filter(Ciclo.completado == True, Ciclo.duracion_segundos > 0)
        .group_by(Ciclo.tipo_maquina)
        .all()
    )
    productividad_data = [
        {
            "tipo_maquina":        r.tipo_maquina,
            "total_ciclos":        r.total_ciclos,
            "duracion_promedio_s": round(r.duracion_promedio or 0, 1),
            "ciclos_por_hora":     round(3600 / r.duracion_promedio, 2)
                                   if r.duracion_promedio and r.duracion_promedio > 0
                                   else 0,
        }
        for r in prod_rows
    ]

    # ── Top equipos por cantidad de análisis ──────────────────────────────────
    equipo_rows = (
        db.query(Video.equipment_id, func.count(AnalysisJob.id).label("n"))
        .join(AnalysisJob, AnalysisJob.video_id == Video.id)
        .filter(Video.equipment_id != None)
        .group_by(Video.equipment_id)
        .order_by(func.count(AnalysisJob.id).desc())
        .limit(10)
        .all()
    )
    total_eq = sum(r.n for r in equipo_rows) or 1
    top_equipos = [
        DistribucionItem(
            nombre = r.equipment_id,
            valor  = r.n,
            pct    = round(r.n / total_eq * 100, 1),
        )
        for r in equipo_rows
    ]

    # ── Últimos 5 jobs ─────────────────────────────────────────────────────────
    ultimos_jobs_db = (
        db.query(AnalysisJob)
        .order_by(AnalysisJob.created_at.desc())
        .limit(5)
        .all()
    )
    trabajos_recientes = []
    for j in ultimos_jobs_db:
        v = db.get(Video, j.video_id)
        dur = None
        if j.started_at and j.finished_at:
            dur = round((j.finished_at - j.started_at).total_seconds(), 1)
        trabajos_recientes.append({
            "job_id":       j.id,
            "video":        v.original_filename if v else "?",
            "estado":       j.status,
            "modelo":       j.model_id or "",
            "creado_en":    j.created_at.isoformat() if j.created_at else "",
            "duracion_seg": dur,
        })

    # ── Últimos 5 reportes ────────────────────────────────────────────────────
    ultimos_reports_db = (
        db.query(Report)
        .order_by(Report.created_at.desc())
        .limit(5)
        .all()
    )
    reportes_recientes = [
        {
            "report_id":  r.id,
            "job_id":     r.analysis_job_id,
            "formato":    r.format,
            "estado":     r.status,
            "creado_en":  r.created_at.isoformat() if r.created_at else "",
        }
        for r in ultimos_reports_db
    ]

    # ── Estado de servicios ───────────────────────────────────────────────────
    servicios = _verificar_servicios(db)

    return DashboardSummary(
        periodo             = periodo,
        videos              = videos_resumen,
        jobs                = jobs_resumen,
        eventos_detectados  = eventos_resumen,
        ciclos              = ciclos_stats,
        productividad       = productividad_data,
        top_equipos         = top_equipos,
        trabajos_recientes  = trabajos_recientes,
        reportes_recientes  = reportes_recientes,
        servicios           = servicios,
        generado_en         = ahora.isoformat(),
    )


@router.get("/estadisticas/jobs")
def estadisticas_jobs(
    periodo: str = Query("semana", pattern="^(hoy|semana|mes|todo)$"),
    db: Session = Depends(get_session_dep),
):
    """Distribución de jobs por estado, con tiempos promedio de procesamiento."""
    from src.database.models import AnalysisJob

    ahora = datetime.now(timezone.utc)
    desde = _calcular_desde(periodo, ahora)

    q = db.query(AnalysisJob)
    if desde:
        q = q.filter(AnalysisJob.created_at >= desde)

    filas = (
        q.with_entities(
            AnalysisJob.status,
            func.count(AnalysisJob.id).label("total"),
        )
        .group_by(AnalysisJob.status)
        .all()
    )

    # Tiempo promedio de procesamiento de los completados
    completados = (
        db.query(
            func.avg(
                func.julianday(AnalysisJob.finished_at) -
                func.julianday(AnalysisJob.started_at)
            ).label("avg_dias")
        )
        .filter(
            AnalysisJob.status == "completed",
            AnalysisJob.started_at != None,
            AnalysisJob.finished_at != None,
        )
        .scalar()
    )
    tiempo_prom_min = round((completados or 0) * 24 * 60, 1)

    return {
        "periodo":             periodo,
        "distribucion":        [{"status": r.status, "total": r.total} for r in filas],
        "tiempo_promedio_min": tiempo_prom_min,
    }


@router.get("/estadisticas/eventos-detectados")
def estadisticas_eventos_detectados(
    periodo: str = Query("semana", pattern="^(hoy|semana|mes|todo)$"),
    db: Session = Depends(get_session_dep),
):
    """Distribución de eventos detectados por categoría y estado de revisión."""
    from src.database.models import DetectedEvent

    ahora = datetime.now(timezone.utc)
    desde = _calcular_desde(periodo, ahora)

    q = db.query(DetectedEvent)
    if desde:
        q = q.filter(DetectedEvent.created_at >= desde)

    por_cat = (
        q.with_entities(
            DetectedEvent.category,
            func.count(DetectedEvent.id).label("total"),
        )
        .group_by(DetectedEvent.category)
        .order_by(func.count(DetectedEvent.id).desc())
        .all()
    )

    por_revision = (
        q.with_entities(
            DetectedEvent.review_status,
            func.count(DetectedEvent.id).label("total"),
        )
        .group_by(DetectedEvent.review_status)
        .all()
    )

    return {
        "periodo":      periodo,
        "total":        q.count(),
        "por_categoria": [{"categoria": r.category, "total": r.total} for r in por_cat],
        "por_revision":  [{"estado": r.review_status, "total": r.total} for r in por_revision],
    }


@router.get("/estadisticas/videos")
def estadisticas_videos(
    periodo: str = Query("todo", pattern="^(hoy|semana|mes|todo)$"),
    db: Session = Depends(get_session_dep),
):
    """Métricas de videos: total, horas, tamaño, distribución por proyecto."""
    from src.database.models import Video

    ahora = datetime.now(timezone.utc)
    desde = _calcular_desde(periodo, ahora)

    q = db.query(Video).filter(Video.retention_status != "deleted")
    if desde:
        q = q.filter(Video.uploaded_at >= desde)

    total          = q.count()
    horas_totales  = (db.query(func.sum(Video.duration_seconds)).scalar() or 0) / 3600
    tamano_total   = (db.query(func.sum(Video.file_size_bytes)).scalar() or 0) / (1024**3)

    por_proyecto = (
        q.with_entities(
            Video.project_id,
            func.count(Video.id).label("total"),
        )
        .filter(Video.project_id != None)
        .group_by(Video.project_id)
        .order_by(func.count(Video.id).desc())
        .all()
    )

    return {
        "periodo":        periodo,
        "total":          total,
        "horas_totales":  round(horas_totales, 2),
        "tamano_total_gb": round(tamano_total, 3),
        "por_proyecto":   [{"proyecto": r.project_id, "total": r.total} for r in por_proyecto],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _calcular_desde(periodo: str, ahora: datetime) -> datetime | None:
    if periodo == "hoy":
        return ahora.replace(hour=0, minute=0, second=0, microsecond=0)
    elif periodo == "semana":
        return ahora - timedelta(days=7)
    elif periodo == "mes":
        return ahora - timedelta(days=30)
    return None


def _verificar_servicios(db: Session) -> list[ServicioEstado]:
    """Verifica el estado de los servicios del sistema."""
    servicios = []

    # ── DB ────────────────────────────────────────────────────────────────────
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        servicios.append(ServicioEstado(nombre="base_de_datos", disponible=True))
    except Exception as e:
        servicios.append(ServicioEstado(nombre="base_de_datos", disponible=False, detalle=str(e)))

    # ── Modelo YOLO ────────────────────────────────────────────────────────────
    try:
        import os
        from src.config import load_config
        config      = load_config()
        modelo_ok   = os.path.exists(config.model_path)
        model_name  = os.path.basename(config.model_path)
        servicios.append(ServicioEstado(
            nombre      = "modelo_yolo",
            disponible  = modelo_ok,
            detalle     = model_name if modelo_ok else f"No encontrado: {config.model_path}",
        ))
    except Exception as e:
        servicios.append(ServicioEstado(nombre="modelo_yolo", disponible=False, detalle=str(e)))

    # ── Jobs en curso ──────────────────────────────────────────────────────────
    try:
        from src.api.routers.video import _RUNTIME, _RUNTIME_LOCK
        with _RUNTIME_LOCK:
            n_activos = len(_RUNTIME)
        servicios.append(ServicioEstado(
            nombre     = "jobs_activos",
            disponible = True,
            detalle    = f"{n_activos} job(s) corriendo en este momento",
        ))
    except Exception as e:
        servicios.append(ServicioEstado(nombre="jobs_activos", disponible=False, detalle=str(e)))

    # ── Almacenamiento ────────────────────────────────────────────────────────
    try:
        import shutil
        total, usado, libre = shutil.disk_usage(".")
        libre_gb = round(libre / (1024 ** 3), 1)
        servicios.append(ServicioEstado(
            nombre     = "almacenamiento",
            disponible = libre_gb > 1.0,
            detalle    = f"{libre_gb} GB libres",
        ))
    except Exception as e:
        servicios.append(ServicioEstado(nombre="almacenamiento", disponible=False, detalle=str(e)))

    return servicios
