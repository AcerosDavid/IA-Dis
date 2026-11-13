"""
Router de Reportes — generación y descarga de reportes de análisis.

Paso 5: Genera PDF, XLSX y CSV a partir de los resultados persistidos.
La generación es asíncrona si el job de análisis tiene muchos eventos.

Endpoints:
  POST   /analysis-jobs/{job_id}/reports  — solicitar generación de reporte
  GET    /reports                         — listar reportes con filtros
  GET    /reports/{id}                    — estado y metadatos
  GET    /reports/{id}/download           — descargar el archivo generado
"""
from __future__ import annotations

import csv
import io
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.api.schemas import ReportOut, ReportCreate, PaginacionMeta
from src.database.db import get_session_dep, RepositorioAuditoria, get_session
from src.database.models import AnalysisJob, Video, Report, DetectedEvent

router = APIRouter(tags=["Reportes"])

# Directorio donde se guardan los reportes generados
REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def _report_to_out(r: Report) -> ReportOut:
    return ReportOut(
        id              = r.id,
        analysis_job_id = r.analysis_job_id,
        format          = r.format,
        status          = r.status,
        storage_key     = r.storage_key,
        created_by      = r.created_by,
        created_at      = r.created_at.isoformat() if r.created_at else "",
        finished_at     = r.finished_at.isoformat() if r.finished_at else None,
        error_message   = r.error_message,
        template_version = r.template_version,
    )


# ── Generador de reportes (en background) ─────────────────────────────────────

def _generar_reporte(report_db_id: int) -> None:
    """
    Genera el archivo del reporte en background.
    Lee los datos del job y eventos desde la DB y produce el archivo solicitado.
    """
    with get_session() as db:
        r = db.get(Report, report_db_id)
        if r is None:
            return

        job    = db.get(AnalysisJob, r.analysis_job_id)
        video  = db.get(Video, job.video_id) if job else None

        if not job or not video:
            r.status        = "failed"
            r.error_message = "Job o video no encontrado"
            r.finished_at   = datetime.now(timezone.utc)
            return

        eventos = (
            db.query(DetectedEvent)
            .filter(DetectedEvent.analysis_job_id == job.id)
            .order_by(DetectedEvent.start_time_seconds.asc())
            .all()
        )

        # Nombre del archivo
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext  = r.format
        nombre = f"reporte_job{job.id}_{ts}.{ext}"
        ruta   = REPORTS_DIR / nombre

        try:
            if r.format == "csv":
                _generar_csv(ruta, job, video, eventos)
            elif r.format == "xlsx":
                _generar_xlsx(ruta, job, video, eventos)
            elif r.format == "pdf":
                _generar_pdf(ruta, job, video, eventos)

            if not ruta.is_file() or ruta.stat().st_size == 0:
                raise RuntimeError(
                    f"El reporte no se generó correctamente ({ruta.name} vacío o ausente)"
                )

            r.storage_key     = str(ruta)
            r.status          = "completed"
            r.finished_at     = datetime.now(timezone.utc)
            r.template_version = "1.0"

        except Exception as e:
            r.status        = "failed"
            r.error_message = str(e)
            r.finished_at   = datetime.now(timezone.utc)
            print(f"[REPORTE] Error generando reporte {report_db_id}: {e}")


def _generar_csv(ruta: Path, job: AnalysisJob, video: Video, eventos: list) -> None:
    """CSV simple — compatible con Excel y cualquier hoja de cálculo."""
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # Encabezado del reporte
        writer.writerow(["REPORTE DE ANÁLISIS — Mining Dispatch"])
        writer.writerow([])
        writer.writerow(["Video",          video.original_filename])
        writer.writerow(["Job ID",         job.id])
        writer.writerow(["Estado",         job.status])
        writer.writerow(["Modelo",         job.model_id or ""])
        writer.writerow(["Inicio",         job.started_at.isoformat() if job.started_at else ""])
        writer.writerow(["Fin",            job.finished_at.isoformat() if job.finished_at else ""])
        writer.writerow(["Duración video", f"{video.duration_seconds or 0:.1f}s"])
        writer.writerow([])

        # Métricas
        metrics = job.metrics_json or {}
        writer.writerow(["Eventos detectados", metrics.get("eventos_nuevos", len(eventos))])
        writer.writerow(["Ciclos detectados",  metrics.get("ciclos_nuevos", 0)])
        writer.writerow([])

        # Tabla de eventos
        writer.writerow([
            "ID", "Categoría", "Descripción",
            "Inicio (s)", "Fin (s)", "Confianza",
            "Severidad", "Estado revisión", "Revisado por", "Comentario",
        ])
        for e in eventos:
            writer.writerow([
                e.id,
                e.category,
                e.description or "",
                round(e.start_time_seconds, 2),
                round(e.end_time_seconds, 2) if e.end_time_seconds else "",
                round(e.confidence, 3) if e.confidence else "",
                e.severity or "",
                e.review_status,
                e.reviewed_by or "",
                e.review_comment or "",
            ])


def _generar_xlsx(ruta: Path, job: AnalysisJob, video: Video, eventos: list) -> None:
    """XLSX con openpyxl — incluye hoja de resumen y hoja de eventos."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError as e:
        # Antes se generaba un CSV renombrado a .xlsx: el archivo resultante
        # era inválido pero el reporte quedaba "completed". Mejor fallar claro.
        raise RuntimeError(
            "openpyxl no está instalado: no se puede generar XLSX. "
            "Instala con: pip install openpyxl"
        ) from e

    wb = openpyxl.Workbook()

    # ── Hoja 1: Resumen ───────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Resumen"

    encabezado_fill = PatternFill("solid", fgColor="1F3864")
    encabezado_font = Font(color="FFFFFF", bold=True, size=11)

    ws1.append(["REPORTE DE ANÁLISIS — Mining Dispatch"])
    ws1["A1"].font = Font(bold=True, size=14)

    ws1.append([])
    ws1.append(["Campo", "Valor"])
    for c in ws1["A3:B3"][0]:
        c.font = encabezado_font
        c.fill = encabezado_fill

    metrics = job.metrics_json or {}
    filas_resumen = [
        ("Video",              video.original_filename),
        ("Job ID",             job.id),
        ("Estado",             job.status),
        ("Modelo",             job.model_id or ""),
        ("Versión",            job.model_version or ""),
        ("Proyecto",           video.project_id or ""),
        ("Equipo/Cámara",      video.equipment_id or ""),
        ("Duración video (s)", video.duration_seconds or ""),
        ("Inicio análisis",    job.started_at.isoformat() if job.started_at else ""),
        ("Fin análisis",       job.finished_at.isoformat() if job.finished_at else ""),
        ("Eventos detectados", metrics.get("eventos_nuevos", len(eventos))),
        ("Ciclos detectados",  metrics.get("ciclos_nuevos", 0)),
        ("Generado el",        datetime.now(timezone.utc).isoformat()),
    ]
    for fila in filas_resumen:
        ws1.append(list(fila))

    ws1.column_dimensions["A"].width = 25
    ws1.column_dimensions["B"].width = 45

    # ── Hoja 2: Eventos ───────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Eventos")
    headers = [
        "ID", "Categoría", "Descripción",
        "Inicio (s)", "Fin (s)", "Confianza",
        "Severidad", "Revisión", "Revisado por", "Comentario",
    ]
    ws2.append(headers)
    for c in ws2[1]:
        c.font = encabezado_font
        c.fill = encabezado_fill
        c.alignment = Alignment(horizontal="center")

    colores_revision = {
        "confirmed": "C6EFCE",
        "rejected":  "FFC7CE",
        "corrected": "FFEB9C",
        "follow_up": "DDEBF7",
        "pending":   "F2F2F2",
    }
    for e in eventos:
        fila = [
            e.id, e.category, e.description or "",
            round(e.start_time_seconds, 2),
            round(e.end_time_seconds, 2) if e.end_time_seconds else "",
            round(e.confidence, 3) if e.confidence else "",
            e.severity or "",
            e.review_status,
            e.reviewed_by or "",
            e.review_comment or "",
        ]
        ws2.append(fila)
        # Color de fila según estado de revisión
        color = colores_revision.get(e.review_status, "F2F2F2")
        for cell in ws2[ws2.max_row]:
            cell.fill = PatternFill("solid", fgColor=color)

    anchos = [8, 20, 35, 12, 12, 12, 12, 15, 18, 35]
    for i, ancho in enumerate(anchos, 1):
        ws2.column_dimensions[openpyxl.utils.get_column_letter(i)].width = ancho

    wb.save(ruta)


def _generar_pdf(ruta: Path, job: AnalysisJob, video: Video, eventos: list) -> None:
    """
    PDF básico con reportlab.
    Si reportlab no está disponible, genera un PDF mínimo en texto plano
    que la mayoría de visores pueden abrir.
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle,
            Paragraph, Spacer,
        )

        doc    = SimpleDocTemplate(str(ruta), pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        story  = []

        # Título
        story.append(Paragraph("Reporte de Análisis — Mining Dispatch", styles["h1"]))
        story.append(Spacer(1, 12))

        # Resumen
        metrics = job.metrics_json or {}
        resumen_data = [
            ["Campo", "Valor"],
            ["Video",              video.original_filename],
            ["Job ID",             str(job.id)],
            ["Estado",             job.status],
            ["Modelo",             job.model_id or ""],
            ["Duración video",     f"{video.duration_seconds or 0:.1f}s"],
            ["Eventos detectados", str(metrics.get("eventos_nuevos", len(eventos)))],
            ["Ciclos detectados",  str(metrics.get("ciclos_nuevos", 0))],
            ["Generado el",        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
        ]
        tabla_res = Table(resumen_data, colWidths=[150, 300])
        tabla_res.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#E8F0FE")),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FF")]),
        ]))
        story.append(tabla_res)
        story.append(Spacer(1, 20))

        # Tabla de eventos
        story.append(Paragraph("Eventos Detectados", styles["h2"]))
        story.append(Spacer(1, 8))

        headers = ["ID", "Categoría", "Inicio (s)", "Confianza", "Revisión", "Comentario"]
        ev_data = [headers]
        for e in eventos:
            ev_data.append([
                str(e.id),
                e.category,
                f"{e.start_time_seconds:.1f}",
                f"{e.confidence:.2f}" if e.confidence else "-",
                e.review_status,
                (e.review_comment or "")[:40],
            ])

        tabla_ev = Table(
            ev_data,
            colWidths=[40, 100, 70, 70, 80, 150],
            repeatRows=1,
        )

        color_estados = {
            "confirmed": colors.HexColor("#C6EFCE"),
            "rejected":  colors.HexColor("#FFC7CE"),
            "corrected": colors.HexColor("#FFEB9C"),
            "follow_up": colors.HexColor("#DDEBF7"),
            "pending":   colors.HexColor("#F2F2F2"),
        }
        estilo_ev = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("GRID",       (0, 0), (-1, -1), 0.3, colors.grey),
        ]
        for idx, e in enumerate(eventos, start=1):
            color = color_estados.get(e.review_status, colors.white)
            estilo_ev.append(("BACKGROUND", (0, idx), (-1, idx), color))

        tabla_ev.setStyle(TableStyle(estilo_ev))
        story.append(tabla_ev)

        doc.build(story)

    except ImportError as e:
        # Antes se renombraba un CSV a .pdf (archivo corrupto marcado como OK).
        raise RuntimeError(
            "reportlab no está instalado: no se puede generar PDF. "
            "Instala con: pip install reportlab"
        ) from e


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analysis-jobs/{job_id}/reports", response_model=ReportOut, status_code=202)
def solicitar_reporte(
    job_id:  int,
    payload: ReportCreate,
    db: Session = Depends(get_session_dep),
):
    """
    Solicita la generación de un reporte para un job completado.
    La generación corre en background. Sondear GET /reports/{id}.
    """
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    if job.status not in ("completed",):
        raise HTTPException(
            409,
            f"Solo se pueden generar reportes de jobs completados (actual: {job.status})",
        )

    # Crear registro del reporte
    report = Report(
        analysis_job_id  = job_id,
        format           = payload.format,
        status           = "generating",
    )
    db.add(report)
    db.flush()
    report_id = report.id

    # Auditoría
    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action="request_report", resource_type="report",
        resource_id=str(report_id),
        context_json={"format": payload.format, "job_id": job_id},
    )

    # COMMIT antes de lanzar el hilo: si el worker consulta la DB antes de que
    # la transacción del endpoint se confirme, no encuentra el reporte.
    db.commit()
    db.refresh(report)

    # Lanzar generación en background
    t = threading.Thread(target=_generar_reporte, args=(report_id,), daemon=True)
    t.start()

    return _report_to_out(report)


@router.get("/reports", response_model=list[ReportOut])
def listar_reportes(
    job_id: int | None = Query(None, description="Filtrar por job"),
    status: str | None = Query(None, description="Filtrar por estado"),
    db: Session = Depends(get_session_dep),
):
    """Lista reportes generados."""
    q = db.query(Report)
    if job_id:
        q = q.filter(Report.analysis_job_id == job_id)
    if status:
        q = q.filter(Report.status == status)
    return [_report_to_out(r) for r in q.order_by(Report.created_at.desc()).all()]


@router.get("/reports/{report_id}", response_model=ReportOut)
def obtener_reporte(report_id: int, db: Session = Depends(get_session_dep)):
    """Retorna estado y metadatos de un reporte."""
    r = db.get(Report, report_id)
    if r is None:
        raise HTTPException(404, "Reporte no encontrado")
    return _report_to_out(r)


@router.get("/reports/{report_id}/download")
def descargar_reporte(report_id: int, db: Session = Depends(get_session_dep)):
    """
    Descarga el archivo del reporte.
    Solo disponible cuando status == 'completed'.
    """
    r = db.get(Report, report_id)
    if r is None:
        raise HTTPException(404, "Reporte no encontrado")

    if r.status == "generating":
        raise HTTPException(202, "El reporte todavía se está generando. Intenta en unos segundos.")
    if r.status == "failed":
        raise HTTPException(500, f"El reporte falló: {r.error_message}")
    if r.status != "completed":
        raise HTTPException(409, f"Reporte en estado '{r.status}' — no disponible")
    if not r.storage_key or not os.path.exists(r.storage_key):
        raise HTTPException(404, "Archivo de reporte no encontrado en disco")

    # Auditoría de descarga
    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action="download_report", resource_type="report",
        resource_id=str(report_id), result="success",
    )

    # Determinar Content-Type
    media_types = {
        "csv":  "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf":  "application/pdf",
    }
    media_type = media_types.get(r.format, "application/octet-stream")
    filename   = os.path.basename(r.storage_key)

    def _stream():
        with open(r.storage_key, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        _stream(),
        media_type  = media_type,
        headers     = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )
