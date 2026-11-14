"""
Generación de reportes PDF y Excel a partir de datos reales del análisis.

No genera datos ficticios: todos los números provienen de la base de datos
o del Procesador tras un análisis completado.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

from src.database.db import get_session, migrate
from src.database.models import Evento


def _fmt_segundos(seg: float) -> str:
    """Convierte segundos a 'HH:MM:SS'."""
    if not seg or seg < 0:
        return "00:00:00"
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _obtener_detecciones_db() -> list[dict]:
    """Obtiene las detecciones de la DB para el reporte."""
    try:
        with get_session() as session:
            eventos = (
                session.query(Evento)
                .order_by(Evento.numero_fotograma.asc())
                .limit(500)
                .all()
            )
            return [
                {
                    "id": e.id,
                    "timestamp": e.timestamp,
                    "tipo_maquina": e.tipo_maquina,
                    "clase_id": e.clase_id,
                    "confianza": e.confianza,
                    "numero_fotograma": e.numero_fotograma,
                    "tipo_evento_ciclo": e.tipo_evento_ciclo or "",
                    "zona_nombre": e.zona_nombre or "",
                }
                for e in eventos
            ]
    except Exception as e:
        print(f"[REPORTES] Error leyendo detecciones: {e}")
        return []


def _obtener_estadisticas_por_categoria() -> list[dict]:
    """Detecciones agrupadas por tipo de máquina."""
    from sqlalchemy import func
    try:
        with get_session() as session:
            rows = (
                session.query(
                    Evento.tipo_maquina,
                    func.count(Evento.id).label("total"),
                    func.avg(Evento.confianza).label("conf_promedio"),
                )
                .group_by(Evento.tipo_maquina)
                .all()
            )
            return [
                {
                    "tipo": r.tipo_maquina,
                    "total": int(r.total),
                    "conf_promedio": round(float(r.conf_promedio or 0), 3),
                }
                for r in rows
            ]
    except Exception:
        return []


# ── PDF con reportlab ──────────────────────────────────────────────────────────

def generar_reporte_pdf(
    ruta: str,
    job_data: dict,
    stats: dict,
    eventos_contador: dict,
    elapsed: float,
) -> str:
    """Genera un reporte PDF con los resultados del análisis."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )

    Path(os.path.dirname(ruta)).mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        ruta, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title=f"Reporte - {job_data['nombre']}",
    )

    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle(
        "TituloReporte", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#0a0e1a"),
        spaceAfter=10,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"], fontSize=9,
        textColor=colors.grey, spaceAfter=16,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#0088aa"), spaceAfter=8, spaceBefore=14,
    )
    normal_style = styles["Normal"]

    elementos = []

    # Título
    elementos.append(Paragraph("MINING DISPATCH — Reporte de Análisis", titulo_style))
    elementos.append(Paragraph(
        f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sub_style,
    ))
    elementos.append(Spacer(1, 10))

    # ── Información general ──────────────────────────────────────────────────
    elementos.append(Paragraph("1. Información General", h2_style))
    info_general = [
        ["Video analizado", job_data.get("nombre", "—")],
        ["Fecha de análisis", job_data.get("creado_en", "—")],
        ["Duración original", _fmt_segundos(job_data.get("duracion_seg", 0))],
        ["Tiempo de procesamiento", _fmt_segundos(elapsed)],
        ["Modelo IA", job_data.get("modelo", "—")],
        ["Dispositivo", job_data.get("dispositivo", "—")],
        ["Resolución", job_data.get("resolucion", "—")],
        ["FPS originales", f"{job_data.get('fps_originales', 0):.1f}"],
        ["Frames analizados", f"{stats.get('total_detecciones', 0)}"],
    ]
    t = Table(info_general, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0088aa")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c0c8d0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elementos.append(t)
    elementos.append(Spacer(1, 14))

    # ── Estadísticas de detección ──────────────────────────────────────────────
    elementos.append(Paragraph("2. Estadísticas de Detección", h2_style))
    stats_data = [
        ["Métrica", "Valor"],
        ["Total de detecciones", str(stats.get("total_detecciones", 0))],
        ["Total de eventos", str(stats.get("total_eventos", 0))],
        ["Tipos de máquina detectados", str(stats.get("total_unicos", 0))],
        ["Confianza promedio", f"{stats.get('confianza_promedio', 0):.3f}"],
    ]
    t2 = Table(stats_data, colWidths=[8*cm, 8*cm])
    t2.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00d4ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c0c8d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elementos.append(t2)
    elementos.append(Spacer(1, 10))

    # ── Por categoría ──────────────────────────────────────────────────────────
    por_categoria = _obtener_estadisticas_por_categoria()
    if por_categoria:
        elementos.append(Paragraph("3. Detecciones por Categoría", h2_style))
        cat_data = [["Tipo de Máquina", "Total", "Confianza Promedio"]]
        for c in por_categoria:
            cat_data.append([
                c["tipo"], str(c["total"]), f"{c['conf_promedio']:.3f}"
            ])
        t3 = Table(cat_data, colWidths=[6*cm, 4*cm, 6*cm])
        t3.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00d4ff")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c0c8d0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        elementos.append(t3)
        elementos.append(Spacer(1, 14))

    # ── Eventos ──────────────────────────────────────────────────────────────────
    elementos.append(Paragraph("4. Estadísticas de Eventos", h2_style))
    ev_data = [
        ["Tipo de Evento", "Cantidad"],
        ["Apariciones", str(eventos_contador.get("aparicion", 0))],
        ["Desapariciones", str(eventos_contador.get("desaparicion", 0))],
        ["Eventos de zona", str(eventos_contador.get("zona", 0))],
    ]
    t4 = Table(ev_data, colWidths=[8*cm, 8*cm])
    t4.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00d4ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c0c8d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    elementos.append(t4)
    elementos.append(Spacer(1, 14))

    # ── Configuración utilizada ──────────────────────────────────────────────
    elementos.append(Paragraph("5. Configuración del Análisis", h2_style))
    config_data = [
        ["Parámetro", "Valor"],
        ["Umbral de confianza", f"{job_data.get('confianza_umbral', 0):.2f}"],
        ["Skip de frames", str(job_data.get("track_skip", 3))],
        ["Resolución de inferencia", str(job_data.get("track_imgsz", 320))],
    ]
    t5 = Table(config_data, colWidths=[8*cm, 8*cm])
    t5.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00d4ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c0c8d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    elementos.append(t5)

    doc.build(elementos)
    print(f"[REPORTES] PDF generado: {ruta}")
    return ruta


# ── Excel con openpyxl ─────────────────────────────────────────────────────────

def generar_reporte_excel(
    ruta: str,
    job_data: dict,
    stats: dict,
    eventos_contador: dict,
) -> str:
    """Genera un reporte Excel con los datos del análisis."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    Path(os.path.dirname(ruta)).mkdir(parents=True, exist_ok=True)

    wb = Workbook()

    # Estilos
    header_font = Font(name="Consolas", bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="0088AA")
    normal_font = Font(name="Consolas", size=10)
    bold_font = Font(name="Consolas", size=10, bold=True)
    thin_border = Border(
        left=Side(style="thin", color="C0C8D0"),
        right=Side(style="thin", color="C0C8D0"),
        top=Side(style="thin", color="C0C8D0"),
        bottom=Side(style="thin", color="C0C8D0"),
    )

    # ── Hoja 1: Resumen ───────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Resumen"

    ws1["A1"] = "MINING DISPATCH — Reporte de Análisis"
    ws1["A1"].font = Font(name="Consolas", bold=True, size=14, color="0088AA")
    ws1.merge_cells("A1:C1")
    ws1["A2"] = f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws1["A2"].font = Font(name="Consolas", size=9, color="666666")
    ws1.merge_cells("A2:C2")

    # Información general
    info_rows = [
        ("Video analizado", job_data.get("nombre", "—")),
        ("Fecha de análisis", job_data.get("creado_en", "—")),
        ("Duración original (s)", round(job_data.get("duracion_seg", 0), 1)),
        ("Resolución", job_data.get("resolucion", "—")),
        ("FPS originales", round(job_data.get("fps_originales", 0), 1)),
        ("Modelo IA", job_data.get("modelo", "—")),
        ("Dispositivo", job_data.get("dispositivo", "—")),
        ("Umbral de confianza", job_data.get("confianza_umbral", 0)),
        ("Skip de frames", job_data.get("track_skip", 3)),
        ("Resolución de inferencia", job_data.get("track_imgsz", 320)),
    ]
    start_row = 4
    ws1[f"A{start_row}"] = "Parámetro"
    ws1[f"B{start_row}"] = "Valor"
    for cell in (ws1[f"A{start_row}"], ws1[f"B{start_row}"]):
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="left")

    for i, (k, v) in enumerate(info_rows, start=start_row + 1):
        ws1[f"A{i}"] = k
        ws1[f"B{i}"] = v
        ws1[f"A{i}"].font = bold_font
        ws1[f"B{i}"].font = normal_font
        ws1[f"A{i}"].border = thin_border
        ws1[f"B{i}"].border = thin_border

    # Estadísticas
    stats_start = start_row + len(info_rows) + 3
    ws1[f"A{stats_start}"] = "Estadísticas de Detección"
    ws1[f"A{stats_start}"].font = Font(name="Consolas", bold=True, size=12, color="0088AA")
    ws1.merge_cells(f"A{stats_start}:B{stats_start}")

    stat_rows = [
        ("Total de detecciones", stats.get("total_detecciones", 0)),
        ("Total de eventos", stats.get("total_eventos", 0)),
        ("Tipos de máquina detectados", stats.get("total_unicos", 0)),
        ("Confianza promedio", round(stats.get("confianza_promedio", 0), 3)),
        ("Apariciones", eventos_contador.get("aparicion", 0)),
        ("Desapariciones", eventos_contador.get("desaparicion", 0)),
        ("Eventos de zona", eventos_contador.get("zona", 0)),
    ]
    ws1[f"A{stats_start+1}"] = "Métrica"
    ws1[f"B{stats_start+1}"] = "Valor"
    for cell in (ws1[f"A{stats_start+1}"], ws1[f"B{stats_start+1}"]):
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border

    for i, (k, v) in enumerate(stat_rows, start=stats_start + 2):
        ws1[f"A{i}"] = k
        ws1[f"B{i}"] = v
        ws1[f"A{i}"].font = bold_font
        ws1[f"B{i}"].font = normal_font
        ws1[f"A{i}"].border = thin_border
        ws1[f"B{i}"].border = thin_border

    # ── Hoja 2: Detecciones por categoría ──────────────────────────────────────
    ws2 = wb.create_sheet("Por Categoría")
    ws2["A1"] = "Detecciones por Tipo de Máquina"
    ws2["A1"].font = Font(name="Consolas", bold=True, size=12, color="0088AA")
    ws2.merge_cells("A1:C1")

    headers = ["Tipo de Máquina", "Total", "Confianza Promedio"]
    for col, h in enumerate(headers, 1):
        cell = ws2.cell(row=3, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border

    por_categoria = _obtener_estadisticas_por_categoria()
    for i, c in enumerate(por_categoria, start=4):
        ws2.cell(row=i, column=1, value=c["tipo"]).font = normal_font
        ws2.cell(row=i, column=2, value=c["total"]).font = normal_font
        ws2.cell(row=i, column=3, value=round(c["conf_promedio"], 3)).font = normal_font
        for col in range(1, 4):
            ws2.cell(row=i, column=col).border = thin_border

    # ── Hoja 3: Eventos detallados ──────────────────────────────────────────────
    ws3 = wb.create_sheet("Eventos")
    ws3["A1"] = "Eventos Registrados (últimos 500)"
    ws3["A1"].font = Font(name="Consolas", bold=True, size=12, color="0088AA")
    ws3.merge_cells("A1:G1")

    headers_ev = ["ID", "Timestamp", "Tipo Máquina", "Clase", "Confianza", "Fotograma", "Tipo Evento"]
    for col, h in enumerate(headers_ev, 1):
        cell = ws3.cell(row=3, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border

    detecciones = _obtener_detecciones_db()
    for i, d in enumerate(detecciones, start=4):
        ws3.cell(row=i, column=1, value=d["id"]).font = normal_font
        ws3.cell(row=i, column=2, value=d["timestamp"]).font = normal_font
        ws3.cell(row=i, column=3, value=d["tipo_maquina"]).font = normal_font
        ws3.cell(row=i, column=4, value=d["clase_id"]).font = normal_font
        ws3.cell(row=i, column=5, value=round(d["confianza"], 3)).font = normal_font
        ws3.cell(row=i, column=6, value=d["numero_fotograma"]).font = normal_font
        ws3.cell(row=i, column=7, value=d["tipo_evento_ciclo"]).font = normal_font
        for col in range(1, 8):
            ws3.cell(row=i, column=col).border = thin_border

    # Ancho de columnas en todas las hojas
    for ws in (ws1, ws2, ws3):
        for col in range(1, ws.max_column + 1):
            max_len = 10
            for row in range(1, ws.max_row + 1):
                val = ws.cell(row=row, column=col).value
                if val and len(str(val)) > max_len:
                    max_len = len(str(val))
            ws.column_dimensions[get_column_letter(col)].width = min(max_len + 3, 40)

    wb.save(ruta)
    print(f"[REPORTES] Excel generado: {ruta}")
    return ruta
