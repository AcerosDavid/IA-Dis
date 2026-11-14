"""
Tests del pipeline reutilizable: timestamps, parseo de eventos y dibujo.
No cargan modelos ni abren ventanas.
"""
from __future__ import annotations

import numpy as np

from src.core.drawing import escalar_frame
from src.pipeline.presencia import calcular_timestamp
from src.pipeline.runner import OpcionesAnalisis, _evento_desde_mensaje


# ─── Timestamps ──────────────────────────────────────────────────────────────

def test_calcular_timestamp_formato():
    assert calcular_timestamp(fps=30.0, frame_number=90) == "00:00:03.000"


def test_calcular_timestamp_fps_cero():
    assert calcular_timestamp(fps=0.0, frame_number=10) == "00:00:00.000"


def test_calcular_timestamp_horas():
    assert calcular_timestamp(fps=1.0, frame_number=3661) == "01:01:01.000"


# ─── Parseo de mensajes de presencia ────────────────────────────────────────

def test_evento_aparicion():
    msg = (
        "[APARICION]    id=7 | maquina=camion | clase=dump_truck "
        "| frame=10 | ts=00:00:01.000 | conf=0.91"
    )
    ev = _evento_desde_mensaje(msg, "00:00:01.000")
    assert ev is not None
    assert ev.tipo == "APARICION"
    assert ev.maquina == "camion"
    assert ev.track_id == 7
    assert ev.confianza == 0.91


def test_evento_desaparicion():
    msg = "[DESAPARICION] id=3 | maquina=pala | frames=120"
    ev = _evento_desde_mensaje(msg, "00:01:00.000")
    assert ev is not None
    assert ev.tipo == "DESAPARICION"
    assert ev.maquina == "pala"
    assert ev.track_id == 3


def test_evento_zona():
    msg = "[ZONA] id=5 | entrada_carga -> zona_carga_norte"
    ev = _evento_desde_mensaje(msg, "00:00:05.000")
    assert ev is not None
    assert ev.tipo == "ZONA"
    assert ev.track_id == 5
    assert ev.zona == "entrada_carga -> zona_carga_norte"


def test_mensaje_desconocido_retorna_none():
    assert _evento_desde_mensaje("[INFO] nada relevante", "00:00:00.000") is None


# ─── Utilidades de imagen / opciones ────────────────────────────────────────

def test_escalar_frame_reduce_manteniendo_aspecto():
    frame = np.zeros((480, 1280, 3), dtype=np.uint8)
    reducido = escalar_frame(frame, 640)
    assert reducido.shape[0] == 240
    assert reducido.shape[1] == 640


def test_escalar_frame_no_agranda():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert escalar_frame(frame, 640).shape == (100, 200, 3)


def test_opciones_analisis_defaults():
    op = OpcionesAnalisis()
    assert op.guardar_db is True
    assert op.usar_ocr is False
    assert op.usar_clasificador is False
    assert op.track_skip == 3
    assert op.track_imgsz == 320
    assert op.exportar_video is False
    assert op.ruta_salida == ""
    assert op.max_frames == 0
