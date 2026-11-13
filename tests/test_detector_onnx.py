"""
Tests del post-proceso ONNX del Detector (sin cargar el modelo real).

Verifican los dos layouts de salida que produce el export de Ultralytics y
que el tracker mantiene el mismo ID entre frames.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.ai.detector import Detector


class _FakeInput:
    name = "images"


class _FakeSession:
    """Sesión ONNX mínima que siempre retorna la misma salida."""

    def __init__(self, salida: np.ndarray) -> None:
        self.salida = salida

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, *_args, **_kwargs):
        return [self.salida]


def _detector_con_salida(salida: np.ndarray) -> Detector:
    d = Detector("models/best.pt", confidence_threshold=0.5)
    d._use_onnx = True
    d._class_names = {0: "dump_truck", 1: "excavator"}
    d._ort_session = _FakeSession(salida)
    return d


def test_onnx_layout_clasico_detecta_y_escala():
    """Salida (1, 4+nc, N) → una detección mapeada al frame original."""
    pred = np.zeros((1, 6, 100), dtype=np.float32)
    pred[0, :, 0] = [160, 120, 50, 40, 0.9, 0.0]

    d = _detector_con_salida(pred)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    dets = d.track_frame(frame, 0, "00:00:00.000", 320)
    assert len(dets) == 1
    det = dets[0]
    assert det["tipo_maquina"] == "camion"
    assert det["confianza"] == pytest.approx(0.9, abs=0.01)
    assert det["bbox"] == pytest.approx([270.0, 120.0, 370.0, 200.0], abs=1.0)


def test_onnx_layout_end2end_detecta():
    """Salida (1, N, 6) de YOLO26 end-to-end → una detección xyxy directa."""
    pred = np.zeros((1, 300, 6), dtype=np.float32)
    pred[0, 0] = [300, 200, 400, 300, 0.88, 1]

    d = _detector_con_salida(pred)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    dets = d.track_frame(frame, 0, "00:00:00.000", 320)
    assert len(dets) == 1
    assert dets[0]["tipo_maquina"] == "excavadora"
    assert dets[0]["bbox"] == pytest.approx([600.0, 320.0, 640.0, 480.0], abs=1.0)


def test_onnx_tracker_mantiene_id_entre_frames():
    """El mismo ID debe sobrevivir frames consecutivos (no crear uno nuevo)."""
    pred = np.zeros((1, 6, 100), dtype=np.float32)
    pred[0, :, 0] = [160, 120, 50, 40, 0.9, 0.0]

    d = _detector_con_salida(pred)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    ids = []
    for i in range(3):
        dets = d.track_frame(frame, i, f"00:00:0{i}.000", 320)
        assert len(dets) == 1
        ids.append(dets[0]["track_id"])

    assert ids == [ids[0]] * 3
