"""
Tests unitarios y de propiedad para la clase Detector.
El modelo YOLO se mockea siempre — no se requiere best.pt ni descarga.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.ai.detector import CLASE_MAP, Detector


# ─── Helpers de mock ─────────────────────────────────────────────────────────

def make_mock_box(clase_nombre: str, confianza: float):
    """Crea un mock de box YOLO con clase y confianza dadas."""
    box = MagicMock()
    box.cls = MagicMock()
    box.cls.__int__ = lambda self: 0
    box.conf = MagicMock()
    box.conf.__float__ = lambda self: confianza
    return box, clase_nombre


def make_mock_resultado(detecciones: list[tuple[str, float]]):
    """Crea un mock de resultado YOLO con lista de (clase, confianza)."""
    resultado = MagicMock()
    names = {i: d[0] for i, d in enumerate(detecciones)}
    resultado.names = names

    boxes = []
    for i, (clase, conf) in enumerate(detecciones):
        box = MagicMock()
        idx = i
        box.cls = MagicMock()
        box.cls.__int__ = lambda self, _i=idx: _i
        box.conf = MagicMock()
        box.conf.__float__ = lambda self, _c=conf: _c
        boxes.append(box)

    resultado.boxes = boxes
    return resultado


# ─── Smoke tests ─────────────────────────────────────────────────────────────

def test_instanciacion_defaults():
    """Detector se instancia con confidence_threshold=0.5 por defecto."""
    d = Detector("models/yolo26n.pt")
    assert d.confidence_threshold == 0.5
    assert d.model is None


def test_instanciacion_custom_threshold():
    """Detector acepta umbral personalizado."""
    d = Detector("models/yolo26n.pt", confidence_threshold=0.75)
    assert d.confidence_threshold == 0.75


def test_load_model_falla_ruta_inexistente(capsys):
    """load_model imprime [MODEL ERROR] y lanza SystemExit si la ruta no existe."""
    d = Detector("ruta/inexistente.pt")
    # Mockear ultralytics.YOLO para simular error de archivo no encontrado
    # sin necesitar el paquete instalado
    import sys
    from types import ModuleType

    # Crear mock del módulo ultralytics si no está disponible
    mock_ultralytics = ModuleType("ultralytics")

    def mock_yolo_cls(path):
        raise FileNotFoundError(f"No such file: {path}")

    mock_ultralytics.YOLO = mock_yolo_cls
    sys.modules["ultralytics"] = mock_ultralytics

    try:
        with pytest.raises(SystemExit):
            d.load_model()
        captured = capsys.readouterr()
        assert "[MODEL ERROR]" in captured.out
    finally:
        # Limpiar para no afectar otros tests
        if "ultralytics" in sys.modules and sys.modules["ultralytics"] is mock_ultralytics:
            del sys.modules["ultralytics"]


def test_detect_frame_sin_modelo_lanza_runtime_error():
    """detect_frame lanza RuntimeError si se llama antes de load_model."""
    d = Detector("models/yolo26n.pt")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        d.detect_frame(frame, 0, "00:00:00.000")


def test_clase_map_correcto():
    """CLASE_MAP mapea truck→camion, excavator→excavadora y wheel_loader→pala."""
    assert CLASE_MAP["truck"] == "camion"
    assert CLASE_MAP["excavator"] == "excavadora"
    assert CLASE_MAP["wheel_loader"] == "pala"


# ─── Property tests ──────────────────────────────────────────────────────────

clases_validas_st   = st.sampled_from(["truck", "excavator"])
clases_invalidas_st = st.text(min_size=1).filter(lambda x: x not in CLASE_MAP)
confianza_alta_st   = st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
confianza_baja_st   = st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False)
umbral_st           = st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)


@given(clases=st.lists(clases_invalidas_st, min_size=1, max_size=10))
@settings(max_examples=100, deadline=None)
def test_property_2_filtrado_clases(clases):
    """
    Property 2: detect_frame descarta clases que no están en CLASE_MAP.

    Validates: Requirements 3.2
    """
    d = Detector("models/yolo26n.pt", confidence_threshold=0.0)
    d.model = MagicMock()

    detecciones = [(c, 0.9) for c in clases]
    d.model.return_value = [make_mock_resultado(detecciones)]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = d.detect_frame(frame, 0, "00:00:00.000")

    # Ninguna clase inválida debe aparecer
    tipos = {e["tipo_maquina"] for e in result}
    assert tipos.issubset({"camion", "pala", "excavadora"})
    assert len(result) == 0  # todas las clases son inválidas


@given(confianza=confianza_baja_st, umbral=umbral_st)
@settings(max_examples=100, deadline=None)
def test_property_3_filtrado_por_umbral(confianza, umbral):
    """
    Property 3: detect_frame descarta detecciones con confianza < umbral.

    Validates: Requirements 3.3
    """
    if confianza >= umbral:
        return  # solo nos interesa el caso confianza < umbral

    d = Detector("models/yolo26n.pt", confidence_threshold=umbral)
    d.model = MagicMock()

    detecciones = [("truck", confianza)]
    d.model.return_value = [make_mock_resultado(detecciones)]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = d.detect_frame(frame, 0, "00:00:00.000")
    assert len(result) == 0


@given(
    clase=clases_validas_st,
    confianza=confianza_alta_st,
    frame_number=st.integers(min_value=0, max_value=10000),
    timestamp=st.text(min_size=1, max_size=15),
)
@settings(max_examples=100, deadline=None)
def test_property_4_estructura_evento(clase, confianza, frame_number, timestamp):
    """
    Property 4: eventos retornados tienen exactamente los campos requeridos
    con tipos y valores correctos.

    Validates: Requirements 3.4
    """
    d = Detector("models/yolo26n.pt", confidence_threshold=0.0)
    d.model = MagicMock()

    detecciones = [(clase, confianza)]
    d.model.return_value = [make_mock_resultado(detecciones)]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = d.detect_frame(frame, frame_number, timestamp)

    assert len(result) == 1
    e = result[0]
    assert set(e.keys()) == {"tipo_maquina", "clase_id", "confianza", "numero_fotograma", "timestamp"}
    assert e["tipo_maquina"] in {"camion", "pala", "excavadora"}
    assert e["clase_id"] == clase
    assert e["confianza"] >= 0.0
    assert e["numero_fotograma"] == frame_number
    assert e["timestamp"] == timestamp
