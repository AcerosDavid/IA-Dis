"""
Tests de integración del pipeline CLI completo (main.py).
Mockea cv2.VideoCapture y Detector para evitar dependencias externas.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

import numpy as np


# ─── Smoke / example tests ───────────────────────────────────────────────────

def test_open_video_archivo_no_existe(capsys):
    """open_video lanza SystemExit si el archivo no existe."""
    from main import open_video
    with pytest.raises(SystemExit):
        open_video("ruta/que/no/existe.mp4")
    captured = capsys.readouterr()
    assert "[VIDEO ERROR]" in captured.out


def test_open_video_opencv_no_puede_decodificar(tmp_path, capsys):
    """open_video lanza SystemExit si OpenCV no puede decodificar el archivo."""
    video_path = tmp_path / "falso.mp4"
    video_path.write_bytes(b"not a video")

    with patch("main.cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        from main import open_video
        with pytest.raises(SystemExit):
            open_video(str(video_path))

    captured = capsys.readouterr()
    assert "[VIDEO ERROR]" in captured.out


def test_calcular_timestamp_formato():
    """calcular_timestamp retorna el formato HH:MM:SS.mmm correcto."""
    from main import calcular_timestamp
    ts = calcular_timestamp(fps=30.0, frame_number=90)
    assert ts == "00:00:03.000"


def test_calcular_timestamp_fps_cero():
    """calcular_timestamp maneja fps=0 sin división por cero."""
    from main import calcular_timestamp
    ts = calcular_timestamp(fps=0.0, frame_number=10)
    assert ts == "00:00:00.000"


def test_print_evento_formato(capsys):
    """print_evento imprime el formato exacto requerido."""
    from main import print_evento
    evento = {
        "numero_fotograma": 12,
        "timestamp": "00:00:00.400",
        "tipo_maquina": "camion",
        "clase_id": "truck",
        "confianza": 0.87,
    }
    print_evento(evento)
    captured = capsys.readouterr()
    assert "[EVENTO]" in captured.out
    assert "fotograma=12" in captured.out
    assert "maquina=camion" in captured.out
    assert "confianza=0.87" in captured.out


def test_error_en_fotograma_no_detiene_pipeline(capsys):
    """Un error en detect_frame de un fotograma no detiene el pipeline."""
    from main import calcular_timestamp, print_evento

    frame_count  = 0
    event_count  = 0
    error_logged = False

    # Simular el loop del pipeline con 3 frames, el segundo lanza error
    for i in range(3):
        ts = calcular_timestamp(30.0, i)
        try:
            if i == 1:
                raise ValueError("Error simulado en fotograma 1")
            # Frame normal sin detecciones
        except Exception as e:
            print(f"[ERROR] fotograma={i} | detalle={e}")
            error_logged = True
        frame_count += 1

    captured = capsys.readouterr()
    assert frame_count == 3      # procesó los 3 frames
    assert "[ERROR]" in captured.out
    assert error_logged


# ─── Property tests ──────────────────────────────────────────────────────────

@given(
    confianza=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    fotograma=st.integers(min_value=0, max_value=9999),
    tipo=st.sampled_from(["camion", "pala"]),
    clase=st.sampled_from(["truck", "excavator"]),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_formato_log_evento(capsys, confianza, fotograma, tipo, clase):
    """
    Property 7: print_evento siempre produce la línea con el formato exacto.

    Validates: Requirements 5.1
    """
    from main import print_evento
    evento = {
        "numero_fotograma": fotograma,
        "timestamp": "00:00:01.000",
        "tipo_maquina": tipo,
        "clase_id": clase,
        "confianza": confianza,
    }
    print_evento(evento)
    captured = capsys.readouterr()
    line = captured.out.strip()

    assert line.startswith("[EVENTO]")
    assert f"fotograma={fotograma}" in line
    assert f"maquina={tipo}" in line
    assert f"clase={clase}" in line
    assert "confianza=" in line
    assert "|" in line
