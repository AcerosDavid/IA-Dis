"""Tests de la configuracion de la aplicacion de escritorio."""
from __future__ import annotations

import os
from unittest.mock import patch

from pydantic_settings import SettingsConfigDict


def _settings_sin_env_file(**kwargs):
    """Crea una subclase de Settings que ignora el .env del disco."""
    from src.config import Settings

    class SettingsTest(Settings):
        model_config = SettingsConfigDict(env_file=None)

    return SettingsTest(**kwargs)


def test_settings_defaults():
    """Settings funciona sin variables de entorno definidas."""
    with patch.dict(os.environ, {}, clear=True):
        s = _settings_sin_env_file()
        assert s.model_path == "models/best.pt"
        assert s.db_url == "sqlite:///data/eventos.db"
        assert s.video_path == ""
        assert s.confidence_threshold == 0.5
        assert s.track_skip == 3
        assert s.track_imgsz == 320


def test_settings_desde_entorno():
    """Las variables de entorno sobreescriben los defaults."""
    env = {
        "MODEL_PATH": "models/custom.pt",
        "DB_URL": "sqlite:///data/custom.db",
        "CONFIDENCE_THRESHOLD": "0.6",
        "TRACK_SKIP": "5",
        "TRACK_IMGSZ": "480",
        "OCR_HABILITADO": "true",
        "WINDOW_GEOMETRY": "1024x600",
    }
    with patch.dict(os.environ, env, clear=True):
        s = _settings_sin_env_file()
        assert s.model_path == "models/custom.pt"
        assert s.db_url == "sqlite:///data/custom.db"
        assert s.confidence_threshold == 0.6
        assert s.track_skip == 5
        assert s.track_imgsz == 480
        assert s.ocr_habilitado is True
        assert s.window_geometry == "1024x600"


def test_settings_campos_obsoletos_ignorados():
    """Variables de la antigua API no rompen la configuracion."""
    with patch.dict(os.environ, {"API_HOST": "0.0.0.0", "CORS_ORIGINS": "*"}, clear=True):
        s = _settings_sin_env_file()
        assert not hasattr(s, "api_host")
        assert not hasattr(s, "cors_origins")
