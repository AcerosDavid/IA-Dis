"""
Smoke tests para src/config.py.
Verifica carga correcta, defaults y fallo en variables faltantes.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch
from pydantic_settings import SettingsConfigDict


def _settings_sin_env_file(**kwargs):
    """
    Crea una subclase de Settings que ignora el .env del disco.
    Necesario para que los tests no lean el .env real del proyecto.
    """
    from src.config import Settings

    class SettingsTest(Settings):
        model_config = SettingsConfigDict(env_file=None)

    return SettingsTest(**kwargs)


def test_settings_carga_con_todas_las_variables():
    """Settings carga correctamente cuando todas las variables estan definidas."""
    env = {
        "VIDEO_PATH": "data/raw/test.mp4",
        "MODEL_PATH": "models/yolo26n.pt",
        "DB_PATH": "data/test.db",
        "CONFIDENCE_THRESHOLD": "0.6",
        "API_HOST": "0.0.0.0",
        "API_PORT": "9000",
    }
    with patch.dict(os.environ, env, clear=True):
        s = _settings_sin_env_file()
        assert s.video_path == "data/raw/test.mp4"
        assert s.model_path == "models/yolo26n.pt"
        assert s.db_path == "data/test.db"
        assert s.confidence_threshold == 0.6
        assert s.api_host == "0.0.0.0"
        assert s.api_port == 9000


def test_confidence_threshold_default():
    """CONFIDENCE_THRESHOLD usa 0.5 cuando no esta definido."""
    env = {
        "VIDEO_PATH": "data/raw/test.mp4",
        "MODEL_PATH": "models/yolo26n.pt",
        "DB_PATH": "data/test.db",
    }
    with patch.dict(os.environ, env, clear=True):
        s = _settings_sin_env_file()
        assert s.confidence_threshold == 0.5


def test_api_host_port_defaults():
    """API_HOST y API_PORT usan sus defaults cuando no estan definidos."""
    env = {
        "VIDEO_PATH": "data/raw/test.mp4",
        "MODEL_PATH": "models/yolo26n.pt",
        "DB_PATH": "data/test.db",
    }
    with patch.dict(os.environ, env, clear=True):
        s = _settings_sin_env_file()
        assert s.api_host == "127.0.0.1"
        assert s.api_port == 8000


@pytest.mark.parametrize("missing_var", ["VIDEO_PATH", "MODEL_PATH", "DB_PATH"])
def test_load_config_falla_si_falta_variable_obligatoria(missing_var, tmp_path):
    """load_config() lanza SystemExit cuando falta una variable obligatoria."""
    env = {
        "VIDEO_PATH": "v.mp4",
        "MODEL_PATH": "m.pt",
        "DB_PATH": "d.db",
    }
    del env[missing_var]

    # Usar un .env vacio en tmp_path para que pydantic-settings no lea el .env real
    env_vacio = tmp_path / ".env"
    env_vacio.write_text("")

    with patch.dict(os.environ, env, clear=True):
        from pydantic import ValidationError
        from src.config import Settings

        class SettingsTest(Settings):
            model_config = SettingsConfigDict(env_file=str(env_vacio))

        from pydantic import ValidationError as VE

        def load_config_test():
            try:
                return SettingsTest()
            except VE as e:
                for error in e.errors():
                    field = error.get("loc", ("desconocido",))[0]
                    print(f"[CONFIG ERROR] Variable {str(field).upper()} no definida")
                raise SystemExit(1) from e

        with pytest.raises(SystemExit):
            load_config_test()