"""
Configuracion centralizada del sistema usando pydantic-settings.
Lee variables de entorno o del archivo .env en la raiz del proyecto.
"""
from __future__ import annotations

from typing import Optional
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuracion de la aplicacion.

    Variables obligatorias:
      - VIDEO_PATH:  ruta al video de prueba local
      - MODEL_PATH:  ruta al modelo YOLO26 (.pt)
      - DB_URL:      URL de conexión a la base de datos
                     SQLite:     sqlite:///data/eventos.db
                     MySQL:      mysql+pymysql://user:pass@host:3306/db
                     PostgreSQL: postgresql+psycopg2://user:pass@host:5432/db

    Variables opcionales:
      - CONFIDENCE_THRESHOLD: umbral de confianza (default: 0.5)
      - API_HOST:             host FastAPI (default: 127.0.0.1)
      - API_PORT:             puerto FastAPI (default: 8000)
      - CAMERA_SOURCE:        fuente de camara para camera.py
                              Ejemplos:
                                0                              -> webcam local
                                http://192.168.1.X:8080/video -> IP Webcam (Android)
                                rtsp://usuario:clave@ip/stream -> camara RTSP industrial
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Obligatorias
    video_path: str
    model_path: str
    db_url: str

    # Opcionales
    confidence_threshold: float = 0.5
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    camera_source: Optional[str] = None   # None = webcam local (indice 0)


def load_config() -> Settings:
    """
    Carga y valida la configuracion desde .env o variables de entorno.
    Falla rapido si alguna variable obligatoria no esta definida.

    Raises:
        SystemExit: si faltan variables obligatorias.
    """
    try:
        return Settings()
    except ValidationError as e:
        for error in e.errors():
            field = error.get("loc", ("desconocido",))[0]
            print(f"[CONFIG ERROR] Variable {str(field).upper()} no definida o invalida")
        raise SystemExit(1) from e