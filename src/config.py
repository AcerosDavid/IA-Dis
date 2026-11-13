"""
Configuracion centralizada de la aplicacion de escritorio.

Lee variables de entorno o del archivo .env en la raiz del proyecto.
Todas las variables tienen valores por defecto: la aplicacion puede abrirse
sin .env y el usuario selecciona video/modelo desde la interfaz.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion de la aplicacion de escritorio."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Rutas ─────────────────────────────────────────────────────────────────
    model_path: str = "models/best.pt"
    db_url: str = "sqlite:///data/eventos.db"
    # Ruta inicial sugerida para el selector de video (puede quedar vacia).
    video_path: str = ""

    # ── Deteccion ─────────────────────────────────────────────────────────────
    confidence_threshold: float = 0.5
    # Fuente de camara: indice ("0") o URL RTSP/HTTP. None = webcam local.
    camera_source: Optional[str] = None

    # ── Rendimiento del pipeline (CPU) ────────────────────────────────────────
    # Procesar 1 de cada N frames. 1 = todos, 3 = recomendado en CPU.
    track_skip: int = 3
    # Resolucion de inferencia; 320 es el optimo para CPU.
    track_imgsz: int = 320
    # Frames sin ver un track antes de emitir "desaparicion".
    frames_ausencia: int = 90
    # Cada cuantos frames procesados se hace COMMIT de la sesion de DB.
    flush_every_frames: int = 60

    # ── Modulos opcionales ────────────────────────────────────────────────────
    ocr_habilitado: bool = False
    clasificador_habilitado: bool = False

    # ── Interfaz ──────────────────────────────────────────────────────────────
    # Ancho maximo del frame mostrado en la ventana (0 = sin escalar).
    display_width: int = 960
    window_geometry: str = "1180x760"


@lru_cache(maxsize=1)
def load_config() -> Settings:
    """Carga y valida la configuracion. Falla rapido si hay valores invalidos."""
    try:
        return Settings()
    except ValidationError as e:
        for error in e.errors():
            field = error.get("loc", ("desconocido",))[0]
            print(f"[CONFIG ERROR] Variable {str(field).upper()} invalida: {error.get('msg')}")
        raise SystemExit(1) from e
