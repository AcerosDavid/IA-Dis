"""
Schemas Pydantic para validación de datos de entrada/salida de la API.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EventoSchema(BaseModel):
    """Schema de respuesta para un evento de detección."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: str
    tipo_maquina: str
    clase_id: str
    confianza: float
    numero_fotograma: int
