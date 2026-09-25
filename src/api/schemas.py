"""
Schemas Pydantic para validación de datos de entrada/salida de la API.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ── Eventos ───────────────────────────────────────────────────────────────────

class EventoSchema(BaseModel):
    """Schema de respuesta para un evento de detección."""

    model_config = ConfigDict(from_attributes=True)

    id:                int
    timestamp:         str
    tipo_maquina:      str
    clase_id:          str
    confianza:         float = Field(ge=0.0, le=1.0)
    numero_fotograma:  int
    zona_nombre:       str | None = None
    tipo_evento_ciclo: str | None = None


# ── Zonas ─────────────────────────────────────────────────────────────────────

class ZonaSchema(BaseModel):
    """Schema de respuesta para una zona operativa."""

    model_config = ConfigDict(from_attributes=True)

    id:            int
    nombre:        str
    tipo:          str
    vertices_json: str
    color_hex:     str
    activa:        bool
    descripcion:   str | None = None


class ZonaCreateSchema(BaseModel):
    """Schema para crear o actualizar una zona via API."""

    nombre:      str
    tipo:        str = Field(pattern="^(carga|descarga|parqueo|transito)$")
    vertices_json: str          # JSON string: "[[x1,y1],[x2,y2],...]"
    color_hex:   str = "#00FFFF"
    activa:      bool = True
    descripcion: str | None = None


# ── Ciclos ────────────────────────────────────────────────────────────────────

class CicloSchema(BaseModel):
    """Schema de respuesta para un ciclo operativo completo."""

    model_config = ConfigDict(from_attributes=True)

    id:                  int
    maquina_id:          str
    tipo_maquina:        str
    ts_entrada_carga:    str | None = None
    ts_salida_carga:     str | None = None
    ts_entrada_descarga: str | None = None
    ts_salida_descarga:  str | None = None
    zona_carga:          str | None = None
    zona_descarga:       str | None = None
    duracion_segundos:   float | None = None
    completado:          bool


# ── Estadísticas ──────────────────────────────────────────────────────────────

class EstadisticasTipoSchema(BaseModel):
    """Conteo de eventos por tipo de máquina."""
    tipo_maquina: str
    total:        int


class EstadisticasZonaSchema(BaseModel):
    """Conteo de eventos por zona."""
    zona_nombre: str
    total:       int


class EstadisticasCiclosSchema(BaseModel):
    """Estadísticas de ciclos completados por tipo de máquina."""
    tipo_maquina:      str
    total:             int
    duracion_promedio: float   # segundos
    duracion_min:      float
    duracion_max:      float


class ResumenSchema(BaseModel):
    """Resumen general del sistema."""
    total_eventos:      int
    total_ciclos:       int
    ciclos_completados: int
    por_tipo:           list[EstadisticasTipoSchema]
    por_zona:           list[EstadisticasZonaSchema]
