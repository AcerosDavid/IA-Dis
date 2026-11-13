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


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS NUEVOS — Plataforma de despacho (Paso 1)
# ══════════════════════════════════════════════════════════════════════════════

# ── Videos ────────────────────────────────────────────────────────────────────

class VideoOut(BaseModel):
    """Schema de respuesta para un video."""
    model_config = ConfigDict(from_attributes=True)

    id:               int
    original_filename: str
    storage_key:      str
    content_type:     str | None = None
    file_size_bytes:  int | None = None
    duration_seconds: float | None = None
    checksum:         str | None = None
    uploaded_by:      str | None = None
    uploaded_at:      str
    project_id:       str | None = None
    equipment_id:     str | None = None
    shift_id:         str | None = None
    recorded_at:      str | None = None
    retention_status: str = "active"


class VideoCreate(BaseModel):
    """Metadatos al registrar un video."""
    project_id:   str | None = None
    equipment_id: str | None = None
    shift_id:    str | None = None
    recorded_at:  str | None = None   # ISO 8601


# ── Analysis Jobs ─────────────────────────────────────────────────────────────

class AnalysisJobOut(BaseModel):
    """Schema de respuesta para un job de análisis."""
    model_config = ConfigDict(from_attributes=True)

    id:               int
    video_id:         int
    status:           str
    stage:            str | None = None
    progress_percent: float | None = None
    analysis_options: dict | None = None
    model_id:         str | None = None
    model_version:    str | None = None
    created_by:       str | None = None
    created_at:       str
    started_at:       str | None = None
    finished_at:      str | None = None
    error_message:    str | None = None
    retry_count:      int = 0


class AnalysisJobCreate(BaseModel):
    """Payload para crear un nuevo job de análisis."""
    model_id:    str | None = None
    model_version: str | None = None
    options:     dict = Field(default_factory=dict)
    metadata:    dict = Field(default_factory=dict)


# ── Detected Events ────────────────────────────────────────────────────────────

class DetectedEventOut(BaseModel):
    """Schema de respuesta para un evento detectado."""
    model_config = ConfigDict(from_attributes=True)

    id:               int
    analysis_job_id:  int
    category:         str
    description:      str | None = None
    start_time_seconds: float
    end_time_seconds:  float | None = None
    confidence:       float | None = None
    severity:         str | None = None
    evidence_reference: str | None = None
    review_status:    str = "pending"
    reviewed_by:      str | None = None
    reviewed_at:      str | None = None
    review_comment:   str | None = None
    created_at:       str


class EventReviewIn(BaseModel):
    """Payload para revisar un evento."""
    review_status: str = Field(pattern="^(confirmed|rejected|corrected|follow_up)$")
    review_comment: str | None = None


# ── Reports ──────────────────────────────────────────────────────────────────

class ReportOut(BaseModel):
    """Schema de respuesta para un reporte."""
    model_config = ConfigDict(from_attributes=True)

    id:              int
    analysis_job_id: int
    format:          str
    status:          str
    storage_key:     str | None = None
    created_by:      str | None = None
    created_at:      str
    finished_at:     str | None = None
    error_message:   str | None = None
    template_version: str | None = None


class ReportCreate(BaseModel):
    """Payload para solicitar un reporte."""
    format: str = Field(pattern="^(pdf|xlsx|csv)$", default="pdf")


# ── Paginación ────────────────────────────────────────────────────────────────

class PaginacionMeta(BaseModel):
    """Metadatos de paginación para respuestas de lista."""
    page:        int
    page_size:   int
    total:       int
    total_pages: int
