"""
Modelos ORM de la base de datos local (SQLite por defecto).

Tablas del nucleo operativo:
  - eventos    : detecciones de maquinaria (aparicion/desaparicion por frame)
  - zonas      : zonas operativas para geofencing
  - ciclos     : ciclos de carga/descarga construidos a partir de las zonas
  - turnos     : turnos de trabajo
  - video_jobs : trabajos de analisis batch (nuevo)
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base declarativa para todos los modelos ORM."""


class Evento(Base):
    """Deteccion de maquinaria registrada durante el analisis."""

    __tablename__ = "eventos"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    timestamp         = Column(String,  nullable=False)   # "HH:MM:SS.mmm"
    tipo_maquina      = Column(String,  nullable=False)
    clase_id          = Column(String,  nullable=False)
    confianza         = Column(Float,   nullable=False)
    numero_fotograma  = Column(Integer, nullable=False)
    zona_nombre       = Column(String,  nullable=True)
    tipo_evento_ciclo = Column(String,  nullable=True)    # aparicion | desaparicion

    def __repr__(self) -> str:
        return (
            f"Evento(id={self.id!r}, ts={self.timestamp!r}, "
            f"tipo={self.tipo_maquina!r}, conf={self.confianza!r})"
        )


class ZonaDB(Base):
    """Zona operativa persistida (geofencing)."""

    __tablename__ = "zonas"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    nombre        = Column(String,  nullable=False, unique=True)
    tipo          = Column(String,  nullable=False)
    vertices_json = Column(String,  nullable=False)
    color_hex     = Column(String,  nullable=False, default="#00FFFF")
    activa        = Column(Boolean, nullable=False, default=True)
    descripcion   = Column(String,  nullable=True)

    def __repr__(self) -> str:
        return f"ZonaDB(nombre={self.nombre!r}, tipo={self.tipo!r}, activa={self.activa!r})"


class Ciclo(Base):
    """Ciclo operativo de carga/descarga."""

    __tablename__ = "ciclos"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    maquina_id          = Column(String,  nullable=False)
    tipo_maquina        = Column(String,  nullable=False)
    ts_entrada_carga    = Column(String,  nullable=True)
    ts_salida_carga     = Column(String,  nullable=True)
    ts_entrada_descarga = Column(String,  nullable=True)
    ts_salida_descarga  = Column(String,  nullable=True)
    zona_carga          = Column(String,  nullable=True)
    zona_descarga       = Column(String,  nullable=True)
    duracion_segundos   = Column(Float,   nullable=True)
    completado          = Column(Boolean, nullable=False, default=False)
    turno_id            = Column(Integer, ForeignKey("turnos.id"), nullable=True)

    turno = relationship("Turno", back_populates="ciclos")

    def __repr__(self) -> str:
        return (
            f"Ciclo(id={self.id!r}, maquina={self.maquina_id!r}, "
            f"completado={self.completado!r}, dur={self.duracion_segundos!r}s)"
        )


class Turno(Base):
    """Turno de trabajo."""

    __tablename__ = "turnos"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    tipo         = Column(String,  nullable=False)
    fecha        = Column(String,  nullable=False)
    ts_inicio    = Column(String,  nullable=False)
    ts_fin       = Column(String,  nullable=True)
    activo       = Column(Boolean, nullable=False, default=True)
    total_ciclos = Column(Integer, nullable=False, default=0)

    ciclos = relationship("Ciclo", back_populates="turno")

    def __repr__(self) -> str:
        return f"Turno(id={self.id!r}, tipo={self.tipo!r}, fecha={self.fecha!r}, activo={self.activo!r})"


class VideoJob(Base):
    """Trabajo de analisis batch de un video.

    Estado: pendiente -> procesando -> generando_video -> completado | error
    """

    __tablename__ = "video_jobs"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    nombre_original    = Column(String,  nullable=False)
    ruta_original      = Column(String,  nullable=False)   # copia local en data/videos/
    ruta_salida        = Column(String,  nullable=True)    # video anotado en data/output/
    ruta_reporte_pdf   = Column(String,  nullable=True)    # data/reports/*.pdf
    ruta_reporte_xlsx   = Column(String,  nullable=True)    # data/reports/*.xlsx

    # Estado del trabajo
    estado             = Column(String,  nullable=False, default="pendiente")
    # pendiente | procesando | generando_video | completado | error | cancelado

    # Progreso (0-100)
    progreso           = Column(Float,   nullable=False, default=0.0)
    frame_actual        = Column(Integer, nullable=False, default=0)
    total_frames       = Column(Integer, nullable=False, default=0)
    fps_video          = Column(Float,   nullable=False, default=0.0)

    # Metadatos del video original
    duracion_seg       = Column(Float,   nullable=True)
    resolucion         = Column(String,  nullable=True)   # "WxH"
    fps_originales     = Column(Float,   nullable=True)

    # Configuracion usada
    modelo             = Column(String,  nullable=True)
    confianza_umbral   = Column(Float,   nullable=True)
    track_skip         = Column(Integer, nullable=True)
    track_imgsz        = Column(Integer, nullable=True)
    dispositivo        = Column(String,  nullable=True)   # cpu | cuda
    guardar_db         = Column(Boolean, nullable=False, default=True)
    usar_ocr           = Column(Boolean, nullable=False, default=False)
    usar_clasificador  = Column(Boolean, nullable=False, default=False)

    # Resultados
    total_detecciones   = Column(Integer, nullable=True)
    total_eventos       = Column(Integer, nullable=True)
    total_unicos        = Column(Integer, nullable=True)
    confianza_promedio  = Column(Float,   nullable=True)
    tiempo_procesamiento = Column(Float,   nullable=True)  # segundos

    # Errores
    error_msg          = Column(Text,    nullable=True)

    # Timestamps
    creado_en          = Column(String,  nullable=False)   # ISO datetime
    iniciado_en        = Column(String,  nullable=True)
    finalizado_en      = Column(String,  nullable=True)

    def __repr__(self) -> str:
        return (
            f"VideoJob(id={self.id!r}, nombre={self.nombre_original!r}, "
            f"estado={self.estado!r}, progreso={self.progreso!r})"
        )
