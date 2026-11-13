"""
Modelos ORM de la base de datos local (SQLite por defecto).

Tablas del nucleo operativo:
  - eventos : detecciones de maquinaria (aparicion/desaparicion por frame)
  - zonas   : zonas operativas para geofencing
  - ciclos  : ciclos de carga/descarga construidos a partir de las zonas
  - turnos  : turnos de trabajo
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
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
