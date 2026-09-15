"""
Modelos ORM de la base de datos usando SQLAlchemy.
Tablas del sistema de despacho minero.
"""
from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base declarativa para todos los modelos ORM."""
    pass


# ── Tabla: eventos ────────────────────────────────────────────────────────────

class Evento(Base):
    """
    Deteccion de maquinaria en un fotograma de video.
    Registrada por el Detector YOLO26 en tiempo real.

    Invariantes:
      - tipo_maquina: 'camion' | 'pala'
      - confianza:    [confidence_threshold, 1.0]
      - numero_fotograma >= 0
    """
    __tablename__ = "eventos"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    timestamp        = Column(String,  nullable=False)  # "HH:MM:SS.mmm"
    tipo_maquina     = Column(String,  nullable=False)  # "camion" | "pala"
    clase_id         = Column(String,  nullable=False)  # clase YOLO: "truck", "excavator"
    confianza        = Column(Float,   nullable=False)  # [0.5, 1.0]
    numero_fotograma = Column(Integer, nullable=False)  # >= 0
    # Geofencing (nullable: se rellena cuando hay zonas configuradas)
    zona_nombre      = Column(String,  nullable=True)   # zona donde ocurrio la deteccion
    tipo_evento_ciclo = Column(String, nullable=True)   # "entrada_carga", "salida_descarga", etc.

    def __repr__(self) -> str:
        return (
            f"Evento(id={self.id!r}, ts={self.timestamp!r}, "
            f"tipo={self.tipo_maquina!r}, conf={self.confianza!r})"
        )


# ── Tabla: zonas ──────────────────────────────────────────────────────────────

class ZonaDB(Base):
    """
    Zona operativa persistida en base de datos.
    Permite configurar zonas sin recompilar el codigo.

    tipo: "carga" | "descarga" | "parqueo" | "transito"
    vertices_json: lista de puntos serializada como JSON string
                   ej: "[[100,100],[400,100],[400,300],[100,300]]"
    color_hex: color para visualizacion ej: "#00FFFF"
    activa: si la zona esta activa para deteccion
    """
    __tablename__ = "zonas"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    nombre        = Column(String,  nullable=False, unique=True)
    tipo          = Column(String,  nullable=False)   # "carga"|"descarga"|"parqueo"|"transito"
    vertices_json = Column(String,  nullable=False)   # JSON: [[x1,y1],[x2,y2],...]
    color_hex     = Column(String,  nullable=False, default="#00FFFF")
    activa        = Column(Boolean, nullable=False, default=True)
    descripcion   = Column(String,  nullable=True)    # Descripcion opcional

    def __repr__(self) -> str:
        return f"ZonaDB(nombre={self.nombre!r}, tipo={self.tipo!r}, activa={self.activa!r})"


# ── Tabla: ciclos ─────────────────────────────────────────────────────────────

class Ciclo(Base):
    """
    Ciclo operativo completo de un camion:
    entrada_carga → salida_carga → entrada_descarga → salida_descarga

    Se construye a partir de los EventoCiclo del GestorGeofencing.
    Permite calcular tiempos de ciclo para reportes de productividad.

    duracion_segundos: tiempo total del ciclo (calculado al cierre)
    completado: True cuando el ciclo tiene entrada Y salida de descarga
    """
    __tablename__ = "ciclos"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    maquina_id           = Column(String,  nullable=False)   # "camion_0", "camion_1"
    tipo_maquina         = Column(String,  nullable=False)   # "camion"
    # Tiempos de cada fase
    ts_entrada_carga     = Column(String,  nullable=True)
    ts_salida_carga      = Column(String,  nullable=True)
    ts_entrada_descarga  = Column(String,  nullable=True)
    ts_salida_descarga   = Column(String,  nullable=True)
    # Resumen
    zona_carga           = Column(String,  nullable=True)
    zona_descarga        = Column(String,  nullable=True)
    duracion_segundos    = Column(Float,   nullable=True)    # calculado al cierre
    completado           = Column(Boolean, nullable=False, default=False)
    turno_id             = Column(Integer, ForeignKey("turnos.id"), nullable=True)

    turno = relationship("Turno", back_populates="ciclos")

    def __repr__(self) -> str:
        return (
            f"Ciclo(id={self.id!r}, maquina={self.maquina_id!r}, "
            f"completado={self.completado!r}, dur={self.duracion_segundos!r}s)"
        )


# ── Tabla: turnos ─────────────────────────────────────────────────────────────

class Turno(Base):
    """
    Turno de trabajo en la operacion minera.
    Agrupa los ciclos realizados en un periodo (manana, tarde, noche).

    tipo: "manana" | "tarde" | "noche"
    activo: solo un turno puede estar activo a la vez
    """
    __tablename__ = "turnos"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    tipo            = Column(String,  nullable=False)   # "manana"|"tarde"|"noche"
    fecha           = Column(String,  nullable=False)   # "YYYY-MM-DD"
    ts_inicio       = Column(String,  nullable=False)   # "HH:MM:SS"
    ts_fin          = Column(String,  nullable=True)    # null si esta activo
    activo          = Column(Boolean, nullable=False, default=True)
    total_ciclos    = Column(Integer, nullable=False, default=0)

    ciclos = relationship("Ciclo", back_populates="turno")

    def __repr__(self) -> str:
        return f"Turno(id={self.id!r}, tipo={self.tipo!r}, fecha={self.fecha!r}, activo={self.activo!r})"
