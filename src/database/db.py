"""
Capa de base de datos local.

Uso:
    from src.database.db import get_session, RepositorioEventos

    with get_session() as db:
        repo = RepositorioEventos(db)
        repo.insertar({...})

Configuracion en .env:
    DB_URL=sqlite:///data/eventos.db
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, Ciclo, Evento, VideoJob, ZonaDB


def _ensure_sqlite_dir(db_url: str) -> None:
    """Crea la carpeta del archivo SQLite si no existe."""
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        return
    path = db_url[len(prefix):]
    if not path or path.startswith(":memory:"):
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _build_engine(db_url: str):
    """Crea el engine SQLAlchemy a partir de la URL de conexion."""
    kwargs: dict = {}

    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        _ensure_sqlite_dir(db_url)

    engine = create_engine(db_url, **kwargs)

    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _record):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

    return engine


def _get_db_url() -> str:
    from src.config import load_config
    return load_config().db_url


# Engine y SessionFactory — lazy, se crean al primer uso.
_engine = None
_SessionFactory = None


def _init() -> None:
    global _engine, _SessionFactory
    if _engine is None:
        _engine = _build_engine(_get_db_url())
        _SessionFactory = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def get_engine():
    """Retorna el engine SQLAlchemy (lo inicializa si es la primera llamada)."""
    _init()
    return _engine


def migrate() -> None:
    """Crea todas las tablas que no existan."""
    _init()
    Base.metadata.create_all(_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Sesion con commit/rollback automatico."""
    _init()
    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Repositorio de Eventos ────────────────────────────────────────────────────

class RepositorioEventos:
    """CRUD sobre la tabla 'eventos'."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def insertar(self, evento: dict) -> None:
        """Inserta un evento con savepoint: un fallo no invalida la sesion."""
        try:
            with self.session.begin_nested():
                self.session.add(Evento(**evento))
        except Exception as e:
            print(f"[DB ERROR] No se pudo insertar evento: {evento} | {e}")

    def obtener_todos(self) -> list[Evento]:
        """Retorna todos los eventos ordenados por numero de fotograma ASC."""
        return (
            self.session.query(Evento)
            .order_by(Evento.numero_fotograma.asc())
            .all()
        )

    def obtener_recientes(self, limite: int = 200) -> list[Evento]:
        """Retorna los ultimos eventos registrados (mas recientes primero)."""
        return (
            self.session.query(Evento)
            .order_by(Evento.id.desc())
            .limit(limite)
            .all()
        )

    def obtener_por_id(self, evento_id: int) -> Evento | None:
        """Retorna el evento con el id dado, o None si no existe."""
        return self.session.get(Evento, evento_id)

    def contar(self) -> int:
        """Total de eventos almacenados."""
        from sqlalchemy import func
        return self.session.query(func.count(Evento.id)).scalar() or 0


# ── Repositorio de Zonas ──────────────────────────────────────────────────────

class RepositorioZonas:
    """CRUD sobre la tabla 'zonas'. Convierte ZonaDB <-> Zona (geometry)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def cargar_activas(self):
        """Retorna las zonas activas como instancias de src.core.geometry.Zona."""
        from src.core.geometry import Zona

        rows = self.session.query(ZonaDB).filter(ZonaDB.activa == True).all()  # noqa: E712
        zonas = []
        for row in rows:
            vertices = json.loads(row.vertices_json)
            h = row.color_hex.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            zonas.append(Zona(
                nombre=row.nombre,
                tipo=row.tipo,
                vertices=vertices,
                color_bgr=(b, g, r),
                activa=row.activa,
            ))
        return zonas

    def guardar(self, zona) -> None:
        """Persiste una Zona (crea o actualiza por nombre)."""
        b, g, r = zona.color_bgr
        color_hex = f"#{r:02X}{g:02X}{b:02X}"

        existente = (
            self.session.query(ZonaDB)
            .filter(ZonaDB.nombre == zona.nombre)
            .first()
        )
        if existente:
            existente.tipo          = zona.tipo
            existente.vertices_json = json.dumps(zona.vertices)
            existente.color_hex     = color_hex
            existente.activa        = zona.activa
        else:
            self.session.add(ZonaDB(
                nombre=zona.nombre,
                tipo=zona.tipo,
                vertices_json=json.dumps(zona.vertices),
                color_hex=color_hex,
                activa=zona.activa,
            ))
        self.session.flush()

    def obtener_todas(self):
        """Retorna todos los registros ZonaDB."""
        return self.session.query(ZonaDB).all()


# ── Repositorio de Ciclos ─────────────────────────────────────────────────────

class RepositorioCiclos:
    """Construye y consulta ciclos operativos a partir de eventos de zona."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def registrar_evento(self, evento_ciclo) -> None:
        """Procesa un EventoCiclo y actualiza el ciclo activo de la maquina."""
        maquina_id  = evento_ciclo.maquina_id
        tipo_evento = evento_ciclo.tipo_evento
        timestamp   = evento_ciclo.timestamp

        ciclo_activo = (
            self.session.query(Ciclo)
            .filter(Ciclo.maquina_id == maquina_id, Ciclo.completado == False)  # noqa: E712
            .first()
        )

        if tipo_evento == "entrada_carga":
            if ciclo_activo is None:
                self.session.add(Ciclo(
                    maquina_id       = maquina_id,
                    tipo_maquina     = evento_ciclo.tipo_maquina,
                    ts_entrada_carga = timestamp,
                    zona_carga       = evento_ciclo.zona_nombre,
                ))
                self.session.flush()

        elif tipo_evento == "salida_carga":
            if ciclo_activo and ciclo_activo.ts_salida_carga is None:
                ciclo_activo.ts_salida_carga = timestamp

        elif tipo_evento == "entrada_descarga":
            if ciclo_activo and ciclo_activo.ts_entrada_descarga is None:
                ciclo_activo.ts_entrada_descarga = timestamp
                ciclo_activo.zona_descarga       = evento_ciclo.zona_nombre

        elif tipo_evento == "salida_descarga":
            if ciclo_activo and ciclo_activo.ts_salida_descarga is None:
                ciclo_activo.ts_salida_descarga = timestamp
                if ciclo_activo.ts_entrada_carga:
                    ciclo_activo.duracion_segundos = _segundos_entre(
                        ciclo_activo.ts_entrada_carga, timestamp
                    )
                ciclo_activo.completado = True
                self.session.flush()

    def obtener_todos(self, completado: bool | None = None) -> list:
        q = self.session.query(Ciclo)
        if completado is not None:
            q = q.filter(Ciclo.completado == completado)
        return q.order_by(Ciclo.id.desc()).all()

    def obtener_por_maquina(self, maquina_id: str) -> list:
        return (
            self.session.query(Ciclo)
            .filter(Ciclo.maquina_id == maquina_id)
            .order_by(Ciclo.id.desc())
            .all()
        )

    def estadisticas(self) -> list[dict]:
        """Estadisticas agregadas de ciclos completados por tipo de maquina."""
        from sqlalchemy import func

        rows = (
            self.session.query(
                Ciclo.tipo_maquina,
                func.count(Ciclo.id).label("total"),
                func.avg(Ciclo.duracion_segundos).label("duracion_promedio"),
                func.min(Ciclo.duracion_segundos).label("duracion_min"),
                func.max(Ciclo.duracion_segundos).label("duracion_max"),
            )
            .filter(Ciclo.completado == True)  # noqa: E712
            .group_by(Ciclo.tipo_maquina)
            .all()
        )
        return [
            {
                "tipo_maquina":      r.tipo_maquina,
                "total":             r.total,
                "duracion_promedio": round(r.duracion_promedio or 0, 1),
                "duracion_min":      round(r.duracion_min or 0, 1),
                "duracion_max":      round(r.duracion_max or 0, 1),
            }
            for r in rows
        ]


def _segundos_entre(ts_inicio: str, ts_fin: str) -> float:
    """Calcula segundos entre dos timestamps 'HH:MM:SS.mmm'. 0.0 si falla."""
    def _parse(ts: str) -> float:
        partes = ts.replace(",", ".").split(":")
        return int(partes[0]) * 3600 + int(partes[1]) * 60 + float(partes[2])

    try:
        return max(0.0, _parse(ts_fin) - _parse(ts_inicio))
    except Exception:
        return 0.0


# ── Repositorio de Jobs de análisis batch ──────────────────────────────────────

class RepositorioJobs:
    """CRUD sobre la tabla 'video_jobs' — trabajos de análisis batch."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def crear(self, datos: dict) -> VideoJob:
        """Crea un nuevo job y lo persiste. Retorna la instancia."""
        job = VideoJob(**datos)
        self.session.add(job)
        self.session.flush()
        return job

    def obtener(self, job_id: int) -> VideoJob | None:
        return self.session.get(VideoJob, job_id)

    def actualizar(self, job_id: int, campos: dict) -> None:
        """Actualiza campos de un job. No hace commit (lo gestiona la sesión)."""
        job = self.session.get(VideoJob, job_id)
        if job is not None:
            for k, v in campos.items():
                setattr(job, k, v)
            self.session.flush()

    def listar_todos(self) -> list[VideoJob]:
        """Lista todos los jobs ordenados por id descendente (más recientes primero)."""
        return self.session.query(VideoJob).order_by(VideoJob.id.desc()).all()

    def listar_por_estado(self, estado: str) -> list[VideoJob]:
        return self.session.query(VideoJob).filter(VideoJob.estado == estado).all()

    def listar_completados(self) -> list[VideoJob]:
        """Jobs con video procesado disponible para visualizar."""
        return (
            self.session.query(VideoJob)
            .filter(VideoJob.estado == "completado")
            .order_by(VideoJob.id.desc())
            .all()
        )

    def eliminar(self, job_id: int) -> dict:
        """Elimina un job y sus archivos asociados. Retorna info de lo borrado."""
        job = self.session.get(VideoJob, job_id)
        if job is None:
            return {}

        rutas = {
            "ruta_original":    job.ruta_original,
            "ruta_salida":       job.ruta_salida,
            "ruta_reporte_pdf":  job.ruta_reporte_pdf,
            "ruta_reporte_xlsx": job.ruta_reporte_xlsx,
        }
        info = {
            "nombre":   job.nombre_original,
            "estado":    job.estado,
            "rutas":     {k: v for k, v in rutas.items() if v},
        }
        self.session.delete(job)
        self.session.flush()
        return info

    def marcar_interrumpidos(self) -> int:
        """Al arrancar la app, marca como 'error' los jobs que quedaron en procesando."""
        jobs = (
            self.session.query(VideoJob)
            .filter(VideoJob.estado.in_(["pendiente", "procesando", "generando_video"]))
            .all()
        )
        import datetime
        ahora = datetime.datetime.now().isoformat(timespec="seconds")
        for job in jobs:
            job.estado = "error"
            job.error_msg = "Interrumpido por cierre de la aplicación"
            job.finalizado_en = ahora
        self.session.flush()
        return len(jobs)
