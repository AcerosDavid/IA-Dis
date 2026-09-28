"""
Capa de base de datos centralizada.

Uso:
    from src.database.db import get_session, get_engine

    # En FastAPI (dependencia inyectable):
    def mi_endpoint(db: Session = Depends(get_session_dep)):
        ...

    # Fuera de FastAPI (scripts, main.py, camera.py):
    with get_session() as db:
        repo = RepositorioEventos(db)
        ...

Configuración en .env:
    SQLite:     DB_URL=sqlite:///data/eventos.db
    MySQL:      DB_URL=mysql+pymysql://user:pass@localhost:3306/mining
    PostgreSQL: DB_URL=postgresql+psycopg2://user:pass@localhost:5432/mining
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, Evento


def _build_engine(db_url: str):
    """Crea el engine SQLAlchemy a partir de la URL de conexión."""
    kwargs: dict = {}

    # SQLite necesita check_same_thread=False para uso en hilos (FastAPI, etc.)
    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(db_url, **kwargs)

    # Habilita WAL + foreign keys para SQLite (sin efecto en otros gestores)
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _record):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

    return engine


def _get_db_url() -> str:
    """Lee DB_URL desde la configuración del proyecto."""
    from src.config import load_config
    return load_config().db_url


# Engine y SessionFactory — lazy, se crean al primer uso
_engine = None
_SessionFactory = None


def _init() -> None:
    """Inicializa engine y SessionFactory si aún no están creados."""
    global _engine, _SessionFactory
    if _engine is None:
        _engine = _build_engine(_get_db_url())
        _SessionFactory = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def get_engine():
    """Retorna el engine SQLAlchemy (lo inicializa si es la primera llamada)."""
    _init()
    return _engine


def migrate() -> None:
    """
    Crea todas las tablas que no existan (equivalente a migrate en Laravel).
    Usar solo en desarrollo; en producción usa las migraciones de Alembic.
    """
    _init()
    Base.metadata.create_all(_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager que provee una sesión con commit/rollback automático.

    Uso en scripts:
        with get_session() as db:
            repo = RepositorioEventos(db)
            repo.insertar(...)
    """
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


def get_session_dep() -> Generator[Session, None, None]:
    """
    Generador para inyección de dependencias en FastAPI (NO usar con 'with').

    Uso:
        from fastapi import Depends
        from src.database.db import get_session_dep

        def mi_endpoint(db: Session = Depends(get_session_dep)):
            ...
    """
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
    """
    CRUD sobre la tabla 'eventos'.
    Recibe una sesión inyectada — no gestiona transacciones propias.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def insertar(self, evento: dict) -> None:
        """
        Inserta un evento usando un savepoint (begin_nested) para aislar el fallo:
        si el insert falla, solo ese insert se deshace —
        el resto de la sesión queda intacto.
        """
        try:
            with self.session.begin_nested():   # savepoint real — no invalida la sesión
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

    def obtener_por_id(self, evento_id: int) -> Evento | None:
        """Retorna el evento con el id dado, o None si no existe."""
        return self.session.get(Evento, evento_id)


# ── Repositorio de Zonas ──────────────────────────────────────────────────────

class RepositorioZonas:
    """
    CRUD sobre la tabla 'zonas'.
    Convierte entre ZonaDB (ORM) y Zona (geometry) automáticamente.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def cargar_activas(self):
        """
        Retorna lista de instancias Zona (geometry) cargadas desde la DB.
        Solo devuelve zonas con activa=True.
        """
        import json
        from src.database.models import ZonaDB
        from src.core.geometry import Zona

        rows = self.session.query(ZonaDB).filter(ZonaDB.activa == True).all()
        zonas = []
        for row in rows:
            vertices = json.loads(row.vertices_json)
            # Convertir color_hex (#RRGGBB) → BGR tuple para OpenCV
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
        """Persiste una Zona (geometry) en la DB. Crea o actualiza."""
        import json
        from src.database.models import ZonaDB

        # Convertir BGR → hex
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
                nombre        = zona.nombre,
                tipo          = zona.tipo,
                vertices_json = json.dumps(zona.vertices),
                color_hex     = color_hex,
                activa        = zona.activa,
            ))
        self.session.flush()

    def obtener_todas(self):
        """Retorna todos los registros ZonaDB."""
        from src.database.models import ZonaDB
        return self.session.query(ZonaDB).all()


# ── Repositorio de Ciclos ─────────────────────────────────────────────────────

class RepositorioCiclos:
    """
    Gestiona la construcción y consulta de ciclos operativos.

    Un ciclo se construye progresivamente a medida que llegan EventoCiclo:
        entrada_carga    → abre ciclo
        salida_carga     → registra ts_salida_carga
        entrada_descarga → registra ts_entrada_descarga
        salida_descarga  → cierra ciclo, calcula duración
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def registrar_evento(self, evento_ciclo) -> None:
        """
        Procesa un EventoCiclo (geometry) y actualiza el ciclo activo
        de esa máquina en la DB.

        Args:
            evento_ciclo: instancia de EventoCiclo de src.core.geometry
        """
        from src.database.models import Ciclo

        maquina_id  = evento_ciclo.maquina_id
        tipo_evento = evento_ciclo.tipo_evento
        timestamp   = evento_ciclo.timestamp

        # Buscar ciclo activo (no completado) para esta máquina
        ciclo_activo = (
            self.session.query(Ciclo)
            .filter(Ciclo.maquina_id == maquina_id, Ciclo.completado == False)
            .first()
        )

        if tipo_evento == "entrada_carga":
            if ciclo_activo is None:
                # Abrir nuevo ciclo
                ciclo = Ciclo(
                    maquina_id       = maquina_id,
                    tipo_maquina     = evento_ciclo.tipo_maquina,
                    ts_entrada_carga = timestamp,
                    zona_carga       = evento_ciclo.zona_nombre,
                )
                self.session.add(ciclo)
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
                # Calcular duración total si tenemos ts inicio y fin
                if ciclo_activo.ts_entrada_carga:
                    try:
                        dur = _segundos_entre(
                            ciclo_activo.ts_entrada_carga, timestamp
                        )
                        ciclo_activo.duracion_segundos = dur
                    except Exception:
                        pass
                ciclo_activo.completado = True
                self.session.flush()

    def obtener_todos(self, completado: bool | None = None) -> list:
        """Retorna ciclos, opcionalmente filtrados por estado completado."""
        from src.database.models import Ciclo
        q = self.session.query(Ciclo)
        if completado is not None:
            q = q.filter(Ciclo.completado == completado)
        return q.order_by(Ciclo.id.desc()).all()

    def obtener_por_maquina(self, maquina_id: str) -> list:
        """Retorna todos los ciclos de una máquina específica."""
        from src.database.models import Ciclo
        return (
            self.session.query(Ciclo)
            .filter(Ciclo.maquina_id == maquina_id)
            .order_by(Ciclo.id.desc())
            .all()
        )

    def estadisticas(self) -> dict:
        """
        Retorna estadísticas agregadas de ciclos completados:
        total, duracion_promedio, duracion_min, duracion_max por tipo_maquina.
        """
        from src.database.models import Ciclo
        from sqlalchemy import func

        rows = (
            self.session.query(
                Ciclo.tipo_maquina,
                func.count(Ciclo.id).label("total"),
                func.avg(Ciclo.duracion_segundos).label("duracion_promedio"),
                func.min(Ciclo.duracion_segundos).label("duracion_min"),
                func.max(Ciclo.duracion_segundos).label("duracion_max"),
            )
            .filter(Ciclo.completado == True)
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
    """
    Calcula segundos entre dos timestamps en formato 'HH:MM:SS.mmm'.
    Retorna 0.0 si no se puede calcular.
    """
    def _parse(ts: str) -> float:
        partes = ts.replace(",", ".").split(":")
        h = int(partes[0])
        m = int(partes[1])
        s = float(partes[2])
        return h * 3600 + m * 60 + s

    try:
        return max(0.0, _parse(ts_fin) - _parse(ts_inicio))
    except Exception:
        return 0.0
