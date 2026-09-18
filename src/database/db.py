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
        Inserta un evento. Usa savepoint para aislar el fallo:
        si el insert falla, solo ese insert se deshace —
        el resto de la sesión queda intacto.
        """
        try:
            self.session.add(Evento(**evento))
            self.session.flush()
        except Exception as e:
            # rollback solo hasta el último savepoint (no toda la sesión)
            self.session.rollback()
            print(f"[DB ERROR] No se pudo insertar evento: {evento} | {e}")

    def obtener_todos(self) -> list[Evento]:
        """Retorna todos los eventos ordenados por timestamp ASC."""
        return (
            self.session.query(Evento)
            .order_by(Evento.timestamp.asc())
            .all()
        )

    def obtener_por_id(self, evento_id: int) -> Evento | None:
        """Retorna el evento con el id dado, o None si no existe."""
        return self.session.get(Evento, evento_id)
