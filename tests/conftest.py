"""
Fixtures compartidos para la suite de tests.
Usa SQLite en memoria para evitar efectos secundarios entre tests.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.database import RepositorioEventos
from src.database.models import Base


# ─── Fixtures de base de datos ───────────────────────────────────────────────

@pytest.fixture
def db_engine():
    """Engine SQLite en memoria con StaticPool para compartir conexión entre hilos."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # Garantiza que todos los hilos usen la misma conexión
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Sesión de base de datos enlazada al engine en memoria."""
    SessionFactory = sessionmaker(bind=db_engine)
    session = SessionFactory()
    yield session
    session.close()


@pytest.fixture
def repo(db_session):
    """Instancia de RepositorioEventos usando la sesión en memoria."""
    return RepositorioEventos(db_session)


# ─── Fixture de API ───────────────────────────────────────────────────────────

@pytest.fixture
def test_client(db_session):
    """
    TestClient de FastAPI con override de la dependencia de DB
    para usar la sesión en memoria en lugar de la DB real.
    """
    from src.api.main import app
    from src.api.routers.eventos import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# ─── Fixture de evento de ejemplo ────────────────────────────────────────────

@pytest.fixture
def evento_ejemplo():
    """Dict con datos de un evento válido para usar en tests."""
    return {
        "tipo_maquina": "camion",
        "clase_id": "truck",
        "confianza": 0.87,
        "numero_fotograma": 42,
        "timestamp": "00:00:01.400",
    }
