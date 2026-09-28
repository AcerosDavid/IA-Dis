"""
Tests unitarios y de propiedad para RepositorioEventos.
Usa SQLite en memoria via fixtures de conftest.py.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.database.db import RepositorioEventos
from src.database.models import Base, Evento


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_evento(**kwargs) -> dict:
    base = {
        "tipo_maquina": "camion",
        "clase_id": "truck",
        "confianza": 0.9,
        "numero_fotograma": 0,
        "timestamp": "00:00:00.000",
    }
    base.update(kwargs)
    return base


# ─── Smoke tests ─────────────────────────────────────────────────────────────

def test_init_db_idempotente(tmp_path):
    """migrate() puede llamarse múltiples veces sin error."""
    import os
    from sqlalchemy import create_engine
    from src.database.models import Base as _Base
    db_path = str(tmp_path / "test.db")
    url = f"sqlite:///{db_path}"
    eng = create_engine(url, connect_args={"check_same_thread": False})
    _Base.metadata.create_all(eng)
    _Base.metadata.create_all(eng)  # segunda llamada no debe fallar
    eng.dispose()


def test_insertar_y_obtener_todos(repo):
    """Insertar un evento y recuperarlo con obtener_todos."""
    repo.insertar(make_evento())
    eventos = repo.obtener_todos()
    assert len(eventos) == 1
    assert eventos[0].tipo_maquina == "camion"


def test_obtener_por_id_existe(repo):
    """obtener_por_id retorna el evento correcto."""
    repo.insertar(make_evento(tipo_maquina="pala", clase_id="excavator"))
    eventos = repo.obtener_todos()
    e = repo.obtener_por_id(eventos[0].id)
    assert e is not None
    assert e.tipo_maquina == "pala"


def test_obtener_por_id_no_existe(repo):
    """obtener_por_id retorna None para un ID inexistente."""
    assert repo.obtener_por_id(9999) is None


def test_insertar_fallido_no_lanza_excepcion(repo, capsys):
    """insertar con datos inválidos loguea [DB ERROR] y no lanza excepción."""
    # Forzar fallo pasando un campo inexistente en el ORM
    repo.insertar({"campo_invalido": "valor"})
    captured = capsys.readouterr()
    assert "[DB ERROR]" in captured.out


# ─── Property tests ──────────────────────────────────────────────────────────

tipo_maquina_st = st.sampled_from(["camion", "pala"])
clase_id_st = st.sampled_from(["truck", "excavator"])
confianza_st = st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
fotograma_st = st.integers(min_value=0, max_value=10000)
timestamp_st = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Nd", "Ll", "Lu"), whitelist_characters=":."))


@given(
    tipo_maquina=tipo_maquina_st,
    clase_id=clase_id_st,
    confianza=confianza_st,
    numero_fotograma=fotograma_st,
    timestamp=timestamp_st,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_5_round_trip_persistencia(
    db_session, tipo_maquina, clase_id, confianza, numero_fotograma, timestamp
):
    """
    Property 5: round-trip de persistencia.
    Para todo evento válido, insertar + obtener_todos retorna el evento con campos intactos.

    Validates: Requirements 4.3, 4.5, 4.6
    """
    # Limpiar DB entre iteraciones
    db_session.query(Evento).delete()
    db_session.commit()

    repo = RepositorioEventos(db_session)
    evento = {
        "tipo_maquina": tipo_maquina,
        "clase_id": clase_id,
        "confianza": confianza,
        "numero_fotograma": numero_fotograma,
        "timestamp": timestamp,
    }
    repo.insertar(evento)
    eventos = repo.obtener_todos()

    assert len(eventos) == 1
    e = eventos[0]
    assert e.tipo_maquina == tipo_maquina
    assert e.clase_id == clase_id
    assert abs(e.confianza - confianza) < 1e-6
    assert e.numero_fotograma == numero_fotograma
    assert e.timestamp == timestamp


@given(
    timestamps=st.lists(
        timestamp_st,
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_6_ordenamiento_por_timestamp(db_session, timestamps):
    """
    Property 6: obtener_todos retorna eventos ordenados por numero_fotograma ASC.

    El campo timestamp es un string libre (formato HH:MM:SS.mmm) cuyo orden
    lexicografico no es fiable para timestamps arbitrarios. El repositorio
    ordena por numero_fotograma (entero) que es siempre correcto.

    Validates: Requirements 4.5
    """
    db_session.query(Evento).delete()
    db_session.commit()

    repo = RepositorioEventos(db_session)
    for i, ts in enumerate(timestamps):
        repo.insertar(make_evento(timestamp=ts, numero_fotograma=i))

    eventos = repo.obtener_todos()
    assert len(eventos) == len(timestamps)
    for i in range(len(eventos) - 1):
        assert eventos[i].numero_fotograma <= eventos[i + 1].numero_fotograma
