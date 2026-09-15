"""
Tests de integración para la API FastAPI.
Usa TestClient con DB en memoria (via fixture test_client de conftest.py).
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.database.database import RepositorioEventos
from src.database.models import Evento


# ─── Smoke / example tests ───────────────────────────────────────────────────

def test_health_check(test_client):
    """GET /health retorna {"status": "ok"} con HTTP 200."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_listar_eventos_vacio(test_client):
    """GET /eventos con DB vacía retorna [] y HTTP 200."""
    resp = test_client.get("/eventos")
    assert resp.status_code == 200
    assert resp.json() == []


def test_listar_eventos_con_datos(test_client, repo, evento_ejemplo):
    """GET /eventos retorna los eventos insertados."""
    repo.insertar(evento_ejemplo)
    resp = test_client.get("/eventos")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["tipo_maquina"] == "camion"


def test_obtener_evento_por_id(test_client, repo, evento_ejemplo):
    """GET /eventos/{id} retorna el evento correcto."""
    repo.insertar(evento_ejemplo)
    eventos = repo.obtener_todos()
    evento_id = eventos[0].id

    resp = test_client.get(f"/eventos/{evento_id}")
    assert resp.status_code == 200
    assert resp.json()["tipo_maquina"] == "camion"
    assert resp.json()["clase_id"] == "truck"


def test_obtener_evento_no_existente(test_client):
    """GET /eventos/{id} retorna 404 con mensaje correcto para ID inexistente."""
    resp = test_client.get("/eventos/9999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Evento no encontrado"


# ─── Property tests ──────────────────────────────────────────────────────────

timestamp_st = st.text(
    min_size=1, max_size=15,
    alphabet=st.characters(whitelist_categories=("Nd",), whitelist_characters=":."),
)


@given(
    n=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_8_get_eventos_retorna_todos(test_client, db_session, n):
    """
    Property 8: GET /eventos retorna exactamente N eventos tras insertar N.

    Validates: Requirements 6.1, 6.3
    """
    db_session.query(Evento).delete()
    db_session.commit()

    repo = RepositorioEventos(db_session)
    for i in range(n):
        repo.insertar({
            "tipo_maquina": "camion",
            "clase_id": "truck",
            "confianza": 0.8,
            "numero_fotograma": i,
            "timestamp": f"00:00:{i:05.2f}",
        })

    resp = test_client.get("/eventos")
    assert resp.status_code == 200
    assert len(resp.json()) == n


@given(
    confianza=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    fotograma=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_9_round_trip_por_id(test_client, db_session, confianza, fotograma):
    """
    Property 9: GET /eventos/{id} retorna exactamente los datos insertados.

    Validates: Requirements 6.4
    """
    db_session.query(Evento).delete()
    db_session.commit()

    repo = RepositorioEventos(db_session)
    repo.insertar({
        "tipo_maquina": "pala",
        "clase_id": "excavator",
        "confianza": confianza,
        "numero_fotograma": fotograma,
        "timestamp": "00:00:01.000",
    })
    eventos = repo.obtener_todos()
    evento_id = eventos[0].id

    resp = test_client.get(f"/eventos/{evento_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tipo_maquina"] == "pala"
    assert data["clase_id"] == "excavator"
    assert abs(data["confianza"] - confianza) < 1e-5
    assert data["numero_fotograma"] == fotograma
