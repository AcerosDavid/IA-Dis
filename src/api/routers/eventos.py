"""
Router de eventos: endpoints para consultar detecciones registradas.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import EventoSchema
from src.database.database import RepositorioEventos, get_session_factory
from src.config import load_config

router = APIRouter()


def get_db():
    """Dependencia FastAPI: provee una sesión de base de datos por request."""
    try:
        config = load_config()
        SessionFactory = get_session_factory(config.db_path)
        session = SessionFactory()
        try:
            yield session
        finally:
            session.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/health")
def health_check():
    """Verifica que la API está operativa."""
    return {"status": "ok"}


@router.get("/eventos", response_model=list[EventoSchema])
def listar_eventos(db: Session = Depends(get_db)):
    """
    Retorna todos los eventos registrados ordenados por timestamp ASC.

    Returns:
        Lista de EventoSchema. Lista vacía si no hay eventos.
    """
    try:
        repo = RepositorioEventos(db)
        return repo.obtener_todos()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/eventos/{evento_id}", response_model=EventoSchema)
def obtener_evento(evento_id: int, db: Session = Depends(get_db)):
    """
    Retorna un evento por su ID.

    Args:
        evento_id: ID del evento a buscar.

    Returns:
        EventoSchema con los datos del evento.

    Raises:
        404: si el evento no existe.
        503: si la base de datos no está disponible.
    """
    try:
        repo = RepositorioEventos(db)
        evento = repo.obtener_por_id(evento_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e

    if evento is None:
        raise HTTPException(status_code=404, detail="Evento no encontrado")

    return evento
