"""
Router de eventos: endpoints para consultar detecciones registradas.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import EventoSchema
from src.database.db import RepositorioEventos, get_session_dep

router = APIRouter()


@router.get("/health")
def health_check():
    """Verifica que la API está operativa."""
    return {"status": "ok"}


@router.get("/eventos", response_model=list[EventoSchema])
def listar_eventos(db: Session = Depends(get_session_dep)):
    """
    Retorna todos los eventos registrados ordenados por timestamp ASC.
    """
    try:
        return RepositorioEventos(db).obtener_todos()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/eventos/{evento_id}", response_model=EventoSchema)
def obtener_evento(evento_id: int, db: Session = Depends(get_session_dep)):
    """
    Retorna un evento por su ID.

    Raises:
        404: si el evento no existe.
        503: si la base de datos no está disponible.
    """
    try:
        evento = RepositorioEventos(db).obtener_por_id(evento_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e

    if evento is None:
        raise HTTPException(status_code=404, detail="Evento no encontrado")

    return evento
