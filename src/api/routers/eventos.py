"""
Router de eventos: endpoints para consultar detecciones registradas.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.schemas import EventoSchema
from src.database.db import get_session_dep

router = APIRouter(prefix="/eventos", tags=["Eventos"])


@router.get("/health")
def health_check():
    """Verifica que la API está operativa."""
    return {"status": "ok"}


@router.get("", response_model=list[EventoSchema])
def listar_eventos(
    tipo_maquina:      str  | None = Query(None, description="Filtrar por tipo (camion, excavadora, pala)"),
    clase_id:          str  | None = Query(None, description="Filtrar por clase YOLO (dump_truck, excavator...)"),
    zona_nombre:       str  | None = Query(None, description="Filtrar por zona operativa"),
    tipo_evento_ciclo: str  | None = Query(None, description="Filtrar por evento (aparicion, desaparicion, entrada_carga...)"),
    limit:             int         = Query(100,  ge=1, le=5000, description="Máximo de resultados"),
    offset:            int         = Query(0,    ge=0,          description="Desplazamiento para paginación"),
    db: Session = Depends(get_session_dep),
):
    """
    Retorna eventos registrados con filtros opcionales y paginación.

    Filtros disponibles:
    - tipo_maquina: camion | excavadora | pala
    - clase_id: dump_truck | excavator | wheel_loader
    - zona_nombre: nombre de la zona operativa
    - tipo_evento_ciclo: aparicion | desaparicion | entrada_carga | salida_carga | entrada_descarga | salida_descarga
    """
    from src.database.models import Evento

    try:
        q = db.query(Evento)
        if tipo_maquina:
            q = q.filter(Evento.tipo_maquina == tipo_maquina)
        if clase_id:
            q = q.filter(Evento.clase_id == clase_id)
        if zona_nombre:
            q = q.filter(Evento.zona_nombre == zona_nombre)
        if tipo_evento_ciclo:
            q = q.filter(Evento.tipo_evento_ciclo == tipo_evento_ciclo)

        return (
            q.order_by(Evento.timestamp.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/{evento_id}", response_model=EventoSchema)
def obtener_evento(evento_id: int, db: Session = Depends(get_session_dep)):
    """
    Retorna un evento por su ID.

    Raises:
        404: si el evento no existe.
        503: si la base de datos no está disponible.
    """
    from src.database.models import Evento

    try:
        evento = db.get(Evento, evento_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e

    if evento is None:
        raise HTTPException(status_code=404, detail="Evento no encontrado")

    return evento
