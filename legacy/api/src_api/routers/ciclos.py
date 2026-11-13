"""
Router de ciclos: consulta de ciclos operativos de maquinaria.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.schemas import CicloSchema
from src.database.db import RepositorioCiclos, get_session_dep

router = APIRouter(prefix="/ciclos", tags=["Ciclos"])


@router.get("", response_model=list[CicloSchema])
def listar_ciclos(
    completado:   bool | None = Query(None, description="Filtrar por completado (true/false)"),
    maquina_id:   str  | None = Query(None, description="Filtrar por ID de máquina (ej: camion_1)"),
    tipo_maquina: str  | None = Query(None, description="Filtrar por tipo (camion, excavadora, pala)"),
    limit:        int         = Query(100,  ge=1, le=1000, description="Máximo de resultados"),
    offset:       int         = Query(0,    ge=0,          description="Desplazamiento para paginación"),
    db: Session = Depends(get_session_dep),
):
    """
    Retorna ciclos operativos con filtros opcionales.

    Un ciclo representa el recorrido completo de una máquina:
    entrada_carga → salida_carga → entrada_descarga → salida_descarga
    """
    from src.database.models import Ciclo

    try:
        q = db.query(Ciclo)
        if completado is not None:
            q = q.filter(Ciclo.completado == completado)
        if maquina_id:
            q = q.filter(Ciclo.maquina_id == maquina_id)
        if tipo_maquina:
            q = q.filter(Ciclo.tipo_maquina == tipo_maquina)
        return q.order_by(Ciclo.id.desc()).offset(offset).limit(limit).all()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/{ciclo_id}", response_model=CicloSchema)
def obtener_ciclo(ciclo_id: int, db: Session = Depends(get_session_dep)):
    """Retorna un ciclo por su ID."""
    from src.database.models import Ciclo

    ciclo = db.get(Ciclo, ciclo_id)
    if ciclo is None:
        raise HTTPException(status_code=404, detail="Ciclo no encontrado")
    return ciclo
