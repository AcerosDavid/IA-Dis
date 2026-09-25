"""
Router de zonas: CRUD de zonas operativas de geofencing.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import ZonaSchema, ZonaCreateSchema
from src.database.db import get_session_dep

router = APIRouter(prefix="/zonas", tags=["Zonas"])


@router.get("", response_model=list[ZonaSchema])
def listar_zonas(db: Session = Depends(get_session_dep)):
    """Retorna todas las zonas registradas."""
    from src.database.models import ZonaDB
    try:
        return db.query(ZonaDB).all()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/{zona_id}", response_model=ZonaSchema)
def obtener_zona(zona_id: int, db: Session = Depends(get_session_dep)):
    """Retorna una zona por ID."""
    from src.database.models import ZonaDB
    zona = db.get(ZonaDB, zona_id)
    if zona is None:
        raise HTTPException(status_code=404, detail="Zona no encontrada")
    return zona


@router.post("", response_model=ZonaSchema, status_code=201)
def crear_zona(zona: ZonaCreateSchema, db: Session = Depends(get_session_dep)):
    """
    Crea una nueva zona operativa.

    vertices_json debe ser un array JSON de puntos [x, y]:
    Ejemplo: "[[100,100],[400,100],[400,300],[100,300]]"
    """
    from src.database.models import ZonaDB
    import json

    # Validar JSON de vértices
    try:
        vertices = json.loads(zona.vertices_json)
        if len(vertices) < 3:
            raise ValueError("Necesita al menos 3 vértices")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"vertices_json inválido: {e}")

    existente = db.query(ZonaDB).filter(ZonaDB.nombre == zona.nombre).first()
    if existente:
        raise HTTPException(status_code=409, detail=f"Ya existe una zona con nombre '{zona.nombre}'")

    try:
        nueva = ZonaDB(**zona.model_dump())
        db.add(nueva)
        db.flush()
        db.refresh(nueva)
        return nueva
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.put("/{zona_id}", response_model=ZonaSchema)
def actualizar_zona(zona_id: int, zona: ZonaCreateSchema, db: Session = Depends(get_session_dep)):
    """Actualiza una zona existente."""
    from src.database.models import ZonaDB

    existente = db.get(ZonaDB, zona_id)
    if existente is None:
        raise HTTPException(status_code=404, detail="Zona no encontrada")

    try:
        for campo, valor in zona.model_dump().items():
            setattr(existente, campo, valor)
        db.flush()
        db.refresh(existente)
        return existente
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.delete("/{zona_id}", status_code=204)
def eliminar_zona(zona_id: int, db: Session = Depends(get_session_dep)):
    """Elimina una zona por ID."""
    from src.database.models import ZonaDB

    zona = db.get(ZonaDB, zona_id)
    if zona is None:
        raise HTTPException(status_code=404, detail="Zona no encontrada")
    try:
        db.delete(zona)
        db.flush()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e
