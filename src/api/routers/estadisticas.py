"""
Router de estadísticas: resumen y métricas del sistema.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.api.schemas import (
    ResumenSchema,
    EstadisticasTipoSchema,
    EstadisticasZonaSchema,
    EstadisticasCiclosSchema,
)
from src.database.db import RepositorioCiclos, get_session_dep

router = APIRouter(prefix="/estadisticas", tags=["Estadísticas"])


@router.get("/resumen", response_model=ResumenSchema)
def resumen(db: Session = Depends(get_session_dep)):
    """
    Retorna un resumen general del sistema:
    - Total de eventos registrados
    - Total y completados de ciclos
    - Conteo de eventos por tipo de máquina
    - Conteo de eventos por zona
    """
    from src.database.models import Evento, Ciclo

    try:
        # Totales
        total_eventos = db.query(func.count(Evento.id)).scalar() or 0
        total_ciclos  = db.query(func.count(Ciclo.id)).scalar() or 0
        ciclos_completados = (
            db.query(func.count(Ciclo.id))
            .filter(Ciclo.completado == True)
            .scalar() or 0
        )

        # Por tipo de máquina
        por_tipo_rows = (
            db.query(Evento.tipo_maquina, func.count(Evento.id).label("total"))
            .group_by(Evento.tipo_maquina)
            .all()
        )
        por_tipo = [
            EstadisticasTipoSchema(tipo_maquina=r.tipo_maquina, total=r.total)
            for r in por_tipo_rows
        ]

        # Por zona (solo eventos con zona registrada)
        por_zona_rows = (
            db.query(Evento.zona_nombre, func.count(Evento.id).label("total"))
            .filter(Evento.zona_nombre != None)
            .group_by(Evento.zona_nombre)
            .all()
        )
        por_zona = [
            EstadisticasZonaSchema(zona_nombre=r.zona_nombre, total=r.total)
            for r in por_zona_rows
        ]

        return ResumenSchema(
            total_eventos      = total_eventos,
            total_ciclos       = total_ciclos,
            ciclos_completados = ciclos_completados,
            por_tipo           = por_tipo,
            por_zona           = por_zona,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/ciclos", response_model=list[EstadisticasCiclosSchema])
def estadisticas_ciclos(db: Session = Depends(get_session_dep)):
    """
    Estadísticas de ciclos completados por tipo de máquina:
    total, duración promedio, mínima y máxima en segundos.
    """
    try:
        repo = RepositorioCiclos(db)
        return [EstadisticasCiclosSchema(**row) for row in repo.estadisticas()]
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e


@router.get("/productividad")
def productividad(db: Session = Depends(get_session_dep)):
    """
    Ciclos completados por hora para cada tipo de máquina.
    Basado en la duración promedio de ciclo.
    """
    from src.database.models import Ciclo
    from sqlalchemy import func

    try:
        rows = (
            db.query(
                Ciclo.tipo_maquina,
                func.count(Ciclo.id).label("total_ciclos"),
                func.avg(Ciclo.duracion_segundos).label("duracion_promedio"),
            )
            .filter(Ciclo.completado == True, Ciclo.duracion_segundos > 0)
            .group_by(Ciclo.tipo_maquina)
            .all()
        )

        resultado = []
        for r in rows:
            ciclos_por_hora = (
                round(3600 / r.duracion_promedio, 2)
                if r.duracion_promedio and r.duracion_promedio > 0
                else 0
            )
            resultado.append({
                "tipo_maquina":       r.tipo_maquina,
                "total_ciclos":       r.total_ciclos,
                "duracion_promedio_s": round(r.duracion_promedio or 0, 1),
                "ciclos_por_hora":    ciclos_por_hora,
            })

        return resultado
    except Exception as e:
        raise HTTPException(status_code=503, detail="Base de datos no disponible") from e
