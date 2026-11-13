"""
Router de Turnos — gestión de turnos operacionales de la mina.

El modelo Turno ya existía en la DB (relacionado con Ciclo).
Este router expone el CRUD completo que faltaba.

Regla de negocio:
  Solo puede haber UN turno activo a la vez.
  Cerrar el turno activo calcula el total de ciclos completados en ese periodo.

Endpoints:
  GET    /turnos              — listar turnos con filtros
  GET    /turnos/activo       — turno activo actual (o 404)
  GET    /turnos/{id}         — detalle de un turno
  POST   /turnos              — abrir nuevo turno
  PATCH  /turnos/{id}/cerrar  — cerrar turno activo
  DELETE /turnos/{id}         — eliminar turno (solo si sin ciclos)
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.db import get_session_dep
from src.database.models import Turno, Ciclo

router = APIRouter(prefix="/turnos", tags=["Turnos"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class TurnoOut(BaseModel):
    """Respuesta de un turno."""
    id:           int
    tipo:         str
    fecha:        str
    ts_inicio:    str
    ts_fin:       str | None = None
    activo:       bool
    total_ciclos: int
    ciclos_completados: int = 0   # calculado en el momento


class TurnoCreate(BaseModel):
    """Payload para abrir un nuevo turno."""
    tipo:  str = Field(pattern="^(manana|tarde|noche)$",
                       description="Turno: manana | tarde | noche")
    fecha: str = Field(description="Fecha del turno en formato YYYY-MM-DD")


class TurnoListOut(BaseModel):
    items: list[TurnoOut]
    total: int


# ── Helper ────────────────────────────────────────────────────────────────────

def _turno_to_out(t: Turno, db: Session) -> TurnoOut:
    ciclos_completados = (
        db.query(func.count(Ciclo.id))
        .filter(Ciclo.turno_id == t.id, Ciclo.completado == True)
        .scalar() or 0
    )
    return TurnoOut(
        id                 = t.id,
        tipo               = t.tipo,
        fecha              = t.fecha,
        ts_inicio          = t.ts_inicio,
        ts_fin             = t.ts_fin,
        activo             = t.activo,
        total_ciclos       = t.total_ciclos,
        ciclos_completados = ciclos_completados,
    )


def _ts_ahora() -> str:
    """Timestamp actual en formato HH:MM:SS."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _fecha_hoy() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=TurnoListOut)
def listar_turnos(
    activo: bool | None = Query(None, description="Filtrar por activo/inactivo"),
    fecha:  str | None  = Query(None, description="Filtrar por fecha YYYY-MM-DD"),
    tipo:   str | None  = Query(None, description="Filtrar por tipo: manana|tarde|noche"),
    page:      int      = Query(1, ge=1),
    page_size: int      = Query(20, ge=1, le=100),
    db: Session = Depends(get_session_dep),
):
    """Lista turnos con filtros opcionales."""
    q = db.query(Turno)
    if activo is not None:
        q = q.filter(Turno.activo == activo)
    if fecha:
        q = q.filter(Turno.fecha == fecha)
    if tipo:
        q = q.filter(Turno.tipo == tipo)

    total   = q.count()
    turnos  = (
        q.order_by(Turno.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return TurnoListOut(
        items = [_turno_to_out(t, db) for t in turnos],
        total = total,
    )


@router.get("/activo", response_model=TurnoOut)
def turno_activo(db: Session = Depends(get_session_dep)):
    """
    Retorna el turno activo actual.
    404 si no hay ninguno abierto en este momento.
    """
    t = db.query(Turno).filter(Turno.activo == True).first()
    if t is None:
        raise HTTPException(404, "No hay ningún turno activo en este momento")
    return _turno_to_out(t, db)


@router.get("/{turno_id}", response_model=TurnoOut)
def obtener_turno(turno_id: int, db: Session = Depends(get_session_dep)):
    """Retorna el detalle de un turno específico."""
    t = db.get(Turno, turno_id)
    if t is None:
        raise HTTPException(404, "Turno no encontrado")
    return _turno_to_out(t, db)


@router.post("", response_model=TurnoOut, status_code=201)
def abrir_turno(datos: TurnoCreate, db: Session = Depends(get_session_dep)):
    """
    Abre un nuevo turno.

    Reglas:
      - Solo puede haber un turno activo a la vez.
      - Si ya hay uno activo, se devuelve 409 — ciérralo primero.
      - La fecha por defecto es hoy (UTC).
    """
    # Verificar que no haya otro activo
    activo_existente = db.query(Turno).filter(Turno.activo == True).first()
    if activo_existente:
        raise HTTPException(
            409,
            f"Ya hay un turno activo: ID={activo_existente.id} "
            f"({activo_existente.tipo} {activo_existente.fecha}). "
            f"Ciérralo con PATCH /turnos/{activo_existente.id}/cerrar antes de abrir uno nuevo.",
        )

    # Validar fecha
    try:
        datetime.strptime(datos.fecha, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, f"Formato de fecha inválido: '{datos.fecha}'. Usa YYYY-MM-DD.")

    nuevo = Turno(
        tipo       = datos.tipo,
        fecha      = datos.fecha,
        ts_inicio  = _ts_ahora(),
        activo     = True,
        total_ciclos = 0,
    )
    db.add(nuevo)
    db.flush()

    return _turno_to_out(nuevo, db)


@router.patch("/{turno_id}/cerrar", response_model=TurnoOut)
def cerrar_turno(turno_id: int, db: Session = Depends(get_session_dep)):
    """
    Cierra un turno activo:
      - Registra ts_fin con la hora actual.
      - Calcula y persiste total_ciclos completados.
      - Marca activo = False.

    Solo se puede cerrar el turno activo.
    """
    t = db.get(Turno, turno_id)
    if t is None:
        raise HTTPException(404, "Turno no encontrado")
    if not t.activo:
        raise HTTPException(
            409,
            f"El turno {turno_id} ya está cerrado (ts_fin={t.ts_fin}).",
        )

    # Contar ciclos completados durante este turno
    ciclos_completados = (
        db.query(func.count(Ciclo.id))
        .filter(Ciclo.turno_id == turno_id, Ciclo.completado == True)
        .scalar() or 0
    )

    t.ts_fin       = _ts_ahora()
    t.activo       = False
    t.total_ciclos = ciclos_completados
    db.flush()

    return _turno_to_out(t, db)


@router.delete("/{turno_id}", status_code=204)
def eliminar_turno(turno_id: int, db: Session = Depends(get_session_dep)):
    """
    Elimina un turno definitivamente.

    Restricciones:
      - No se puede eliminar un turno activo.
      - No se puede eliminar un turno con ciclos asociados.
    """
    t = db.get(Turno, turno_id)
    if t is None:
        raise HTTPException(404, "Turno no encontrado")
    if t.activo:
        raise HTTPException(
            409,
            "No se puede eliminar el turno activo. Ciérralo primero con PATCH /turnos/{id}/cerrar.",
        )

    # Verificar ciclos asociados
    n_ciclos = (
        db.query(func.count(Ciclo.id))
        .filter(Ciclo.turno_id == turno_id)
        .scalar() or 0
    )
    if n_ciclos > 0:
        raise HTTPException(
            409,
            f"El turno tiene {n_ciclos} ciclos asociados. "
            f"No se puede eliminar — los datos históricos deben conservarse.",
        )

    db.delete(t)
    db.flush()
