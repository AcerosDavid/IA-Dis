"""
Router de autenticación — login, refresh, gestión de usuarios.

Endpoints públicos:
  POST /auth/login          — obtener access + refresh token
  POST /auth/refresh         — renovar access token con refresh token

Endpoints autenticados:
  GET  /auth/me              — perfil del usuario actual
  POST /auth/logout          — invalidar sesión (solo informativo — JWT es stateless)

Endpoints solo admin:
  GET  /auth/users           — listar usuarios
  POST /auth/users           — crear usuario
  GET  /auth/users/{id}      — detalle de usuario
  PATCH /auth/users/{id}     — actualizar (rol, activo, contraseña)
  DELETE /auth/users/{id}    — desactivar usuario (soft)
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.core.security import (
    hashear_password, verificar_password,
    crear_access_token, crear_refresh_token, decodificar_token,
    get_usuario_actual, requerir_rol,
)
from src.database.db import get_session_dep, RepositorioAuditoria
from src.database.models import Usuario

router = APIRouter(prefix="/auth", tags=["Autenticación"])

# ── Schemas ───────────────────────────────────────────────────────────────────

class TokenOut(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int   # segundos hasta expiración del access token


class RefreshIn(BaseModel):
    refresh_token: str


class UsuarioOut(BaseModel):
    id:              int
    username:        str
    email:           str | None = None
    nombre_completo: str | None = None
    rol:             str
    activo:          bool
    created_at:      str
    last_login:      str | None = None


class UsuarioCreate(BaseModel):
    username:        str = Field(min_length=3, max_length=50)
    password:        str = Field(min_length=6)
    email:           str | None = None
    nombre_completo: str | None = None
    rol:             str = Field(
        default="consulta",
        pattern="^(admin|supervisor|analista|consulta)$",
    )
    project_ids: list[str] | None = None


class UsuarioUpdate(BaseModel):
    email:           str | None = None
    nombre_completo: str | None = None
    rol:             str | None = Field(
        default=None,
        pattern="^(admin|supervisor|analista|consulta)$",
    )
    activo:          bool | None = None
    password:        str | None = Field(default=None, min_length=6)
    project_ids:     list[str] | None = None


# ── Helper ────────────────────────────────────────────────────────────────────

def _usuario_to_out(u: Usuario) -> UsuarioOut:
    return UsuarioOut(
        id              = u.id,
        username        = u.username,
        email           = u.email,
        nombre_completo = u.nombre_completo,
        rol             = u.rol,
        activo          = u.activo,
        created_at      = u.created_at.isoformat() if u.created_at else "",
        last_login      = u.last_login.isoformat() if u.last_login else None,
    )


# ── Endpoints públicos ────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenOut)
def login(
    form:  OAuth2PasswordRequestForm = Depends(),
    db:    Session = Depends(get_session_dep),
):
    """
    Autentica al usuario y retorna access token + refresh token.

    El access token dura ACCESS_TOKEN_EXPIRE_MINUTES (default: 60 min).
    El refresh token dura REFRESH_TOKEN_EXPIRE_DAYS (default: 7 días).

    El frontend debe guardar ambos tokens de forma segura
    (httpOnly cookie o memoria — nunca localStorage).
    """
    usuario = (
        db.query(Usuario)
        .filter(Usuario.username == form.username)
        .first()
    )

    if usuario is None or not verificar_password(form.password, usuario.hashed_password):
        # Auditoría de intento fallido
        repo_audit = RepositorioAuditoria(db)
        repo_audit.registrar(
            action       = "login_failed",
            actor        = form.username,
            result       = "error",
            context_json = {"motivo": "credenciales incorrectas"},
        )
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not usuario.activo:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Cuenta desactivada")

    # Actualizar last_login
    usuario.last_login = datetime.now(timezone.utc)
    db.flush()

    # Auditoría de login exitoso
    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action        = "login",
        actor         = usuario.username,
        resource_type = "usuario",
        resource_id   = str(usuario.id),
        result        = "success",
    )

    from src.core.security import ACCESS_EXPIRE_MINUTES
    return TokenOut(
        access_token  = crear_access_token(usuario.username, usuario.rol),
        refresh_token = crear_refresh_token(usuario.username, usuario.rol),
        token_type    = "bearer",
        expires_in    = ACCESS_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenOut)
def refresh_token(
    datos: RefreshIn,
    db:    Session = Depends(get_session_dep),
):
    """
    Renueva el access token usando el refresh token.
    No requiere usuario/contraseña — solo el refresh token vigente.
    """
    token_data = decodificar_token(datos.refresh_token, tipo_esperado="refresh")

    # Verificar que el usuario sigue activo
    usuario = (
        db.query(Usuario)
        .filter(Usuario.username == token_data.username)
        .first()
    )
    if usuario is None or not usuario.activo:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Usuario no disponible")

    from src.core.security import ACCESS_EXPIRE_MINUTES
    return TokenOut(
        access_token  = crear_access_token(usuario.username, usuario.rol),
        refresh_token = crear_refresh_token(usuario.username, usuario.rol),
        token_type    = "bearer",
        expires_in    = ACCESS_EXPIRE_MINUTES * 60,
    )


# ── Endpoints autenticados ────────────────────────────────────────────────────

@router.get("/me", response_model=UsuarioOut)
def perfil_actual(usuario: Usuario = Depends(get_usuario_actual)):
    """Retorna el perfil del usuario autenticado."""
    return _usuario_to_out(usuario)


@router.post("/logout", status_code=204)
def logout(
    usuario: Usuario = Depends(get_usuario_actual),
    db:      Session = Depends(get_session_dep),
):
    """
    Registra el logout en auditoría.
    JWT es stateless — la invalidación real requiere una blacklist
    (pendiente para el Paso 9 si se implementa).
    """
    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action        = "logout",
        actor         = usuario.username,
        resource_type = "usuario",
        resource_id   = str(usuario.id),
        result        = "success",
    )


# ── Gestión de usuarios (solo admin) ─────────────────────────────────────────

@router.get("/users", response_model=list[UsuarioOut])
def listar_usuarios(
    activo:   bool | None = None,
    rol:      str | None  = None,
    db:       Session     = Depends(get_session_dep),
    _usuario: Usuario     = Depends(requerir_rol("admin")),
):
    """Lista todos los usuarios del sistema. Solo admin."""
    q = db.query(Usuario)
    if activo is not None:
        q = q.filter(Usuario.activo == activo)
    if rol:
        q = q.filter(Usuario.rol == rol)
    return [_usuario_to_out(u) for u in q.order_by(Usuario.id).all()]


@router.post("/users", response_model=UsuarioOut, status_code=201)
def crear_usuario(
    datos:    UsuarioCreate,
    db:       Session = Depends(get_session_dep),
    _usuario: Usuario = Depends(requerir_rol("admin")),
):
    """Crea un nuevo usuario. Solo admin."""
    # Verificar unicidad de username
    if db.query(Usuario).filter(Usuario.username == datos.username).first():
        raise HTTPException(409, f"El username '{datos.username}' ya existe")

    if datos.email:
        if db.query(Usuario).filter(Usuario.email == datos.email).first():
            raise HTTPException(409, f"El email '{datos.email}' ya está registrado")

    nuevo = Usuario(
        username        = datos.username,
        email           = datos.email,
        hashed_password = hashear_password(datos.password),
        nombre_completo = datos.nombre_completo,
        rol             = datos.rol,
        project_ids     = datos.project_ids,
        activo          = True,
    )
    db.add(nuevo)
    db.flush()

    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action="create_user", actor=_usuario.username,
        resource_type="usuario", resource_id=str(nuevo.id),
        result="success", context_json={"rol": datos.rol},
    )

    return _usuario_to_out(nuevo)


@router.get("/users/{user_id}", response_model=UsuarioOut)
def obtener_usuario(
    user_id:  int,
    db:       Session = Depends(get_session_dep),
    _usuario: Usuario = Depends(requerir_rol("admin")),
):
    """Retorna el detalle de un usuario. Solo admin."""
    u = db.get(Usuario, user_id)
    if u is None:
        raise HTTPException(404, "Usuario no encontrado")
    return _usuario_to_out(u)


@router.patch("/users/{user_id}", response_model=UsuarioOut)
def actualizar_usuario(
    user_id:  int,
    datos:    UsuarioUpdate,
    db:       Session = Depends(get_session_dep),
    _usuario: Usuario = Depends(requerir_rol("admin")),
):
    """
    Actualiza campos de un usuario.
    Solo admin puede cambiar el rol o desactivar cuentas.
    """
    u = db.get(Usuario, user_id)
    if u is None:
        raise HTTPException(404, "Usuario no encontrado")

    # Proteger contra quitarle el rol admin al último admin
    if datos.rol and datos.rol != "admin" and u.rol == "admin":
        n_admins = db.query(Usuario).filter(
            Usuario.rol == "admin", Usuario.activo == True
        ).count()
        if n_admins <= 1:
            raise HTTPException(
                409,
                "No se puede cambiar el rol del único administrador activo. "
                "Crea otro administrador primero.",
            )

    if datos.email           is not None: u.email           = datos.email
    if datos.nombre_completo is not None: u.nombre_completo = datos.nombre_completo
    if datos.rol             is not None: u.rol             = datos.rol
    if datos.activo          is not None: u.activo          = datos.activo
    if datos.project_ids     is not None: u.project_ids     = datos.project_ids
    if datos.password:
        u.hashed_password = hashear_password(datos.password)

    db.flush()

    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action="update_user", actor=_usuario.username,
        resource_type="usuario", resource_id=str(user_id),
        result="success",
        context_json={
            k: v for k, v in datos.model_dump(exclude_none=True).items()
            if k != "password"   # nunca loguear contraseñas
        },
    )

    return _usuario_to_out(u)


@router.delete("/users/{user_id}", status_code=204)
def desactivar_usuario(
    user_id:  int,
    db:       Session = Depends(get_session_dep),
    _usuario: Usuario = Depends(requerir_rol("admin")),
):
    """
    Desactiva un usuario (soft delete).
    No puede desactivarse a sí mismo ni al último admin activo.
    """
    u = db.get(Usuario, user_id)
    if u is None:
        raise HTTPException(404, "Usuario no encontrado")
    if u.id == _usuario.id:
        raise HTTPException(409, "No puedes desactivar tu propia cuenta")

    if u.rol == "admin":
        n_admins = db.query(Usuario).filter(
            Usuario.rol == "admin", Usuario.activo == True
        ).count()
        if n_admins <= 1:
            raise HTTPException(
                409, "No puedes desactivar el único administrador activo."
            )

    u.activo = False
    db.flush()

    repo_audit = RepositorioAuditoria(db)
    repo_audit.registrar(
        action="deactivate_user", actor=_usuario.username,
        resource_type="usuario", resource_id=str(user_id), result="success",
    )
