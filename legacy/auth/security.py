"""
security.py — Módulo central de autenticación y autorización JWT.

Responsabilidades:
  - Hash y verificación de contraseñas con bcrypt.
  - Creación y validación de tokens JWT (access + refresh).
  - Dependencias de FastAPI para proteger endpoints:
      get_usuario_actual()    → verifica token y retorna usuario
      requerir_rol(*roles)    → verifica que el usuario tiene uno de los roles
  - Creación del usuario administrador por defecto al arrancar.

Configuración en .env:
    SECRET_KEY    = clave secreta para firmar JWT (mínimo 32 caracteres)
        ACCESS_TOKEN_EXPIRE_MINUTES  = 60  (default)
    REFRESH_TOKEN_EXPIRE_DAYS    = 7   (default)

Roles y permisos:
    admin      → puede todo
    supervisor → leer todo, revisar eventos, generar reportes
    analista   → cargar videos, ejecutar análisis, revisar sus propios eventos
    consulta   → solo GET sobre recursos autorizados

Uso en endpoints:
    # Solo usuarios autenticados
    @router.get("/ruta")
    def mi_endpoint(usuario = Depends(get_usuario_actual)):
        ...

    # Solo admin o supervisor
    @router.post("/ruta")
    def mi_endpoint(usuario = Depends(requerir_rol("admin", "supervisor"))):
        ...
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import bcrypt as _bcrypt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config import load_runtime
from src.database.db import get_session_dep
from src.database.models import Usuario

# ── Configuración ──────────────────────────────────────────────────────────────

# load_runtime() lee el .env con pydantic-settings. Antes se usaba os.getenv(),
# que NO ve el .env salvo que las variables estén exportadas en el proceso.
_rt = load_runtime()

# SECRET_KEY desde .env; si no está, genera una aleatoria en memoria.
# En producción SIEMPRE definir en .env para que los tokens sobrevivan reinicios.
if _rt.secret_key.strip():
    SECRET_KEY = _rt.secret_key.strip()
else:
    SECRET_KEY = secrets.token_hex(32)
    print(
        "[SECURITY] ADVERTENCIA: SECRET_KEY no definida en .env. "
        "Los tokens JWT se invalidarán en cada reinicio. "
        "Define SECRET_KEY en tu .env para producción."
    )

ALGORITHM              = "HS256"
ACCESS_EXPIRE_MINUTES  = _rt.access_token_expire_minutes
REFRESH_EXPIRE_DAYS    = _rt.refresh_token_expire_days

# Jerarquía de roles (mayor índice = mayor privilegio)
_JERARQUIA: dict[str, int] = {
    "consulta":  0,
    "analista":  1,
    "supervisor": 2,
    "admin":     3,
}

# ── Herramientas de hash ───────────────────────────────────────────────────────
# Usamos bcrypt directamente (passlib 1.7.4 es incompatible con bcrypt>=4.0)

def hashear_password(password: str) -> str:
    """Retorna el hash bcrypt de la contraseña (salt incluido)."""
    pwd_bytes = password.encode("utf-8")
    return _bcrypt.hashpw(pwd_bytes, _bcrypt.gensalt(rounds=12)).decode("utf-8")


def verificar_password(password_plano: str, hash_guardado: str) -> bool:
    """Verifica si una contraseña en texto plano coincide con el hash bcrypt."""
    try:
        return _bcrypt.checkpw(
            password_plano.encode("utf-8"),
            hash_guardado.encode("utf-8"),
        )
    except Exception:
        return False


# ── Tokens JWT ─────────────────────────────────────────────────────────────────

class TokenData(BaseModel):
    username: str
    rol:      str
    tipo:     str   # "access" | "refresh"


def crear_access_token(username: str, rol: str) -> str:
    """Crea un JWT de acceso con vida limitada."""
    exp = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": username, "rol": rol, "tipo": "access", "exp": exp},
        SECRET_KEY, algorithm=ALGORITHM,
    )


def crear_refresh_token(username: str, rol: str) -> str:
    """Crea un JWT de refresh con vida larga (para renovar el access token)."""
    exp = datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": username, "rol": rol, "tipo": "refresh", "exp": exp},
        SECRET_KEY, algorithm=ALGORITHM,
    )


def decodificar_token(token: str, tipo_esperado: str = "access") -> TokenData:
    """
    Decodifica y valida un JWT.
    Lanza HTTPException 401 si el token es inválido, expirado o del tipo incorrecto.
    """
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        rol      = payload.get("rol", "consulta")
        tipo     = payload.get("tipo", "access")
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    if not username:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token sin usuario")
    if tipo != tipo_esperado:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            f"Tipo de token incorrecto: se esperaba '{tipo_esperado}', llegó '{tipo}'",
        )
    return TokenData(username=username, rol=rol, tipo=tipo)


# ── Dependencias FastAPI ──────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_usuario_actual(
    token: str    = Depends(oauth2_scheme),
    db:   Session = Depends(get_session_dep),
) -> Usuario:
    """
    Dependencia que extrae y valida el usuario del token JWT.
    Usar en cualquier endpoint que requiera autenticación.

    Uso:
        @router.get("/ruta")
        def endpoint(usuario: Usuario = Depends(get_usuario_actual)):
            return {"saludo": f"Hola, {usuario.username}"}
    """
    token_data = decodificar_token(token, tipo_esperado="access")

    usuario = (
        db.query(Usuario)
        .filter(Usuario.username == token_data.username)
        .first()
    )
    if usuario is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Usuario no encontrado")
    if not usuario.activo:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Cuenta desactivada")
    return usuario


def requerir_rol(*roles: str):
    """
    Dependencia de fábrica que verifica que el usuario tiene uno de los roles dados.

    Usa jerarquía: si se requiere "supervisor", un "admin" también puede pasar.

    Uso:
        @router.delete("/ruta")
        def eliminar(usuario = Depends(requerir_rol("admin"))):
            ...

        @router.post("/ruta")
        def crear(usuario = Depends(requerir_rol("analista", "supervisor", "admin"))):
            ...
    """
    nivel_minimo = min(_JERARQUIA.get(r, 99) for r in roles)

    def _verificar(usuario: Usuario = Depends(get_usuario_actual)) -> Usuario:
        nivel_usuario = _JERARQUIA.get(usuario.rol, -1)
        if nivel_usuario < nivel_minimo:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Permiso insuficiente. Roles requeridos: {list(roles)}. "
                f"Tu rol: {usuario.rol}",
            )
        return usuario

    return _verificar


def get_usuario_opcional(
    token: str | None = Depends(OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)),
    db:   Session     = Depends(get_session_dep),
) -> Usuario | None:
    """
    Como get_usuario_actual pero retorna None si no hay token.
    Útil para endpoints que son públicos pero muestran más datos si estás autenticado.
    """
    if token is None:
        return None
    try:
        return get_usuario_actual(token, db)
    except HTTPException:
        return None


# ── Bootstrap: usuario admin por defecto ─────────────────────────────────────

ADMIN_DEFAULT_USERNAME = "admin"
ADMIN_DEFAULT_PASSWORD = _rt.admin_password


def crear_admin_si_no_existe(db: Session) -> None:
    """
    Crea el usuario admin por defecto al arrancar el sistema.
    Si ya existe uno, no hace nada.

    La contraseña viene de ADMIN_PASSWORD en .env.
    Cámbiala en producción o desde el endpoint PATCH /auth/users/{id}.
    """
    existe = db.query(Usuario).filter(Usuario.rol == "admin").first()
    if existe:
        return

    admin = Usuario(
        username        = ADMIN_DEFAULT_USERNAME,
        email           = "admin@mining-dispatch.local",
        hashed_password = hashear_password(ADMIN_DEFAULT_PASSWORD),
        nombre_completo = "Administrador del Sistema",
        rol             = "admin",
        activo          = True,
    )
    db.add(admin)
    db.flush()
    print(
        f"[SECURITY] Usuario admin creado: username='{ADMIN_DEFAULT_USERNAME}'. "
        "Cambia ADMIN_PASSWORD en .env y la contraseña tras el primer login."
    )
