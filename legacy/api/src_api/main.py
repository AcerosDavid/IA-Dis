"""
Aplicacion FastAPI principal.
Inicializa la app, configura CORS y registra todos los routers.

Ejecutar con:
    uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import eventos, ciclos, zonas, estadisticas, video
from src.api.routers import videos, events, reports
from src.api.routers import models, turnos
from src.api.routers import auth, video_tools
from src.config import load_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Crea las tablas al arrancar la API, recupera jobs interrumpidos y libera recursos al cerrar."""
    from src.database.db import migrate, get_session
    from src.api.routers.video import recuperar_jobs_interrumpidos
    from src.core.security import crear_admin_si_no_existe

    migrate()
    recuperar_jobs_interrumpidos()

    # Crear usuario admin por defecto si no existe ninguno
    with get_session() as db:
        crear_admin_si_no_existe(db)

    yield


app = FastAPI(
    title="Mining Dispatch API",
    description="API REST para el sistema de despacho inteligente de maquinaria minera.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Con credenciales (JWT en cookies/headers) NO se puede usar ["*"]: el navegador
# rechaza la combinación. CORS_ORIGINS admite lista separada por comas.
_cors_raw      = load_runtime().cors_origins.strip()
_cors_origins  = [o.strip() for o in _cors_raw.split(",") if o.strip()] or ["*"]
_allow_creds   = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ── Routers ───────────────────────────────────────────────────────────────────
# Routers heredados
app.include_router(eventos.router)
app.include_router(ciclos.router)
app.include_router(zonas.router)
app.include_router(estadisticas.router)
app.include_router(video.router)
# Herramientas de video (probe/thumbnail/frame/clip/download) — antes del
# router genérico /videos para que /videos/storage/usage no choque con {id}.
app.include_router(video_tools.router)
# Routers nuevos (Pasos 3-5)
app.include_router(videos.router)
app.include_router(events.router)
app.include_router(reports.router)
# Routers nuevos (Paso 7)
app.include_router(models.router)
app.include_router(turnos.router)
# Router de autenticación (Paso 8)
app.include_router(auth.router)


@app.get("/")
def root():
    return {
        "sistema": "Mining Dispatch API",
        "version": "0.2.0",
        "docs":    "/docs",
        "endpoints": {
            # Heredados
            "eventos":        "/eventos",
            "ciclos":         "/ciclos",
            "zonas":          "/zonas",
            "estadisticas":   "/estadisticas/resumen",
            "productividad":  "/estadisticas/productividad",
            "video_analizar": "/video/analizar",
            "video_jobs":     "/video/jobs",
            "health":         "/eventos/health",
            # Nuevos
            "videos":         "/videos",
            "events":         "/events",
            "reports":        "/reports",
            "analysis_jobs":  "/video/jobs",
            "models":         "/models",
            "turnos":         "/turnos",
            "turno_activo":   "/turnos/activo",
            # Auth (Paso 8)
            "login":          "/auth/login",
            "perfil":         "/auth/me",
            "usuarios":       "/auth/users",
        }
    }
