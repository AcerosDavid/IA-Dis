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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Crea las tablas al arrancar la API y libera recursos al cerrar."""
    from src.database.db import migrate
    migrate()
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(eventos.router)
app.include_router(ciclos.router)
app.include_router(zonas.router)
app.include_router(estadisticas.router)
app.include_router(video.router)


@app.get("/")
def root():
    return {
        "sistema": "Mining Dispatch API",
        "version": "0.2.0",
        "docs":    "/docs",
        "endpoints": {
            "eventos":       "/eventos",
            "ciclos":        "/ciclos",
            "zonas":         "/zonas",
            "estadisticas":  "/estadisticas/resumen",
            "productividad": "/estadisticas/productividad",
            "health":        "/eventos/health",
        }
    }
