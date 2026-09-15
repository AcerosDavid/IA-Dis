"""
Aplicacion FastAPI principal.
Inicializa la app, configura CORS y registra todos los routers.

Ejecutar con:
    uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import eventos

app = FastAPI(
    title="Mining Dispatch API",
    description="API REST para el sistema de despacho inteligente de maquinaria minera.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Permite requests desde cualquier origen en desarrollo.
# En produccion reemplaza allow_origins=["*"] con los dominios especificos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(eventos.router, tags=["Eventos"])

# Routers futuros (descomentar cuando esten implementados):
# from src.api.routers import ciclos, zonas, turnos, reportes
# app.include_router(ciclos.router,   tags=["Ciclos"])
# app.include_router(zonas.router,    tags=["Zonas"])
# app.include_router(turnos.router,   tags=["Turnos"])
# app.include_router(reportes.router, tags=["Reportes"])
