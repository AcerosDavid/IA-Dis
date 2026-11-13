"""
Router de Modelos — catálogo de modelos de IA disponibles en el sistema.

El catálogo se construye dinámicamente leyendo los archivos .pt del disco,
combinando con metadatos estáticos conocidos (arquitectura, capacidades, etc.).
No requiere tabla en DB — los modelos son archivos, no registros.

Endpoints:
  GET /models           — lista todos los modelos disponibles
  GET /models/{model_id} — detalle de un modelo específico
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/models", tags=["Modelos"])

# ── Directorios donde buscar modelos ─────────────────────────────────────────
MODELOS_DIRS = [Path("models"), Path("weights")]

# ── Metadatos estáticos por nombre de archivo ─────────────────────────────────
# Lo que sabemos de cada modelo más allá de lo que está en el archivo .pt.
# Los modelos no listados aquí se catalogan como "desconocido" pero siguen
# apareciendo si están en disco.

_METADATA: dict[str, dict] = {
    # ── YOLO (detección de maquinaria) ─────────────────────────────────────────
    "best_2.pt": {
        "name":         "YOLO Mining v2",
        "type":         "detection",
        "architecture": "YOLOv11n",
        "description":  "Modelo de detección de maquinaria minera. "
                        "Detecta camiones, excavadoras y palas mecánicas.",
        "capabilities": ["detect_machinery", "track_ids", "count_vehicles"],
        "classes":      ["camion", "excavadora", "pala"],
        "input_size":   320,
        "status":       "active",
    },
    "best.pt": {
        "name":         "YOLO Mining v1",
        "type":         "detection",
        "architecture": "YOLOv11n",
        "description":  "Primera versión del modelo de detección (legacy).",
        "capabilities": ["detect_machinery", "track_ids"],
        "classes":      ["camion", "excavadora", "pala"],
        "input_size":   320,
        "status":       "deprecated",
    },
    "best.v1.pt": {
        "name":         "YOLO Mining v1.1",
        "type":         "detection",
        "architecture": "YOLOv11s",
        "description":  "Versión intermedia con mayor capacidad (9.4M params).",
        "capabilities": ["detect_machinery", "track_ids"],
        "classes":      ["camion", "excavadora", "pala"],
        "input_size":   640,
        "status":       "archived",
    },
    "yolo26n.pt": {
        "name":         "YOLO26 Nano",
        "type":         "detection",
        "architecture": "YOLOv11n",
        "description":  "Modelo base nano para uso general (no especializado en minería).",
        "capabilities": ["detect_objects"],
        "classes":      ["dump_truck", "excavator", "wheel_loader"],
        "input_size":   320,
        "status":       "base",
    },
    # ── LSTM (clasificación de estados) ────────────────────────────────────────
    "lstm_camion.pt": {
        "name":         "LSTM Estado Camión",
        "type":         "classification_temporal",
        "architecture": "LSTM Bidireccional (2 capas, hidden=256)",
        "description":  "Clasifica el estado operacional de camiones a partir "
                        "de secuencias temporales de 16 frames.",
        "capabilities": ["classify_state"],
        "classes":      ["vacio", "cargado"],
        "input_size":   576,    # dim del vector de características (feature_dim)
        "status":       "active",
    },
    "lstm_excavadora.pt": {
        "name":         "LSTM Estado Excavadora",
        "type":         "classification_temporal",
        "architecture": "LSTM Bidireccional (2 capas, hidden=256)",
        "description":  "Clasifica el estado operacional de excavadoras.",
        "capabilities": ["classify_state"],
        "classes":      ["escarbando_cargando", "descargando"],
        "input_size":   576,
        "status":       "active",
    },
    "lstm_pala.pt": {
        "name":         "LSTM Estado Pala",
        "type":         "classification_temporal",
        "architecture": "LSTM Bidireccional (2 capas, hidden=256)",
        "description":  "Clasifica el estado operacional de palas mecánicas.",
        "capabilities": ["classify_state"],
        "classes":      ["escarbando_cargando", "descargando"],
        "input_size":   576,
        "status":       "active",
    },
}


# ── Schemas ───────────────────────────────────────────────────────────────────

class ModeloOut(BaseModel):
    model_id:      str          # nombre del archivo sin extensión → ID único
    filename:      str          # nombre completo del archivo
    name:          str          # nombre legible
    type:          str          # "detection" | "classification_temporal"
    architecture:  str
    description:   str
    capabilities:  list[str]    # qué tareas puede hacer
    classes:       list[str]    # clases que detecta/clasifica
    input_size:    int          # resolución de entrada (px) o dim del vector
    status:        str          # "active" | "deprecated" | "archived" | "base"
    disponible:    bool         # True si el archivo existe en disco
    size_mb:       float        # tamaño del archivo en MB
    path:          str          # ruta absoluta en disco


# ── Lógica de catálogo ────────────────────────────────────────────────────────

def _escanear_modelos() -> list[ModeloOut]:
    """
    Escanea los directorios de modelos y construye el catálogo.
    Combina metadatos estáticos con información del archivo en disco.
    """
    encontrados: dict[str, ModeloOut] = {}

    for directorio in MODELOS_DIRS:
        if not directorio.exists():
            continue
        for archivo in sorted(directorio.glob("*.pt")):
            filename = archivo.name
            # Evitar duplicados (mismo archivo en models/ y weights/)
            if filename in encontrados:
                continue

            meta = _METADATA.get(filename, {})
            size_mb = round(archivo.stat().st_size / (1024 * 1024), 1)

            # Extraer versión del nombre del archivo para el model_id
            model_id = archivo.stem   # "best_2" desde "best_2.pt"

            encontrados[filename] = ModeloOut(
                model_id     = model_id,
                filename     = filename,
                name         = meta.get("name",         model_id),
                type         = meta.get("type",         "unknown"),
                architecture = meta.get("architecture", "desconocida"),
                description  = meta.get("description",  "Sin descripción disponible."),
                capabilities = meta.get("capabilities", []),
                classes      = meta.get("classes",      []),
                input_size   = meta.get("input_size",   0),
                status       = meta.get("status",       "unknown"),
                disponible   = True,
                size_mb      = size_mb,
                path         = str(archivo.resolve()),
            )

    # Añadir modelos definidos en metadatos pero no encontrados en disco
    for filename, meta in _METADATA.items():
        if filename not in encontrados:
            model_id = Path(filename).stem
            encontrados[filename] = ModeloOut(
                model_id     = model_id,
                filename     = filename,
                name         = meta.get("name",         model_id),
                type         = meta.get("type",         "unknown"),
                architecture = meta.get("architecture", "desconocida"),
                description  = meta.get("description",  ""),
                capabilities = meta.get("capabilities", []),
                classes      = meta.get("classes",      []),
                input_size   = meta.get("input_size",   0),
                status       = "unavailable",
                disponible   = False,
                size_mb      = 0.0,
                path         = "",
            )

    # Ordenar: activos primero, luego por nombre
    orden = {"active": 0, "base": 1, "archived": 2, "deprecated": 3,
             "unavailable": 4, "unknown": 5}
    return sorted(
        encontrados.values(),
        key=lambda m: (orden.get(m.status, 99), m.name),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ModeloOut])
def listar_modelos(
    type:   str | None = None,
    status: str | None = None,
):
    """
    Lista todos los modelos disponibles con sus capacidades.

    Filtra por tipo:
      - detection            → modelos YOLO de detección de objetos
      - classification_temporal → modelos LSTM de clasificación de estados

    Filtra por estado:
      - active       → en uso activo
      - deprecated   → reemplazado por versión nueva
      - archived     → guardado para referencia
      - unavailable  → definido pero sin archivo en disco
    """
    modelos = _escanear_modelos()

    if type:
        modelos = [m for m in modelos if m.type == type]
    if status:
        modelos = [m for m in modelos if m.status == status]

    return modelos


@router.get("/{model_id}", response_model=ModeloOut)
def obtener_modelo(model_id: str):
    """
    Retorna el detalle de un modelo específico.
    El model_id es el nombre del archivo sin extensión (ej: best_2, lstm_camion).
    """
    modelos = _escanear_modelos()
    for m in modelos:
        if m.model_id == model_id:
            return m
    raise HTTPException(404, f"Modelo '{model_id}' no encontrado")
