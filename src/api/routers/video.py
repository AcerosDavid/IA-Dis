"""
Router de análisis de video.

Endpoints:
  POST /video/analizar   — sube un video y lanza el pipeline YOLO+ByteTrack en background
  GET  /video/estado/{job_id} — consulta el progreso del análisis
  GET  /video/jobs            — lista todos los jobs activos/completados
  DELETE /video/jobs/{job_id} — cancela o elimina un job
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Query
from pydantic import BaseModel

router = APIRouter(prefix="/video", tags=["Video"])


# ── Estado de un job ──────────────────────────────────────────────────────────

class JobEstado(str, Enum):
    PENDIENTE  = "pendiente"
    PROCESANDO = "procesando"
    COMPLETADO = "completado"
    ERROR      = "error"
    CANCELADO  = "cancelado"


@dataclass
class JobInfo:
    job_id:          str
    nombre_archivo:  str
    estado:          JobEstado       = JobEstado.PENDIENTE
    progreso:        float           = 0.0          # 0–100
    frame_actual:    int             = 0
    total_frames:    int             = 0
    fps_video:       float           = 0.0
    eventos_nuevos:  int             = 0
    ciclos_nuevos:   int             = 0
    maquinas_activas: int            = 0
    detecciones_frame: list[dict]    = field(default_factory=list)   # último frame procesado
    error_msg:       str | None      = None
    ts_inicio:       float           = field(default_factory=time.time)
    ts_fin:          float | None    = None
    cancelar:        bool            = False       # flag para cancelación cooperativa
    _tmp_path:       str | None      = None        # archivo temporal en disco


# Almacén en memoria — dict[job_id, JobInfo]
_JOBS: dict[str, JobInfo] = {}
_JOBS_LOCK = threading.Lock()

MAX_JOBS_ACTIVOS = 3          # evitar saturar la CPU
MAX_FILE_MB      = 500        # límite de tamaño de upload


# ── Schemas de respuesta ──────────────────────────────────────────────────────

class DeteccionOut(BaseModel):
    track_id:    int
    maquina_id:  str
    tipo_maquina: str
    confianza:   float
    bbox:        list[float]   # [x1, y1, x2, y2]
    zona_actual: str | None = None


class JobEstadoOut(BaseModel):
    job_id:           str
    nombre_archivo:   str
    estado:           JobEstado
    progreso:         float
    frame_actual:     int
    total_frames:     int
    fps_video:        float
    eventos_nuevos:   int
    ciclos_nuevos:    int
    maquinas_activas: int
    detecciones_frame: list[DeteccionOut]
    error_msg:        str | None
    duracion_seg:     float | None


# ── Pipeline de análisis ──────────────────────────────────────────────────────

def _run_pipeline(job_id: str) -> None:
    """
    Ejecuta el pipeline completo en un hilo background.
    Reutiliza exactamente la misma lógica de main.py:
      Detector → GestorPresencia → GestorGeofencing → RepositorioCiclos
    """
    from src.config import load_config
    from src.database.db import (
        RepositorioEventos, RepositorioCiclos,
        RepositorioZonas, get_session, migrate,
    )
    from src.ai.detector import Detector
    from src.core.geometry import BBox, GestorGeofencing, Zona

    # Importar GestorPresencia desde main (ya está ahí)
    import importlib, sys
    # Importar dinámicamente para no circular
    main_mod = importlib.import_module("main")
    GestorPresencia = main_mod.GestorPresencia
    calcular_timestamp = main_mod.calcular_timestamp

    ZONAS_DEFAULT = [
        Zona.rectangulo("zona_carga",    "carga",    50,  50, 300, 200, color_bgr=(0, 255, 0)),
        Zona.rectangulo("zona_descarga", "descarga", 500, 50, 800, 200, color_bgr=(0, 0, 255)),
    ]
    TRACK_IMGSZ  = 416   # 416 es óptimo para 920MX (2GB VRAM): buen balance precisión/velocidad
    TRACK_SKIP   = 1     # Con GPU procesar cada frame es viable
    FRAMES_AUS   = 90

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return

    job.estado = JobEstado.PROCESANDO
    tmp_path = job._tmp_path

    try:
        config = load_config()
        migrate()

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV no pudo abrir el archivo de video.")

        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        job.total_frames = max(total, 1)
        job.fps_video    = fps

        with get_session() as session:
            repo        = RepositorioEventos(session)
            repo_ciclos = RepositorioCiclos(session)
            gestor      = GestorPresencia(frames_ausencia=FRAMES_AUS)
            gestor_geo  = GestorGeofencing()

            # Cargar zonas desde DB; si no hay, usar defaults
            repo_zonas = RepositorioZonas(session)
            zonas_db   = repo_zonas.cargar_activas()
            if zonas_db:
                for z in zonas_db:
                    gestor_geo.agregar_zona(z)
            else:
                for z in ZONAS_DEFAULT:
                    gestor_geo.agregar_zona(z)
                    repo_zonas.guardar(z)

            detector = Detector(config.model_path, config.confidence_threshold)
            detector.load_model()

            frame_count = 0
            ev_count    = 0
            ciclo_count = 0

            while True:
                # Cancelación cooperativa
                if job.cancelar:
                    job.estado = JobEstado.CANCELADO
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = calcular_timestamp(fps, frame_count)

                if frame_count % TRACK_SKIP == 0:
                    try:
                        detecciones = detector.track_frame(
                            frame, frame_count, timestamp, imgsz=TRACK_IMGSZ
                        )
                        mensajes = gestor.procesar(
                            detecciones, frame_count,
                            repo, gestor_geo, repo_ciclos,
                        )
                        for msg in mensajes:
                            if "[APARICION]" in msg or "[DESAPARICION]" in msg:
                                ev_count += 1
                            elif "[ZONA]" in msg:
                                ciclo_count += 1

                        # Construir detecciones del frame para el frontend
                        dets_out = []
                        for d in gestor.activos_para_dibujar():
                            estado_geo = gestor_geo._estados.get(d["maquina_id"])
                            zona_act   = estado_geo.zona_actual if estado_geo else None
                            dets_out.append({
                                "track_id":    d["track_id"],
                                "maquina_id":  d["maquina_id"],
                                "tipo_maquina": d["tipo_maquina"],
                                "confianza":   round(d["confianza"], 3),
                                "bbox":        [round(v, 1) for v in d["bbox"]],
                                "zona_actual": zona_act,
                            })

                        # Actualizar job (sin lock en lectura de listas pequeñas)
                        job.frame_actual       = frame_count
                        job.progreso           = round((frame_count / job.total_frames) * 100, 1)
                        job.eventos_nuevos     = ev_count
                        job.ciclos_nuevos      = ciclo_count
                        job.maquinas_activas   = gestor.n_activos
                        job.detecciones_frame  = dets_out

                    except Exception as e:
                        # Error en un frame no es fatal — seguir procesando
                        print(f"[VIDEO ROUTER] frame={frame_count} error: {e}")

                frame_count += 1

            # Cerrar presencia (registrar desapariciones al final)
            gestor.cerrar(repo, gestor_geo)

        cap.release()

        if job.estado != JobEstado.CANCELADO:
            job.estado    = JobEstado.COMPLETADO
            job.progreso  = 100.0
            job.ts_fin    = time.time()

    except Exception as e:
        job.estado    = JobEstado.ERROR
        job.error_msg = str(e)
        job.ts_fin    = time.time()
        print(f"[VIDEO ROUTER] job={job_id} falló: {e}")
    finally:
        # Limpiar archivo temporal
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analizar", response_model=JobEstadoOut, status_code=202)
async def analizar_video(
    file: UploadFile = File(..., description="Archivo de video (.mp4, .avi, .mov, .mkv)"),
):
    """
    Sube un video y lanza el pipeline de detección YOLO+ByteTrack en background.

    Devuelve inmediatamente con job_id=202 Accepted.
    Sondea GET /video/estado/{job_id} para seguir el progreso.
    """
    # Validar tipo de archivo
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Formato no soportado: {ext}. Usa: {', '.join(allowed_exts)}",
        )

    # Límite de tamaño
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo demasiado grande ({size_mb:.1f} MB). Máximo: {MAX_FILE_MB} MB",
        )

    # Verificar jobs activos
    with _JOBS_LOCK:
        activos = sum(
            1 for j in _JOBS.values()
            if j.estado in (JobEstado.PENDIENTE, JobEstado.PROCESANDO)
        )
        if activos >= MAX_JOBS_ACTIVOS:
            raise HTTPException(
                status_code=429,
                detail=f"Hay {activos} análisis en curso. Espera a que terminen.",
            )

    # Guardar en archivo temporal
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(content)

    # Crear job
    job_id = str(uuid.uuid4())
    job    = JobInfo(
        job_id         = job_id,
        nombre_archivo = file.filename or "video",
        _tmp_path      = tmp_path,
    )

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    # Lanzar pipeline en hilo separado
    t = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    t.start()

    return _job_to_out(job)


@router.get("/estado/{job_id}", response_model=JobEstadoOut)
def estado_job(job_id: str):
    """Retorna el estado actual del análisis (sondear cada 1-2 s)."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return _job_to_out(job)


@router.get("/jobs", response_model=list[JobEstadoOut])
def listar_jobs(
    estado: JobEstado | None = Query(None, description="Filtrar por estado"),
):
    """Lista todos los jobs de análisis."""
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    if estado:
        jobs = [j for j in jobs if j.estado == estado]
    return [_job_to_out(j) for j in jobs]


@router.delete("/jobs/{job_id}", status_code=204)
def cancelar_job(job_id: str):
    """Cancela un job en progreso o elimina uno completado/fallido."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    if job.estado in (JobEstado.PENDIENTE, JobEstado.PROCESANDO):
        job.cancelar = True   # el hilo lo lee y para
    else:
        with _JOBS_LOCK:
            _JOBS.pop(job_id, None)


# ── Helper ────────────────────────────────────────────────────────────────────

def _job_to_out(job: JobInfo) -> JobEstadoOut:
    dur = None
    if job.ts_fin:
        dur = round(job.ts_fin - job.ts_inicio, 1)
    elif job.estado == JobEstado.PROCESANDO:
        dur = round(time.time() - job.ts_inicio, 1)

    return JobEstadoOut(
        job_id           = job.job_id,
        nombre_archivo   = job.nombre_archivo,
        estado           = job.estado,
        progreso         = job.progreso,
        frame_actual     = job.frame_actual,
        total_frames     = job.total_frames,
        fps_video        = job.fps_video,
        eventos_nuevos   = job.eventos_nuevos,
        ciclos_nuevos    = job.ciclos_nuevos,
        maquinas_activas = job.maquinas_activas,
        detecciones_frame= [DeteccionOut(**d) for d in job.detecciones_frame],
        error_msg        = job.error_msg,
        duracion_seg     = dur,
    )
