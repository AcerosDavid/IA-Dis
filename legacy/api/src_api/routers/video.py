"""
Router de análisis de video — pipeline con jobs persistentes.

Paso 2: Los jobs ahora persisten en la tabla analysis_jobs de la DB.
  - Sobreviven a reinicios del servidor.
  - Al arrancar, los jobs "processing" se marcan como "failed" (recovery).
  - El estado y progreso se consultan desde la DB, no de memoria.
  - Las detecciones en tiempo real siguen en memoria (JobInfo runtime)
    pero el estado persistente está en la DB.

Endpoints:
  POST   /video/analizar           — sube video, crea registro + job y lanza análisis
  GET    /video/estado/{job_id}    — progreso + detecciones actuales (polling)
  GET    /video/estado/{job_id}/stream — progreso en tiempo real (SSE, sin polling)
  GET    /video/jobs              — lista jobs desde la DB con filtros
  DELETE /video/jobs/{job_id}     — cancela o elimina un job
  POST   /video/jobs/{job_id}/retry — reintenta un job fallido
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.api.schemas import AnalysisJobOut, VideoOut, PaginacionMeta
from src.config import Settings, load_config
from src.core.video_tools import probe_video
from src.database.db import get_session_dep
from src.database.models import Video, AnalysisJob
from src.pipeline.presencia import GestorPresencia, calcular_timestamp

router = APIRouter(prefix="/video", tags=["Video"])

# ── Caché de detectores y control de concurrencia ─────────────────────────────
# Cargar YOLO cuesta segundos; en CPU se reutiliza entre jobs. Con
# ANALYSIS_CONCURRENCY=1 el estado del tracker nunca se comparte entre videos.
_DETECTOR_CACHE: dict[str, "object"] = {}
_DETECTOR_LOCK  = threading.Lock()
_SEMAFORO: threading.BoundedSemaphore | None = None
_SEMAFORO_LOCK  = threading.Lock()


# ── Estados del job (strings, no enum — coinciden con la DB) ──────────────────

class JobEstado(str, Enum):
    QUEUED      = "queued"
    PROCESSING  = "processing"
    COMPLETED   = "completed"
    FAILED      = "failed"
    CANCELLED   = "cancelled"
    CANCEL_REQ  = "cancel_requested"


# ── Estado runtime en memoria (solo detecciones en vivo) ──────────────────────
# El estado persistente vive en la DB. Esto es solo para el streaming
# de detecciones en tiempo real mientras el job corre.

@dataclass
class JobRuntime:
    """Estado efímero del job mientras se ejecuta — no persiste."""
    job_db_id:         int
    progreso:          float        = 0.0
    frame_actual:      int          = 0
    total_frames:      int          = 0
    fps_video:         float        = 0.0
    eventos_nuevos:    int          = 0
    ciclos_nuevos:     int          = 0
    maquinas_activas:  int          = 0
    detecciones_frame: list[dict]   = field(default_factory=list)
    error_msg:         str | None   = None
    cancelar:          bool         = False
    _tmp_path:         str | None   = None
    _video_path:       str | None   = None   # ruta persistente si se guardó
    _thread:           threading.Thread | None = None
    ts_inicio:         float        = field(default_factory=time.time)
    ts_fin:            float | None = None


# Cache runtime en memoria — clave = job_db_id (int)
_RUNTIME: dict[int, JobRuntime] = {}
_RUNTIME_LOCK = threading.Lock()


# ── Schemas de respuesta ──────────────────────────────────────────────────────

class DeteccionOut(BaseModel):
    track_id:     int
    maquina_id:   str
    tipo_maquina: str
    confianza:    float
    bbox:         list[float]
    zona_actual:  str | None = None


class JobEstadoOut(BaseModel):
    """Respuesta unificada: combina estado persistente (DB) + runtime (memoria)."""
    job_id:            int           # ID de la DB (entero)
    video_id:          int
    nombre_archivo:    str
    estado:            str
    stage:             str | None = None
    progreso:          float
    frame_actual:      int
    total_frames:      int
    fps_video:         float
    eventos_nuevos:    int
    ciclos_nuevos:     int
    maquinas_activas:  int
    detecciones_frame: list[DeteccionOut]
    error_msg:         str | None
    duracion_seg:      float | None


class JobListOut(BaseModel):
    """Item de lista de jobs — sin detecciones en vivo."""
    job_id:        int
    video_id:      int
    nombre_archivo: str
    estado:        str
    stage:         str | None = None
    progreso:      float | None = None
    created_at:    str
    started_at:    str | None = None
    finished_at:   str | None = None
    error_message: str | None = None


class JobListPaginado(BaseModel):
    """Lista paginada de jobs."""
    items:      list[JobListOut]
    paginacion: PaginacionMeta


# ── Caché de detectores / concurrencia ───────────────────────────────────────

def _get_semaforo(config: Settings) -> threading.BoundedSemaphore:
    """Semáforo global que limita cuántos análisis corren a la vez."""
    global _SEMAFORO
    with _SEMAFORO_LOCK:
        if _SEMAFORO is None:
            _SEMAFORO = threading.BoundedSemaphore(max(1, config.analysis_concurrency))
        return _SEMAFORO


def _get_detector(config: Settings, fresh: bool = False):
    """
    Devuelve un Detector con el modelo cargado.

    Con concurrencia=1 se reutiliza el mismo detector entre jobs (evita
    recargar YOLO, que en CPU cuesta varios segundos). `reset()` limpia el
    estado del tracker antes de cada video.
    """
    from src.ai.detector import Detector

    if fresh or config.analysis_concurrency > 1:
        detector = Detector(config.model_path, config.confidence_threshold)
        detector.load_model()
        return detector

    key = f"{os.path.abspath(config.model_path)}|{config.confidence_threshold}"
    with _DETECTOR_LOCK:
        detector = _DETECTOR_CACHE.get(key)
        if detector is None:
            detector = Detector(config.model_path, config.confidence_threshold)
            detector.load_model()
            _DETECTOR_CACHE[key] = detector
        detector.reset()
        return detector


def _actualizar_job(job_db_id: int, **campos) -> None:
    """Actualiza un job en una sesión corta (segura entre hilos)."""
    from src.database.db import RepositorioJobs, get_session
    with get_session() as session:
        RepositorioJobs(session).actualizar(job_db_id, campos)


# ── Pipeline principal (secuencial, optimizado para CPU) ──────────────────────

def _procesar_video(job_db_id: int, rt: JobRuntime, video_path: str, config: Settings) -> None:
    """
    Decodifica el video e infiere de forma secuencial.

    Por qué secuencial:
      - En CPU la inferencia es el cuello de botella; un hilo lector sólo
        agrega latencia y descarta frames.
      - `cap.grab()` avanza sin decodificar a color; con TRACK_SKIP sólo se
        decodifican los frames que realmente se infieren (ahorro importante).
      - La sesión de DB vive y se usa únicamente en este hilo.
    """
    from src.database.db import (
        RepositorioCiclos, RepositorioEventos, RepositorioJobs, RepositorioZonas, get_session,
    )
    from src.core.geometry import GestorGeofencing

    info = probe_video(video_path)
    if info is None:
        raise RuntimeError("OpenCV no pudo abrir o decodificar el archivo de video.")

    rt.total_frames = max(info.frame_count, 1)
    rt.fps_video    = info.fps or 30.0

    detector = _get_detector(config)
    skip     = max(1, config.track_skip)
    imgsz    = max(64, config.track_imgsz)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV no pudo abrir el archivo de video.")

    frame_idx   = 0
    procesados  = 0
    ev_count    = 0
    ciclo_count = 0
    t_inicio    = time.perf_counter()

    try:
        with get_session() as session:
            repo        = RepositorioEventos(session)
            repo_ciclos = RepositorioCiclos(session)
            gestor      = GestorPresencia(frames_ausencia=config.frames_ausencia)
            gestor_geo  = GestorGeofencing()

            for zona in RepositorioZonas(session).cargar_activas():
                gestor_geo.agregar_zona(zona)

            while not rt.cancelar:
                # Avanzar sin decodificar a color los frames que se saltan.
                if not cap.grab():
                    break

                if frame_idx % skip == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        timestamp = calcular_timestamp(rt.fps_video, frame_idx)

                        nuevas_dets = detector.track_frame(
                            frame, frame_idx, timestamp, imgsz=imgsz,
                        )
                        mensajes = gestor.procesar(
                            nuevas_dets, frame_idx, repo, gestor_geo, repo_ciclos,
                        )
                        for msg in mensajes:
                            if "[APARICION]" in msg or "[DESAPARICION]" in msg:
                                ev_count += 1
                            elif "[ZONA]" in msg:
                                ciclo_count += 1

                        dets_out = []
                        for d in gestor.activos_para_dibujar():
                            estado_geo = gestor_geo._estados.get(d["maquina_id"])
                            dets_out.append({
                                "track_id":     d["track_id"],
                                "maquina_id":   d["maquina_id"],
                                "tipo_maquina": d["tipo_maquina"],
                                "confianza":    round(d["confianza"], 3),
                                "bbox":         [round(v, 1) for v in d["bbox"]],
                                "zona_actual":  estado_geo.zona_actual if estado_geo else None,
                            })

                        rt.eventos_nuevos    = ev_count
                        rt.ciclos_nuevos     = ciclo_count
                        rt.maquinas_activas  = gestor.n_activos
                        rt.detecciones_frame = dets_out
                        procesados += 1

                        # Commit periódico: libera el write-lock de SQLite y
                        # evita transacciones abiertas durante todo el video.
                        if config.flush_every_frames > 0 and procesados % config.flush_every_frames == 0:
                            session.commit()

                frame_idx += 1
                rt.frame_actual = frame_idx
                rt.progreso = round(min(frame_idx / rt.total_frames, 1.0) * 100, 1)

            gestor.cerrar(repo, gestor_geo)
            session.commit()

        rt.progreso = 100.0 if not rt.cancelar else rt.progreso

    finally:
        cap.release()

    rt.ts_fin = time.time()
    duracion  = round(time.perf_counter() - t_inicio, 2)
    fps_proc  = round(procesados / duracion, 2) if duracion > 0 else 0.0

    _actualizar_job(
        job_db_id,
        status="cancelled" if rt.cancelar else "completed",
        progress_percent=rt.progreso,
        finished_at=datetime.now(timezone.utc),
        metrics_json={
            "eventos_nuevos":   ev_count,
            "ciclos_nuevos":    ciclo_count,
            "frames_procesados": procesados,
            "frames_totales":   rt.total_frames,
            "track_skip":       skip,
            "track_imgsz":      imgsz,
            "duracion_seg":     duracion,
            "fps_procesamiento": fps_proc,
        },
    )


def _run_pipeline(job_db_id: int) -> None:
    """Ejecuta el análisis de un job y actualiza su estado en la DB."""
    with _RUNTIME_LOCK:
        rt = _RUNTIME.get(job_db_id)
    if rt is None:
        return

    config     = load_config()
    video_path = rt._video_path or rt._tmp_path
    sem        = _get_semaforo(config)

    # El job puede quedar encolado esperando turno.
    _actualizar_job(job_db_id, stage="en_cola")
    sem.acquire()

    try:
        if rt.cancelar:
            _actualizar_job(
                job_db_id, status="cancelled", stage="cancelado",
                finished_at=datetime.now(timezone.utc),
            )
            return

        rt.ts_inicio = time.time()
        _actualizar_job(
            job_db_id, status="processing", stage="detectando",
            started_at=datetime.now(timezone.utc),
        )

        if not video_path or not os.path.isfile(video_path):
            raise RuntimeError("Archivo de video no encontrado en disco.")

        _procesar_video(job_db_id, rt, video_path, config)

    except Exception as e:
        rt.error_msg = str(e)
        rt.ts_fin    = time.time()
        print(f"[VIDEO ROUTER] job={job_db_id} falló: {e}")
        import traceback
        traceback.print_exc()
        _actualizar_job(
            job_db_id, status="failed", stage="error",
            error_message=str(e), finished_at=datetime.now(timezone.utc),
        )
    finally:
        sem.release()
        if rt._tmp_path and os.path.exists(rt._tmp_path):
            try:
                os.remove(rt._tmp_path)
            except OSError:
                pass
        with _RUNTIME_LOCK:
            _RUNTIME.pop(job_db_id, None)


# ── Recuperación tras reinicio ────────────────────────────────────────────────

def recuperar_jobs_interrumpidos() -> None:
    """
    Al arrancar la API, los jobs "queued"/"processing" se marcan como
    "failed": sus hilos murieron con el proceso anterior y no se pueden
    reanudar. El usuario puede reintentarlos con POST /video/jobs/{id}/retry.
    """
    from src.database.db import get_session, RepositorioJobs

    with get_session() as session:
        repo = RepositorioJobs(session)
        jobs = (
            repo.obtener_por_estados(["queued", "processing"])
            if hasattr(repo, "obtener_por_estados")
            else repo.obtener_procesando()
        )
        for job in jobs:
            repo.actualizar(job.id, {
                "status": "failed",
                "error_message": "Interrumpido por reinicio del servidor",
                "finished_at": datetime.now(timezone.utc),
            })
            print(f"[RECOVERY] Job {job.id} marcado como failed (interrumpido por reinicio)")

    if jobs:
        print(f"[RECOVERY] {len(jobs)} job(s) recuperados tras reinicio")


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _buscar_video_por_checksum(db: Session, checksum: str) -> Video | None:
    """Devuelve el video no borrado con ese SHA-256 (el más reciente)."""
    return (
        db.query(Video)
        .filter(
            Video.checksum == checksum,
            or_(Video.retention_status.is_(None), Video.retention_status != "deleted"),
        )
        .order_by(Video.id.desc())
        .first()
    )


@router.post("/analizar", response_model=JobEstadoOut, status_code=202)
async def analizar_video(
    file: UploadFile = File(..., description="Archivo de video (.mp4, .avi, .mov, .mkv)"),
    db: Session = Depends(get_session_dep),
):
    """
    Sube un video, crea el registro en la tabla videos, crea el job en
    analysis_jobs y lanza el pipeline en background.
    """
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(415, f"Formato no soportado: {ext}. Usa: {', '.join(allowed_exts)}")

    config = load_config()
    max_mb = config.max_upload_mb

    # Verificar concurrencia (jobs activos en DB)
    activos = (
        db.query(AnalysisJob)
        .filter(AnalysisJob.status.in_(["queued", "processing"]))
        .count()
    )
    if activos >= config.analysis_max_activos:
        raise HTTPException(429, f"Hay {activos} análisis en curso. Espera a que terminen.")

    # ── 1. Guardar en streaming (no cargar 500 MB en RAM) ─────────────────────
    os.makedirs(config.videos_dir, exist_ok=True)
    storage_name = f"{uuid.uuid4().hex}{ext}"
    storage_path = os.path.join(config.videos_dir, storage_name)

    hasher    = hashlib.sha256()
    size      = 0
    chunk     = 1024 * 1024
    try:
        with open(storage_path, "wb") as f:
            while True:
                data = await file.read(chunk)
                if not data:
                    break
                size += len(data)
                if size > max_mb * 1024 * 1024:
                    raise HTTPException(
                        413, f"Archivo muy grande. Máximo: {max_mb} MB",
                    )
                hasher.update(data)
                f.write(data)
    except HTTPException:
        try:
            os.remove(storage_path)
        except OSError:
            pass
        raise
    finally:
        await file.close()

    if size == 0:
        os.remove(storage_path)
        raise HTTPException(422, "El archivo está vacío")

    checksum = hasher.hexdigest()

    # ── 2. Reutilizar el video si ya fue subido (mismo SHA-256) ────────────────
    video = _buscar_video_por_checksum(db, checksum)
    if video is not None and os.path.isfile(video.storage_key):
        try:
            os.remove(storage_path)
        except OSError:
            pass
        storage_path = video.storage_key
    else:
        info = probe_video(storage_path)
        if info is None or not info.readable:
            try:
                os.remove(storage_path)
            except OSError:
                pass
            raise HTTPException(422, "El archivo no es un video válido o está corrupto")

        video = Video(
            original_filename = file.filename or "video",
            storage_key       = storage_path,
            content_type      = file.content_type,
            file_size_bytes   = size,
            duration_seconds  = round(info.duration_seconds, 1),
            checksum          = checksum,
        )
        db.add(video)
        db.flush()

    # ── 3. Crear registros y COMMIT antes de lanzar el hilo ────────────────────
    job_db = AnalysisJob(
        video_id = video.id,
        status   = "queued",
        model_id = os.path.basename(config.model_path),
    )
    db.add(job_db)
    db.flush()
    job_db_id = job_db.id
    db.commit()
    db.refresh(job_db)

    # ── 4. Crear runtime y lanzar thread ───────────────────────────────────────
    rt = JobRuntime(
        job_db_id   = job_db_id,
        _video_path = storage_path,
    )
    with _RUNTIME_LOCK:
        _RUNTIME[job_db_id] = rt

    t = threading.Thread(target=_run_pipeline, args=(job_db_id,), daemon=True)
    rt._thread = t
    t.start()

    return _job_to_out(db, job_db, rt, video.original_filename)


@router.get("/estado/{job_id}", response_model=JobEstadoOut)
def estado_job(job_id: int, db: Session = Depends(get_session_dep)):
    """Retorna el estado actual del análisis. Sondear cada ~1 s."""
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    video = db.get(Video, job.video_id)
    nombre = video.original_filename if video else "?"

    with _RUNTIME_LOCK:
        rt = _RUNTIME.get(job_id)

    return _job_to_out(db, job, rt, nombre)


@router.get("/jobs", response_model=JobListPaginado)
def listar_jobs(
    estado:    str | None = Query(None, description="Filtrar por estado"),
    video_id:  int | None = Query(None, description="Filtrar por video"),
    page:      int        = Query(1, ge=1, description="Página"),
    page_size: int        = Query(20, ge=1, le=100, description="Elementos por página"),
    db: Session = Depends(get_session_dep),
):
    """Lista jobs desde la DB con filtros y paginación."""
    q = db.query(AnalysisJob)
    if estado:
        q = q.filter(AnalysisJob.status == estado)
    if video_id:
        q = q.filter(AnalysisJob.video_id == video_id)

    total = q.count()
    total_pages = max(1, (total + page_size - 1) // page_size)
    jobs = (
        q.order_by(AnalysisJob.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = []
    for j in jobs:
        v = db.get(Video, j.video_id)
        items.append(JobListOut(
            job_id        = j.id,
            video_id      = j.video_id,
            nombre_archivo = v.original_filename if v else "?",
            estado        = j.status,
            stage         = j.stage,
            progreso      = j.progress_percent,
            created_at    = j.created_at.isoformat() if j.created_at else "",
            started_at    = j.started_at.isoformat() if j.started_at else None,
            finished_at   = j.finished_at.isoformat() if j.finished_at else None,
            error_message = j.error_message,
        ))

    return JobListPaginado(
        items=items,
        paginacion=PaginacionMeta(
            page=page, page_size=page_size,
            total=total, total_pages=total_pages,
        ),
    )


@router.delete("/jobs/{job_id}", status_code=204)
def cancelar_job(job_id: int, db: Session = Depends(get_session_dep)):
    """Cancela un job en progreso."""
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    if job.status in ("queued", "processing"):
        job.status = "cancel_requested"
        db.flush()
        with _RUNTIME_LOCK:
            rt = _RUNTIME.get(job_id)
            if rt:
                rt.cancelar = True
    else:
        # Job ya terminado — no se puede cancelar, solo informar
        raise HTTPException(409, f"Job ya está en estado '{job.status}' — no se puede cancelar")


@router.post("/jobs/{job_id}/retry", response_model=JobEstadoOut)
def reintentar_job(job_id: int, db: Session = Depends(get_session_dep)):
    """Reintenta un job fallido o cancelado — crea una nueva ejecución."""
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    if job.status not in ("failed", "cancelled"):
        raise HTTPException(409, f"Solo se pueden reintentar jobs fallidos o cancelados (actual: {job.status})")

    video = db.get(Video, job.video_id)
    if video is None or not os.path.exists(video.storage_key):
        raise HTTPException(404, "Archivo de video no encontrado — no se puede reintentar")

    # Verificar concurrencia
    config = load_config()
    activos = (
        db.query(AnalysisJob)
        .filter(AnalysisJob.status.in_(["queued", "processing"]))
        .count()
    )
    if activos >= config.analysis_max_activos:
        raise HTTPException(429, f"Hay {activos} análisis en curso. Espera a que terminen.")

    # Crear NUEVO job (no reusar el anterior)
    nuevo_job = AnalysisJob(
        video_id    = job.video_id,
        status      = "queued",
        model_id    = job.model_id,
        retry_count = job.retry_count + 1,
    )
    db.add(nuevo_job)
    db.flush()
    db.commit()
    db.refresh(nuevo_job)

    rt = JobRuntime(
        job_db_id   = nuevo_job.id,
        _video_path = video.storage_key,
    )
    with _RUNTIME_LOCK:
        _RUNTIME[nuevo_job.id] = rt

    t = threading.Thread(target=_run_pipeline, args=(nuevo_job.id,), daemon=True)
    rt._thread = t
    t.start()

    return _job_to_out(db, nuevo_job, rt, video.original_filename)


# ── SSE: progreso en tiempo real ─────────────────────────────────────────────

@router.get("/estado/{job_id}/stream")
async def stream_estado(job_id: int):
    """
    Server-Sent Events (SSE) — progreso en tiempo real sin polling.

    El frontend abre una conexión con EventSource:
        const es = new EventSource('/video/estado/42/stream');
        es.onmessage = (e) => { const data = JSON.parse(e.data); ... };

    El servidor envía un evento cada 500ms con el estado actual del job.
    Cuando el job termina (completed/failed/cancelled), envía un evento final
    y cierra la conexión.

    Si el job no está corriendo, envía el estado persistente de la DB
    y cierra inmediatamente.
    """
    import json as _json
    from fastapi.responses import StreamingResponse
    from src.database.db import get_session
    from src.database.models import AnalysisJob, Video

    async def _generar_eventos():
        INTERVALO_MS = 500   # frecuencia de actualización

        # Verificar que el job existe y extraer datos básicos
        with get_session() as db:
            job  = db.get(AnalysisJob, job_id)
            if job is None:
                yield f"event: error\ndata: {_json.dumps({'detail': 'Job no encontrado'})}\n\n"
                return
            video = db.get(Video, job.video_id)
            nombre = video.original_filename if video else "?"
            # Extraer datos del job DENTRO de la sesión (evita DetachedInstanceError)
            job_status    = job.status
            job_progress   = job.progress_percent or 0.0
            job_started    = job.started_at
            job_finished   = job.finished_at
            job_error      = job.error_message
            job_video_id   = job.video_id

        # Estados terminales — enviar uno y cerrar
        ESTADOS_TERMINALES = {"completed", "failed", "cancelled"}

        while True:
            # Leer estado actual desde la DB (dentro de sesión)
            with get_session() as db:
                job = db.get(AnalysisJob, job_id)
                if job is None:
                    yield f"event: error\ndata: {_json.dumps({'detail': 'Job desaparecido'})}\n\n"
                    return
                job_status    = job.status
                job_progress   = job.progress_percent or 0.0
                job_started    = job.started_at
                job_finished   = job.finished_at
                job_error      = job.error_message
                video          = db.get(Video, job.video_id)
                nombre         = video.original_filename if video else "?"

            # Datos del runtime si está corriendo
            with _RUNTIME_LOCK:
                rt = _RUNTIME.get(job_id)

            if rt:
                payload = {
                    "job_id":           job_id,
                    "estado":           job_status,
                    "progreso":         rt.progreso,
                    "frame_actual":     rt.frame_actual,
                    "total_frames":     rt.total_frames,
                    "fps_video":        rt.fps_video,
                    "eventos_nuevos":   rt.eventos_nuevos,
                    "ciclos_nuevos":    rt.ciclos_nuevos,
                    "maquinas_activas": rt.maquinas_activas,
                    "detecciones":      rt.detecciones_frame,
                    "nombre_archivo":   nombre,
                }
            else:
                # Job no está en runtime — usar datos de la DB
                dur = None
                if job_started and job_finished:
                    dur = round((job_finished - job_started).total_seconds(), 1)
                payload = {
                    "job_id":         job_id,
                    "estado":         job_status,
                    "progreso":       job_progress,
                    "nombre_archivo": nombre,
                    "duracion_seg":   dur,
                    "error_msg":      job_error,
                }

            # Enviar evento
            yield f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"

            # Si el job está terminado, cerrar la conexión
            if job_status in ESTADOS_TERMINALES:
                yield f"event: done\ndata: {_json.dumps({'job_id': job_id, 'estado': job_status})}\n\n"
                return

            # Esperar antes del siguiente evento
            import asyncio
            await asyncio.sleep(INTERVALO_MS / 1000)

    return StreamingResponse(
        _generar_eventos(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering":  "no",   # nginx: no bufferear SSE
            "Connection":        "keep-alive",
        },
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _job_to_out(db: Session, job: AnalysisJob, rt: JobRuntime | None, nombre_archivo: str) -> JobEstadoOut:
    """Combina estado persistente (DB) + runtime (memoria) en la respuesta."""

    if rt:
        # Job en ejecución — usar datos del runtime
        dur = round((rt.ts_fin or time.time()) - rt.ts_inicio, 1) if rt.ts_inicio else None
        return JobEstadoOut(
            job_id            = job.id,
            video_id          = job.video_id,
            nombre_archivo    = nombre_archivo,
            estado            = job.status,
            stage             = job.stage,
            progreso          = rt.progreso,
            frame_actual      = rt.frame_actual,
            total_frames      = rt.total_frames,
            fps_video         = rt.fps_video,
            eventos_nuevos    = rt.eventos_nuevos,
            ciclos_nuevos     = rt.ciclos_nuevos,
            maquinas_activas  = rt.maquinas_activas,
            detecciones_frame = [DeteccionOut(**d) for d in rt.detecciones_frame],
            error_msg         = rt.error_msg or job.error_message,
            duracion_seg      = dur,
        )
    else:
        # Job no está corriendo — solo datos de la DB
        dur = None
        if job.started_at and job.finished_at:
            dur = round((job.finished_at - job.started_at).total_seconds(), 1)

        return JobEstadoOut(
            job_id            = job.id,
            video_id          = job.video_id,
            nombre_archivo    = nombre_archivo,
            estado            = job.status,
            stage             = job.stage,
            progreso          = job.progress_percent or 0.0,
            frame_actual      = 0,
            total_frames      = 0,
            fps_video         = 0.0,
            eventos_nuevos    = (job.metrics_json or {}).get("eventos_nuevos", 0) if job.metrics_json else 0,
            ciclos_nuevos     = (job.metrics_json or {}).get("ciclos_nuevos", 0) if job.metrics_json else 0,
            maquinas_activas  = 0,
            detecciones_frame = [],
            error_msg         = job.error_message,
            duracion_seg      = dur,
        )
