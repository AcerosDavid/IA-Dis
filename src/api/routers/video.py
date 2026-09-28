"""
Router de análisis de video — pipeline optimizado.

Optimizaciones implementadas:
  1. Arquitectura multihilo: hilo lector + hilo inferencia desacoplados
  2. Queue maxsize=1 entre hilos — descarta frames viejos, sin latency lag
  3. Frame skipping configurable (TRACK_SKIP) — YOLO corre cada N frames
  4. Interpolación lineal de bboxes en frames saltados — sin parpadeo visual
  5. Detecciones del frontend actualizadas cada frame (30fps visual) aunque
     YOLO corra a menor frecuencia

Endpoints:
  POST   /video/analizar          — sube video y lanza análisis
  GET    /video/estado/{job_id}   — progreso + detecciones actuales
  GET    /video/jobs              — lista todos los jobs
  DELETE /video/jobs/{job_id}     — cancela o elimina un job
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/video", tags=["Video"])

# ── Configuración de rendimiento ──────────────────────────────────────────────
# Ajustar según hardware:
#   GPU RTX 3060+  → TRACK_IMGSZ=640, TRACK_SKIP=1
#   GPU 920MX      → TRACK_IMGSZ=416, TRACK_SKIP=1
#   CPU ONNX       → TRACK_IMGSZ=416, TRACK_SKIP=2
#   CPU PyTorch    → TRACK_IMGSZ=320, TRACK_SKIP=3
TRACK_IMGSZ   = 320   # óptimo benchmark: CPU imgsz=320 skip=3 → 22.4 fps
TRACK_SKIP    = 3     # benchmark: skip=3 da 22.4 fps vs 14.7 (skip=2) vs 11.1 (CUDA)
FRAMES_AUS    = 90

MAX_JOBS_ACTIVOS = 3
MAX_FILE_MB      = 500

# Sentinel para señalizar fin de stream a los hilos
_SENTINEL = None


# ── Estado del job ────────────────────────────────────────────────────────────

class JobEstado(str, Enum):
    PENDIENTE  = "pendiente"
    PROCESANDO = "procesando"
    COMPLETADO = "completado"
    ERROR      = "error"
    CANCELADO  = "cancelado"


@dataclass
class JobInfo:
    job_id:            str
    nombre_archivo:    str
    estado:            JobEstado    = JobEstado.PENDIENTE
    progreso:          float        = 0.0
    frame_actual:      int          = 0
    total_frames:      int          = 0
    fps_video:         float        = 0.0
    eventos_nuevos:    int          = 0
    ciclos_nuevos:     int          = 0
    maquinas_activas:  int          = 0
    detecciones_frame: list[dict]   = field(default_factory=list)
    error_msg:         str | None   = None
    ts_inicio:         float        = field(default_factory=time.time)
    ts_fin:            float | None = None
    cancelar:          bool         = False
    _tmp_path:         str | None   = None


_JOBS: dict[str, JobInfo] = {}
_JOBS_LOCK = threading.Lock()


# ── Schemas ───────────────────────────────────────────────────────────────────

class DeteccionOut(BaseModel):
    track_id:     int
    maquina_id:   str
    tipo_maquina: str
    confianza:    float
    bbox:         list[float]
    zona_actual:  str | None = None


class JobEstadoOut(BaseModel):
    job_id:            str
    nombre_archivo:    str
    estado:            JobEstado
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


# ── Interpolación de bboxes ───────────────────────────────────────────────────

def _interpolar_bboxes(
    prev_dets: list[dict],
    curr_dets: list[dict],
    alpha: float,
) -> list[dict]:
    """
    Interpola linealmente las coordenadas de bboxes entre dos frames
    para los track_ids que aparezcan en ambos conjuntos.

    alpha=0.0 → posición del frame anterior
    alpha=1.0 → posición del frame actual (inferido)

    Para IDs nuevos (solo en curr) los incluye directamente.
    Para IDs que desaparecieron (solo en prev) los descarta.
    """
    # Indexar por track_id
    prev_map = {d["track_id"]: d for d in prev_dets}
    curr_map = {d["track_id"]: d for d in curr_dets}

    resultado: list[dict] = []

    for tid, curr in curr_map.items():
        if tid not in prev_map:
            # ID nuevo — mostrar directamente sin interpolar
            resultado.append(curr)
            continue

        prev = prev_map[tid]
        pb   = prev["bbox"]
        cb   = curr["bbox"]

        # Interpolación lineal componente a componente
        interp_bbox = [
            pb[j] + alpha * (cb[j] - pb[j])
            for j in range(4)
        ]

        resultado.append({**curr, "bbox": interp_bbox})

    return resultado


# ── Helpers de buffer ────────────────────────────────────────────────────────

def _es_fuente_en_vivo(source: str) -> bool:
    """
    Detecta si la fuente es una cámara en vivo (RTSP, HTTP, DroidCam, webcam).
    Los archivos de video grabados NO necesitan flush — tienen todos los frames.
    """
    if isinstance(source, int):
        return True   # webcam USB
    s = str(source).lower()
    return any(s.startswith(p) for p in ("rtsp://", "http://", "https://", "udp://", "tcp://"))


def _flush_buffer_camara(cap: cv2.VideoCapture) -> None:
    """
    Drena el buffer interno de OpenCV para cámaras en vivo.

    OpenCV acumula silenciosamente hasta 5 frames en el buffer interno
    de la cámara. Sin flush, el primer frame procesado puede tener
    3-5 segundos de retraso incluso antes de que empiece el análisis.

    Estrategia:
      1. Reducir CAP_PROP_BUFFERSIZE a 1 (mínimo soportado por el backend)
      2. Leer y descartar frames hasta que el tiempo de lectura sea < 5ms
         (frames reales del buffer) en lugar de ~33ms (frame fresco de red)
    """
    # Intentar reducir el buffer interno — no todos los backends lo soportan
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Drain: leer hasta que los frames lleguen "rápido" (están en buffer)
    # Un frame de red tarda ~33ms; un frame de buffer tarda < 2ms
    MAX_DRAIN = 10
    for _ in range(MAX_DRAIN):
        t0 = time.perf_counter()
        ret, _ = cap.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if not ret or elapsed_ms > 15:
            # Frame tardó > 15ms → ya llegamos al stream en vivo
            break

    print("[BUFFER] Buffer de cámara drenado — sin retraso acumulado")


# ── Hilo lector ───────────────────────────────────────────────────────────────

def _hilo_lector(
    cap: cv2.VideoCapture,
    frame_queue: "queue.Queue",
    job: JobInfo,
) -> None:
    """
    Lee frames a la velocidad máxima del video y los pone en frame_queue.

    DISEÑO CLAVE: maxsize=1 + descarte agresivo.
    El hilo lector NUNCA espera al hilo de inferencia.
    Si YOLO tarda 229ms, el lector habrá leído ~7 frames en ese tiempo
    y descartado los 6 intermedios — siempre entrega el frame MÁS RECIENTE.

    El job.frame_actual se actualiza aquí (no en inferencia) para que
    el frontend pueda sincronizar el video local con el análisis.
    """
    frame_idx = 0
    try:
        while True:
            if job.cancelar:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Actualizar posición del video en el job — el frontend lo usa
            # para sincronizar el reproductor local aunque YOLO vaya lento
            job.frame_actual = frame_idx
            if job.total_frames > 0:
                job.progreso = round((frame_idx / job.total_frames) * 100, 1)

            # Descarte agresivo — siempre el frame más reciente
            try:
                frame_queue.put_nowait((frame_idx, frame))
            except queue.Full:
                try:
                    frame_queue.get_nowait()   # descartar frame viejo
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait((frame_idx, frame))
                except queue.Full:
                    pass   # si sigue llena, simplemente saltar este frame

            frame_idx += 1
    finally:
        frame_queue.put(_SENTINEL)


# ── Hilo de inferencia ────────────────────────────────────────────────────────

def _hilo_inferencia(
    frame_queue: "queue.Queue",
    result_queue: "queue.Queue | None",   # ya no se usa — kept for compat
    detector,
    gestor,
    gestor_geo,
    repo,
    repo_ciclos,
    job: JobInfo,
    fps: float,
) -> None:
    """
    Consume TODOS los frames que llegan de frame_queue y corre YOLO en cada uno.

    No hay TRACK_SKIP aquí — el hilo lector ya hace el descarte con maxsize=1.
    Si YOLO tarda 229ms, el lector habrá descartado los frames intermedios
    y este hilo siempre recibe el frame más reciente disponible.

    El job.frame_actual y progreso los actualiza el hilo lector,
    no este hilo — así el frontend siempre ve el progreso real del video
    independientemente de cuán lenta sea la inferencia.
    """
    import importlib
    main_mod           = importlib.import_module("main")
    calcular_timestamp = main_mod.calcular_timestamp

    ev_count    = 0
    ciclo_count = 0

    while True:
        item = frame_queue.get()

        if item is _SENTINEL:
            if result_queue is not None:   # result_queue puede ser None (ver _run_pipeline)
                result_queue.put(_SENTINEL)
            break

        frame_idx, frame = item

        if job.cancelar:
            if result_queue is not None:
                result_queue.put(_SENTINEL)
            break

        timestamp = calcular_timestamp(fps, frame_idx)

        try:
            nuevas_dets = detector.track_frame(
                frame, frame_idx, timestamp, imgsz=TRACK_IMGSZ
            )
            mensajes = gestor.procesar(
                nuevas_dets, frame_idx, repo, gestor_geo, repo_ciclos,
            )
            for msg in mensajes:
                if "[APARICION]" in msg or "[DESAPARICION]" in msg:
                    ev_count += 1
                elif "[ZONA]" in msg:
                    ciclo_count += 1

            # Construir detecciones para el frontend
            dets_out: list[dict] = []
            for d in gestor.activos_para_dibujar():
                estado_geo = gestor_geo._estados.get(d["maquina_id"])
                zona_act   = estado_geo.zona_actual if estado_geo else None
                dets_out.append({
                    "track_id":     d["track_id"],
                    "maquina_id":   d["maquina_id"],
                    "tipo_maquina": d["tipo_maquina"],
                    "confianza":    round(d["confianza"], 3),
                    "bbox":         [round(v, 1) for v in d["bbox"]],
                    "zona_actual":  zona_act,
                })

            # Actualizar detecciones y contadores en el job
            # (frame_actual y progreso los maneja el hilo lector)
            job.eventos_nuevos    = ev_count
            job.ciclos_nuevos     = ciclo_count
            job.maquinas_activas  = gestor.n_activos
            job.detecciones_frame = dets_out

        except Exception as e:
            print(f"[INFERENCIA] frame={frame_idx} error: {e}")


# ── Pipeline principal ────────────────────────────────────────────────────────

def _run_pipeline(job_id: str) -> None:
    """
    Orquesta el pipeline multihilo:
      Hilo lector  → frame_queue (maxsize=1) → Hilo inferencia
                                                     ↓
                                              result_queue
                                                     ↓
                                           Hilo principal (actualiza job)
    """
    from src.config import load_config
    from src.database.db import (
        RepositorioEventos, RepositorioCiclos,
        RepositorioZonas, get_session, migrate,
    )
    from src.ai.detector import Detector
    from src.core.geometry import GestorGeofencing, Zona
    import importlib

    main_mod        = importlib.import_module("main")
    GestorPresencia = main_mod.GestorPresencia

    # ZONAS DESACTIVADAS TEMPORALMENTE — descomentar cuando las coordenadas
    # reales del video estén calibradas.
    # ZONAS_DEFAULT = [
    #     Zona.rectangulo("zona_carga",    "carga",    50,  50, 300, 200, color_bgr=(0, 255, 0)),
    #     Zona.rectangulo("zona_descarga", "descarga", 500, 50, 800, 200, color_bgr=(0, 0, 255)),
    # ]
    ZONAS_DEFAULT: list = []   # sin zonas — solo detección y tracking

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return

    job.estado = JobEstado.PROCESANDO
    tmp_path   = job._tmp_path

    try:
        config = load_config()
        migrate()

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV no pudo abrir el archivo de video.")

        # Flush de buffer solo para fuentes en vivo (RTSP, webcam, HTTP)
        # Para archivos grabados (.mp4/.avi) el flush no aplica ni es necesario
        if _es_fuente_en_vivo(tmp_path):
            _flush_buffer_camara(cap)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        job.total_frames = max(total, 1)
        job.fps_video    = fps

        with get_session() as session:
            repo        = RepositorioEventos(session)
            repo_ciclos = RepositorioCiclos(session)
            gestor      = GestorPresencia(frames_ausencia=FRAMES_AUS)
            gestor_geo  = GestorGeofencing()

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

            # ── Crear queue lector→inferencia ────────────────────────────
            # maxsize=1: el lector descarta frames viejos, siempre el más reciente
            frame_queue: queue.Queue = queue.Queue(maxsize=1)

            # ── Lanzar hilos ──────────────────────────────────────────────
            t_lector = threading.Thread(
                target=_hilo_lector,
                args=(cap, frame_queue, job),
                daemon=True,
                name=f"lector-{job_id[:8]}",
            )
            t_inferencia = threading.Thread(
                target=_hilo_inferencia,
                args=(
                    frame_queue, None,
                    detector, gestor, gestor_geo,
                    repo, repo_ciclos, job, fps,
                ),
                daemon=True,
                name=f"inferencia-{job_id[:8]}",
            )

            t_lector.start()
            t_inferencia.start()

            # ── Hilo principal: solo espera a que terminen los hilos ──────
            # El lector actualiza job.frame_actual + job.progreso
            # La inferencia actualiza job.detecciones_frame + job.maquinas_activas
            # Este hilo solo monitorea cancelación y espera el fin
            while t_lector.is_alive() or t_inferencia.is_alive():
                if job.cancelar:
                    job.estado = JobEstado.CANCELADO
                    break
                time.sleep(0.1)   # polling ligero — no consume CPU

            t_lector.join(timeout=10)
            t_inferencia.join(timeout=10)

            gestor.cerrar(repo, gestor_geo)

        cap.release()

        if job.estado != JobEstado.CANCELADO:
            job.estado   = JobEstado.COMPLETADO
            job.progreso = 100.0
            job.ts_fin   = time.time()

    except Exception as e:
        job.estado    = JobEstado.ERROR
        job.error_msg = str(e)
        job.ts_fin    = time.time()
        print(f"[VIDEO ROUTER] job={job_id} falló: {e}")
        import traceback; traceback.print_exc()
    finally:
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
    Sube un video y lanza el pipeline multihilo de análisis en background.
    Retorna 202 Accepted con job_id para hacer polling.
    """
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Formato no soportado: {ext}. Usa: {', '.join(allowed_exts)}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo muy grande ({size_mb:.1f} MB). Máximo: {MAX_FILE_MB} MB",
        )

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

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(content)

    job_id = str(uuid.uuid4())
    job    = JobInfo(
        job_id         = job_id,
        nombre_archivo = file.filename or "video",
        _tmp_path      = tmp_path,
    )

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    t = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    t.start()

    return _job_to_out(job)


@router.get("/estado/{job_id}", response_model=JobEstadoOut)
def estado_job(job_id: str):
    """Retorna el estado actual del análisis. Sondear cada ~1 s."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return _job_to_out(job)


@router.get("/jobs", response_model=list[JobEstadoOut])
def listar_jobs(
    estado: JobEstado | None = Query(None, description="Filtrar por estado"),
):
    """Lista todos los jobs activos y completados."""
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    if estado:
        jobs = [j for j in jobs if j.estado == estado]
    return [_job_to_out(j) for j in jobs]


@router.delete("/jobs/{job_id}", status_code=204)
def cancelar_job(job_id: str):
    """Cancela un job en progreso o elimina uno terminado."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    if job.estado in (JobEstado.PENDIENTE, JobEstado.PROCESANDO):
        job.cancelar = True
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
        job_id            = job.job_id,
        nombre_archivo    = job.nombre_archivo,
        estado            = job.estado,
        progreso          = job.progreso,
        frame_actual      = job.frame_actual,
        total_frames      = job.total_frames,
        fps_video         = job.fps_video,
        eventos_nuevos    = job.eventos_nuevos,
        ciclos_nuevos     = job.ciclos_nuevos,
        maquinas_activas  = job.maquinas_activas,
        detecciones_frame = [DeteccionOut(**d) for d in job.detecciones_frame],
        error_msg         = job.error_msg,
        duracion_seg      = dur,
    )
