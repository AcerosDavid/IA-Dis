"""
Gestión de trabajos de análisis batch.

Orquesta el análisis asíncrono de videos:
  1. El usuario carga un video → se copia a data/videos/ y se crea un Job en DB.
  2. El GestorJobs lanza un Procesador en un hilo de fondo (uno a la vez por defecto).
  3. La UI consulta el estado del job en cada poll.
  4. Al terminar, se generan el reporte PDF/Excel y se actualiza el job.

No usa Celery, Redis ni FastAPI: para una app de escritorio con SQLite,
threading + la DB es la solución correcta y suficiente.
"""
from __future__ import annotations

import datetime
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings, load_config
from src.database.db import RepositorioJobs, get_session, migrate
from src.pipeline.runner import EventoUI, OpcionesAnalisis, Procesador


@dataclass
class JobProgressInfo:
    """Snapshot del estado de un job para consultar desde la UI."""
    job_id: int
    estado: str
    progreso: float
    frame_actual: int
    total_frames: int
    fps: float
    activos: int
    eventos: int
    ciclos: int
    mensaje: str
    error: str | None
    nombre: str
    ruta_salida: str | None


def _iso_ahora() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _copiar_video(origen: str, destino_dir: str) -> str:
    """Copia el video de origen a la carpeta de videos del programa."""
    Path(destino_dir).mkdir(parents=True, exist_ok=True)
    nombre = os.path.basename(origen)
    # Evitar sobrescribir: si ya existe, añadir timestamp
    destino = os.path.join(destino_dir, nombre)
    if os.path.exists(destino):
        stem, ext = os.path.splitext(nombre)
        destino = os.path.join(destino_dir, f"{stem}_{int(time.time())}{ext}")
    shutil.copy2(origen, destino)
    return destino


def _probe_video(path: str) -> dict:
    """Obtiene metadatos del video (duración, resolución, FPS)."""
    import cv2
    cap = cv2.VideoCapture(path)
    info = {"duracion_seg": 0.0, "resolucion": "", "fps_originales": 0.0, "total_frames": 0}
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        info["fps_originales"] = float(fps) if fps and fps > 0 else 0.0
        info["total_frames"] = total
        info["resolucion"] = f"{w}x{h}" if w and h else ""
        if fps and total:
            info["duracion_seg"] = total / fps
        cap.release()
    return info


class GestorJobs:
    """
    Gestiona la cola de trabajos de análisis batch.

    Mantiene un hilo worker que procesa jobs uno a la vez (configurable).
    La UI interactúa vía métodos thread-safe (usa Lock para el estado en memoria).
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or load_config()
        self._lock = threading.Lock()
        self._cola: queue.Queue[int] = queue.Queue()  # job_ids pendientes
        self._procesadores: dict[int, Procesador] = {}
        self._job_eventos: dict[int, list[EventoUI]] = {}  # eventos por job para la UI
        self._worker_thread: threading.Thread | None = None
        self._detener_worker = threading.Event()
        self._running = False
        self._ultimo_error: str | None = None

        # Asegurar que las carpetas existen
        for d in (self.config.videos_dir, self.config.output_dir, self.config.reports_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

        # Recuperar jobs interrumpidos al iniciar
        try:
            migrate()
            with get_session() as session:
                RepositorioJobs(session).marcar_interrumpidos()
        except Exception as e:
            print(f"[GESTOR JOBS] No se pudieron recuperar jobs interrumpidos: {e}")

        self._iniciar_worker()

    # ── Worker loop ─────────────────────────────────────────────────────────────

    def _iniciar_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._detener_worker.clear()
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="batch-worker"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Bucle que procesa jobs de la cola uno a la vez."""
        while not self._detener_worker.is_set():
            try:
                job_id = self._cola.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._procesar_job(job_id)
            except Exception as e:
                print(f"[GESTOR JOBS] Error procesando job {job_id}: {e}")
                with get_session() as session:
                    RepositorioJobs(session).actualizar(job_id, {
                        "estado": "error",
                        "error_msg": str(e),
                        "finalizado_en": _iso_ahora(),
                    })
                self._ultimo_error = str(e)
            finally:
                with self._lock:
                    self._procesadores.pop(job_id, None)
                    self._job_eventos.pop(job_id, None)

    # ── API pública ─────────────────────────────────────────────────────────────

    def crear_job(
        self,
        ruta_video: str,
        guardar_db: bool = True,
        usar_ocr: bool = False,
        usar_clasificador: bool = False,
        track_skip: int | None = None,
        track_imgsz: int | None = None,
        confianza: float | None = None,
    ) -> dict:
        """
        Crea un nuevo trabajo de análisis.

        1. Copia el video a data/videos/
        2. Obtiene metadatos (duración, resolución, FPS)
        3. Crea el registro en DB con estado 'pendiente'
        4. Encola el job para procesamiento

        Returns:
            Dict con job_id, nombre, y metadatos del video.
        """
        if not os.path.exists(ruta_video):
            raise FileNotFoundError(f"Video no encontrado: {ruta_video}")

        # Copiar video a la carpeta del programa
        ruta_local = _copiar_video(ruta_video, self.config.videos_dir)
        nombre = os.path.basename(ruta_local)

        # Obtener metadatos
        info = _probe_video(ruta_local)

        # Ruta de salida del video anotado
        stem, _ = os.path.splitext(nombre)
        ruta_salida = os.path.join(self.config.output_dir, f"{stem}_anotado.mp4")

        # Crear job en DB
        with get_session() as session:
            repo = RepositorioJobs(session)
            job = repo.crear({
                "nombre_original":  nombre,
                "ruta_original":     ruta_local,
                "ruta_salida":       ruta_salida,
                "estado":            "pendiente",
                "progreso":          0.0,
                "total_frames":      info["total_frames"],
                "fps_video":          info["fps_originales"],
                "duracion_seg":       info["duracion_seg"],
                "resolucion":         info["resolucion"],
                "fps_originales":     info["fps_originales"],
                "modelo":             self.config.model_path,
                "confianza_umbral":   confianza if confianza is not None else self.config.confidence_threshold,
                "track_skip":         track_skip if track_skip is not None else self.config.track_skip,
                "track_imgsz":        track_imgsz if track_imgsz is not None else self.config.track_imgsz,
                "dispositivo":        "cpu" if self.config.forzar_cpu else "auto",
                "guardar_db":         guardar_db,
                "usar_ocr":           usar_ocr,
                "usar_clasificador":  usar_clasificador,
                "creado_en":          _iso_ahora(),
            })
            job_id = job.id

        # Encolar
        self._cola.put(job_id)

        return {
            "job_id": job_id,
            "nombre": nombre,
            "ruta_local": ruta_local,
            "duracion_seg": info["duracion_seg"],
            "resolucion": info["resolucion"],
            "fps": info["fps_originales"],
            "total_frames": info["total_frames"],
        }

    def cancelar_job(self, job_id: int) -> None:
        """Detiene un job activo o lo saca de la cola."""
        with self._lock:
            proc = self._procesadores.get(job_id)
        if proc is not None and proc.is_alive():
            proc.detener()
        # Marcar como cancelado en DB
        with get_session() as session:
            RepositorioJobs(session).actualizar(job_id, {
                "estado": "cancelado",
                "finalizado_en": _iso_ahora(),
            })

    def eliminar_job(self, job_id: int) -> dict:
        """
        Elimina un job de la DB y borra todos los archivos asociados:
        video original, video anotado y reportes.
        """
        with get_session() as session:
            repo = RepositorioJobs(session)
            info = repo.eliminar(job_id)
        # Borrar archivos físicos
        borrados = []
        for label, ruta in info.get("rutas", {}).items():
            if ruta and os.path.exists(ruta):
                try:
                    os.remove(ruta)
                    borrados.append(ruta)
                except OSError as e:
                    print(f"[GESTOR JOBS] No se pudo borrar {ruta}: {e}")
        info["borrados"] = borrados
        return info

    def obtener_estado(self, job_id: int) -> JobProgressInfo | None:
        """Retorna el snapshot del estado de un job."""
        with get_session() as session:
            job = RepositorioJobs(session).obtener(job_id)
            if job is None:
                return None

            # Si hay un procesador activo, usar su estado en tiempo real
            with self._lock:
                proc = self._procesadores.get(job_id)
            fps_actual = 0.0
            activos = 0
            eventos = 0
            ciclos = 0
            if proc is not None and proc.is_alive():
                fps_actual = proc.estado.fps
                frame_actual = proc.estado.frame_actual
                total = proc.estado.total_frames
                progreso = proc.estado.progreso
                mensaje = proc.estado.mensaje
                error = proc.estado.error
                activos = proc.estado.activos
                eventos = proc.estado.eventos
                ciclos = proc.estado.ciclos
                if proc.terminado and error:
                    estado_runtime = "error"
                elif proc.terminado:
                    estado_runtime = "completado"
                else:
                    estado_runtime = "procesando"
            else:
                frame_actual = job.frame_actual
                total = job.total_frames
                progreso = job.progreso
                mensaje = job.estado.upper()
                error = job.error_msg
                estado_runtime = job.estado

            return JobProgressInfo(
                job_id=job_id,
                estado=estado_runtime,
                progreso=progreso,
                frame_actual=frame_actual,
                total_frames=total,
                fps=fps_actual,
                activos=activos,
                eventos=eventos,
                ciclos=ciclos,
                mensaje=mensaje,
                error=error,
                nombre=job.nombre_original,
                ruta_salida=job.ruta_salida,
            )

    def drenar_eventos(self, job_id: int) -> list[EventoUI]:
        """Retorna los eventos acumulados para un job y limpia el buffer."""
        with self._lock:
            eventos = self._job_eventos.get(job_id, [])
            self._job_eventos[job_id] = []
            return eventos

    def listar_jobs(self) -> list[dict]:
        """Lista todos los jobs con su estado actual."""
        with get_session() as session:
            jobs = RepositorioJobs(session).listar_todos()
            resultado = []
            for job in jobs:
                # Verificar si el video de salida existe físicamente
                video_disponible = bool(
                    job.ruta_salida and os.path.exists(job.ruta_salida)
                )
                resultado.append({
                    "id": job.id,
                    "nombre": job.nombre_original,
                    "estado": job.estado,
                    "progreso": job.progreso,
                    "frame_actual": job.frame_actual,
                    "total_frames": job.total_frames,
                    "fps_video": job.fps_video,
                    "duracion_seg": job.duracion_seg or 0.0,
                    "resolucion": job.resolucion,
                    "modelo": job.modelo,
                    "total_detecciones": job.total_detecciones,
                    "total_eventos": job.total_eventos,
                    "total_unicos": job.total_unicos,
                    "confianza_promedio": job.confianza_promedio,
                    "tiempo_procesamiento": job.tiempo_procesamiento,
                    "ruta_salida": job.ruta_salida,
                    "ruta_reporte_pdf": job.ruta_reporte_pdf,
                    "ruta_reporte_xlsx": job.ruta_reporte_xlsx,
                    "video_disponible": video_disponible,
                    "error_msg": job.error_msg,
                    "creado_en": job.creado_en,
                    "finalizado_en": job.finalizado_en,
                })
            return resultado

    def shutdown(self) -> None:
        """Detiene el worker y todos los procesadores activos."""
        self._detener_worker.set()
        with self._lock:
            for proc in self._procesadores.values():
                if proc.is_alive():
                    proc.detener()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)

    # ── Procesamiento de un job ──────────────────────────────────────────────────

    def _procesar_job(self, job_id: int) -> None:
        """Procesa un job individual de principio a fin."""
        with get_session() as session:
            job = RepositorioJobs(session).obtener(job_id)
            if job is None:
                raise RuntimeError(f"Job {job_id} no encontrado")
            ruta_original = job.ruta_original
            ruta_salida = job.ruta_salida
            guardar_db = job.guardar_db
            usar_ocr = job.usar_ocr
            usar_clasif = job.usar_clasificador
            track_skip = job.track_skip or self.config.track_skip
            track_imgsz = job.track_imgsz or self.config.track_imgsz
            confianza = job.confianza_umbral or self.config.confidence_threshold

        # Marcar como procesando
        with get_session() as session:
            RepositorioJobs(session).actualizar(job_id, {
                "estado": "procesando",
                "iniciado_en": _iso_ahora(),
            })

        # Crear config específica para este job
        config_job = Settings(
            model_path=self.config.model_path,
            confidence_threshold=confianza,
            track_skip=track_skip,
            track_imgsz=track_imgsz,
            forzar_cpu=self.config.forzar_cpu,
        )

        opciones = OpcionesAnalisis(
            guardar_db=guardar_db,
            usar_ocr=usar_ocr,
            usar_clasificador=usar_clasif,
            track_skip=track_skip,
            track_imgsz=track_imgsz,
            display_width=0,  # no mostrar preview en batch
            exportar_video=True,
            ruta_salida=ruta_salida,
        )

        proc = Procesador(ruta_original, opciones, config_job)
        with self._lock:
            self._procesadores[job_id] = proc

        t_inicio = time.perf_counter()
        proc.start()

        # Monitorear progreso y actualizar DB periódicamente
        eventos_contador = {"aparicion": 0, "desaparicion": 0, "zona": 0}
        ultimo_update_db = 0.0

        while proc.is_alive():
            # Drenar eventos: contar para el reporte y almacenar para la UI
            while True:
                try:
                    ev = proc.eventos.get_nowait()
                except queue.Empty:
                    break
                if ev.tipo == "APARICION":
                    eventos_contador["aparicion"] += 1
                elif ev.tipo == "DESAPARICION":
                    eventos_contador["desaparicion"] += 1
                elif ev.tipo == "ZONA":
                    eventos_contador["zona"] += 1
                self._job_eventos.setdefault(job_id, []).append(ev)

            ahora = time.perf_counter()
            if ahora - ultimo_update_db >= 2.0:
                ultimo_update_db = ahora
                est = proc.estado
                # Actualizar DB cada 2s
                try:
                    with get_session() as session:
                        RepositorioJobs(session).actualizar(job_id, {
                            "progreso": est.progreso,
                            "frame_actual": est.frame_actual,
                            "total_frames": est.total_frames,
                        })
                except Exception as e:
                    print(f"[GESTOR JOBS] Error actualizando DB: {e}")

            time.sleep(0.3)

        proc.join()

        est = proc.estado
        elapsed = time.perf_counter() - t_inicio

        if est.error:
            with get_session() as session:
                RepositorioJobs(session).actualizar(job_id, {
                    "estado": "error",
                    "error_msg": est.error,
                    "progreso": est.progreso,
                    "frame_actual": est.frame_actual,
                    "finalizado_en": _iso_ahora(),
                })
            raise RuntimeError(est.error)

        # Calcular estadísticas desde la DB
        stats = self._calcular_stats()

        # Generar reportes
        ruta_pdf, ruta_xlsx = self._generar_reportes(
            job_id, ruta_original, stats, eventos_contador, elapsed
        )

        # Marcar como completado
        with get_session() as session:
            RepositorioJobs(session).actualizar(job_id, {
                "estado": "completado",
                "progreso": 100.0,
                "frame_actual": est.frames_procesados,
                "total_frames": est.total_frames,
                "total_detecciones": stats["total_detecciones"],
                "total_eventos": stats["total_eventos"],
                "total_unicos": stats["total_unicos"],
                "confianza_promedio": stats["confianza_promedio"],
                "tiempo_procesamiento": round(elapsed, 2),
                "ruta_reporte_pdf": ruta_pdf,
                "ruta_reporte_xlsx": ruta_xlsx,
                "finalizado_en": _iso_ahora(),
            })

        print(f"[GESTOR JOBS] Job {job_id} completado en {elapsed:.1f}s")

    def _calcular_stats(self) -> dict:
        """Calcula estadísticas de detección desde la DB."""
        from sqlalchemy import func
        try:
            with get_session() as session:
                from src.database.models import Evento
                total_detecciones = session.query(func.count(Evento.id)).scalar() or 0
                total_eventos = total_detecciones
                conf_avg = session.query(func.avg(Evento.confianza)).scalar() or 0.0
                # Únicos por tipo_maquina + clase_id (proxy de vehículos únicos)
                total_unicos = (
                    session.query(Evento.tipo_maquina)
                    .distinct().count()
                )
                return {
                    "total_detecciones": int(total_detecciones),
                    "total_eventos": int(total_eventos),
                    "total_unicos": int(total_unicos),
                    "confianza_promedio": round(float(conf_avg), 3),
                }
        except Exception as e:
            print(f"[GESTOR JOBS] Error calculando stats: {e}")
            return {
                "total_detecciones": 0, "total_eventos": 0,
                "total_unicos": 0, "confianza_promedio": 0.0,
            }

    def _generar_reportes(
        self,
        job_id: int,
        ruta_video: str,
        stats: dict,
        eventos_contador: dict,
        elapsed: float,
    ) -> tuple[str | None, str | None]:
        """Genera los reportes PDF y Excel."""
        try:
            from src.reports.generador import generar_reporte_pdf, generar_reporte_excel

            stem = f"reporte_{job_id}"
            ruta_pdf = os.path.join(self.config.reports_dir, f"{stem}.pdf")
            ruta_xlsx = os.path.join(self.config.reports_dir, f"{stem}.xlsx")

            # Datos del job
            with get_session() as session:
                job = RepositorioJobs(session).obtener(job_id)
                job_data = {
                    "nombre": job.nombre_original,
                    "duracion_seg": job.duracion_seg,
                    "resolucion": job.resolucion,
                    "fps_originales": job.fps_originales,
                    "modelo": job.modelo,
                    "confianza_umbral": job.confianza_umbral,
                    "track_skip": job.track_skip,
                    "track_imgsz": job.track_imgsz,
                    "dispositivo": job.dispositivo,
                    "creado_en": job.creado_en,
                }

            generar_reporte_pdf(
                ruta_pdf, job_data, stats, eventos_contador, elapsed
            )
            generar_reporte_excel(
                ruta_xlsx, job_data, stats, eventos_contador
            )
            return ruta_pdf, ruta_xlsx
        except Exception as e:
            print(f"[GESTOR JOBS] Error generando reportes: {e}")
            return None, None
