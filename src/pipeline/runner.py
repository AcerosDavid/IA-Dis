"""
Motor de procesamiento de video/camara para la aplicacion de escritorio.

Ejecuta YOLO + tracker + presencia + geofencing + (OCR / clasificador opcional)
en un hilo de fondo y publica el frame anotado y los eventos para la interfaz.

No depende de ninguna API: la UI consume `Procesador` directamente.
"""
from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.ai.clasificador_estado import ClasificadorEstado
from src.ai.clasificador_lstm import ClasificadorLSTM
from src.ai.detector import Detector
from src.ai.ocr_flota import LectorFlota
from src.config import Settings, load_config
from src.core.drawing import (
    dibujar_hud,
    dibujar_panel_flota,
    dibujar_tracking,
    escalar_frame,
)
from src.core.geometry import GestorGeofencing, dibujar_zonas
from src.database.db import (
    RepositorioCiclos,
    RepositorioEventos,
    RepositorioZonas,
    get_session,
    migrate,
)
from src.pipeline.presencia import GestorPresencia, calcular_timestamp


# ── Modelos de datos de la UI ─────────────────────────────────────────────────

@dataclass
class OpcionesAnalisis:
    """Opciones que el usuario configura en la interfaz."""
    guardar_db:       bool = True
    usar_ocr:         bool = False
    usar_clasificador: bool = False
    track_skip:       int  = 3
    track_imgsz:      int  = 320
    display_width:    int  = 960


@dataclass
class EstadoAnalisis:
    """Snapshot del estado del procesamiento (lectura desde la UI)."""
    fuente:       str   = ""
    ejecutando:   bool  = False
    pausado:      bool  = False
    frame_actual: int   = 0
    total_frames: int   = 0
    fps:          float = 0.0
    activos:      int   = 0
    eventos:      int   = 0
    ciclos:       int   = 0
    progreso:     float = 0.0
    mensaje:      str   = "Listo"
    error:        str | None = None


@dataclass
class EventoUI:
    """Evento listo para mostrarse en la tabla de la interfaz."""
    timestamp: str
    tipo:      str
    maquina:   str
    track_id:  int | str
    confianza: float | None = None
    zona:      str | None = None


# ── Repositorios nulos (cuando el usuario no quiere guardar en BD) ────────────

class _RepoEventosNulo:
    def insertar(self, *_args, **_kwargs) -> None:
        return None


class _RepoCiclosNulo:
    def registrar_evento(self, *_args, **_kwargs) -> None:
        return None


# ── Parseo de mensajes de presencia ──────────────────────────────────────────

_RE_APARICION = re.compile(
    r"\[APARICION\]\s+id=(\d+)\s*\|\s*maquina=(\w+).*?conf=([\d.]+)"
)
_RE_DESAPARICION = re.compile(
    r"\[DESAPARICION\]\s+id=(\d+)\s*\|\s*maquina=(\w+)"
)
_RE_ZONA = re.compile(
    r"\[ZONA\]\s+id=(\d+)\s*\|\s*(\w+)\s*->\s*(\w+)"
)


def _evento_desde_mensaje(msg: str, timestamp: str) -> EventoUI | None:
    """Convierte un mensaje textual de GestorPresencia en un EventoUI."""
    m = _RE_APARICION.search(msg)
    if m:
        return EventoUI(
            timestamp=timestamp, tipo="APARICION", maquina=m.group(2),
            track_id=int(m.group(1)), confianza=float(m.group(3)),
        )
    m = _RE_DESAPARICION.search(msg)
    if m:
        return EventoUI(
            timestamp=timestamp, tipo="DESAPARICION", maquina=m.group(2),
            track_id=int(m.group(1)),
        )
    m = _RE_ZONA.search(msg)
    if m:
        return EventoUI(
            timestamp=timestamp, tipo="ZONA", maquina="",
            track_id=int(m.group(1)), zona=f"{m.group(2)} -> {m.group(3)}",
        )
    return None


# ── Procesador ───────────────────────────────────────────────────────────────

class Procesador(threading.Thread):
    """
    Hilo de analisis. La UI llama `pausar()`, `reanudar()` y `detener()`.

    Publica el ultimo frame en `obtener_frame()` y los eventos en
    `eventos` (queue.Queue) para que la interfaz los consuma sin bloquear.
    """

    def __init__(
        self,
        fuente:  str | int,
        opciones: OpcionesAnalisis,
        config:  Settings | None = None,
    ) -> None:
        super().__init__(daemon=True, name="procesador-mineria")
        self.fuente    = fuente
        self.opciones  = opciones
        self.config    = config or load_config()
        self.estado    = EstadoAnalisis(fuente=str(fuente))
        self.eventos:  queue.Queue[EventoUI] = queue.Queue()

        self._frame_lock  = threading.Lock()
        self._ultimo_frame: np.ndarray | None = None
        self._pausado     = threading.Event()
        self._detener     = threading.Event()
        self._es_camara   = self._detectar_camara(fuente)
        self._terminado   = False

    # ── Control desde la UI ───────────────────────────────────────────────────

    def pausar(self) -> None:
        self._pausado.set()
        self.estado.pausado = True

    def reanudar(self) -> None:
        self._pausado.clear()
        self.estado.pausado = False

    def detener(self) -> None:
        self._detener.set()
        self._pausado.clear()

    @property
    def terminado(self) -> bool:
        return self._terminado

    def obtener_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._ultimo_frame

    def _publicar_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._ultimo_frame = frame

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def run(self) -> None:
        self.estado.ejecutando = True
        self.estado.mensaje = "Iniciando..."
        cap = None
        try:
            migrate()
            cap = self._abrir_fuente()
            fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.estado.total_frames = total

            self.estado.mensaje = "Cargando modelo YOLO..."
            detector = Detector(self.config.model_path, self.config.confidence_threshold)
            detector.load_model()

            self.estado.mensaje = "Procesando..."
            with get_session() as session:
                if self.opciones.guardar_db:
                    repo        = RepositorioEventos(session)
                    repo_ciclos = RepositorioCiclos(session)
                else:
                    repo        = _RepoEventosNulo()
                    repo_ciclos = _RepoCiclosNulo()

                gestor_geo = self._cargar_zonas(session)
                gestor     = GestorPresencia(frames_ausencia=self.config.frames_ausencia)

                lector_ocr = None
                if self.opciones.usar_ocr:
                    lector_ocr = LectorFlota()
                    threading.Thread(
                        target=lector_ocr._init_reader, daemon=True, name="ocr-init",
                    ).start()

                clasificador = None
                clf_lstm = None
                if self.opciones.usar_clasificador:
                    clasificador = ClasificadorEstado()
                    clf_lstm = ClasificadorLSTM(device="cpu")

                self._loop(
                    cap, fps_video, total, detector, gestor, gestor_geo,
                    repo, repo_ciclos, lector_ocr, clasificador, clf_lstm,
                    session,
                )

        except Exception as e:
            self.estado.error = str(e)
            self.estado.mensaje = f"Error: {e}"
            print(f"[PROCESADOR ERROR] {e}")
        finally:
            if cap is not None:
                cap.release()
            self.estado.ejecutando = False
            if self.estado.error is None:
                self.estado.mensaje = "Finalizado"
            self._terminado = True

    # ── Loop principal ────────────────────────────────────────────────────────

    def _loop(
        self,
        cap,
        fps_video: float,
        total_frames: int,
        detector: Detector,
        gestor: GestorPresencia,
        gestor_geo: GestorGeofencing,
        repo,
        repo_ciclos,
        lector_ocr: LectorFlota | None,
        clasificador: ClasificadorEstado | None,
        clf_lstm: ClasificadorLSTM | None,
        session,
    ) -> None:
        frame_idx     = 0
        ultimas_dets: list[dict] = []
        numeros: dict[int, str]  = {}
        estados: dict[int, str]  = {}

        t_fps      = time.perf_counter()
        frames_fps = 0

        while not self._detener.is_set():
            if self._pausado.is_set():
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                if self._es_camara:
                    time.sleep(0.05)
                    continue
                break

            timestamp = calcular_timestamp(fps_video, frame_idx)
            paso = max(1, self.opciones.track_skip)

            if frame_idx % paso == 0:
                detecciones = detector.track_frame(
                    frame, frame_idx, timestamp, imgsz=self.opciones.track_imgsz,
                )
                mensajes = gestor.procesar(
                    detecciones, frame_idx, repo, gestor_geo, repo_ciclos,
                )

                for msg in mensajes:
                    print(msg)
                    ev = _evento_desde_mensaje(msg, timestamp)
                    if ev is not None:
                        self.eventos.put(ev)
                    if "[APARICION]" in msg or "[DESAPARICION]" in msg:
                        self.estado.eventos += 1
                        if "[DESAPARICION]" in msg:
                            self._limpiar_track(msg, lector_ocr, clasificador, clf_lstm)
                    elif "[ZONA]" in msg:
                        self.estado.ciclos += 1

                activos = gestor.activos_para_dibujar()

                if activos:
                    self._clasificar(frame, activos, detector, clasificador, clf_lstm, frame_idx)
                    if clf_lstm is not None and frame_idx % 300 == 0:
                        clf_lstm.limpiar_inactivos(frame_idx)

                ultimas_dets = activos
                numeros = lector_ocr.todos_conocidos() if lector_ocr else {}
                if clf_lstm is not None:
                    estados = clf_lstm.todos_estados()
                elif clasificador is not None:
                    estados = clasificador.todos_estados()
                else:
                    estados = {}

                frames_fps += 1
                elapsed = time.perf_counter() - t_fps
                if elapsed >= 2.0:
                    self.estado.fps = frames_fps / elapsed
                    frames_fps = 0
                    t_fps = time.perf_counter()

            # ── Anotar y publicar ─────────────────────────────────────────────
            frame_vis = frame.copy()
            dibujar_zonas(frame_vis, gestor_geo)
            dibujar_tracking(frame_vis, ultimas_dets, numeros, estados)
            if numeros:
                dibujar_panel_flota(frame_vis, numeros)

            ts_disp = calcular_timestamp(fps_video, frame_idx)
            hud = (
                f"{ts_disp} | activos={gestor.n_activos} "
                f"| eventos={self.estado.eventos} | {self.estado.fps:.0f} fps"
            )
            dibujar_hud(frame_vis, hud, pausado=self.estado.pausado)

            frame_vis = escalar_frame(frame_vis, self.opciones.display_width)
            self._publicar_frame(frame_vis)

            self.estado.frame_actual = frame_idx
            self.estado.activos = gestor.n_activos
            if total_frames > 0:
                self.estado.progreso = min(100.0, (frame_idx + 1) / total_frames * 100)

            frame_idx += 1
            if session is not None and frame_idx % max(1, self.config.flush_every_frames) == 0:
                try:
                    session.commit()
                except Exception:
                    session.rollback()

        # Registrar desapariciones pendientes al terminar
        try:
            mensajes_cierre = gestor.cerrar(repo, gestor_geo)
            ts_cierre = calcular_timestamp(fps_video, frame_idx)
            for msg in mensajes_cierre:
                ev = _evento_desde_mensaje(msg, ts_cierre)
                if ev is not None:
                    self.eventos.put(ev)
                self.estado.eventos += 1
        except Exception as e:
            print(f"[PROCESADOR] No se pudo cerrar presencia: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detectar_camara(fuente: str | int) -> bool:
        """True para webcams (indice) y streams RTSP/HTTP."""
        if isinstance(fuente, int):
            return True
        if isinstance(fuente, str):
            valor = fuente.strip().lower()
            return valor.isdigit() or valor.startswith(("rtsp://", "http://", "https://"))
        return False

    def _abrir_fuente(self) -> cv2.VideoCapture:
        if self._es_camara:
            if isinstance(self.fuente, str) and self.fuente.strip().isdigit():
                source: str | int = int(self.fuente.strip())
            else:
                source = self.fuente
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            ruta = str(self.fuente)
            if not os.path.exists(ruta):
                raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
            cap = cv2.VideoCapture(ruta)

        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {self.fuente}")
        return cap

    def _cargar_zonas(self, session) -> GestorGeofencing:
        gestor_geo = GestorGeofencing()
        try:
            for zona in RepositorioZonas(session).cargar_activas():
                gestor_geo.agregar_zona(zona)
        except Exception as e:
            print(f"[PROCESADOR] No se pudieron cargar zonas: {e}")
        return gestor_geo

    def _clasificar(
        self,
        frame: np.ndarray,
        activos: list[dict],
        detector: Detector,
        clasificador: ClasificadorEstado | None,
        clf_lstm: ClasificadorLSTM | None,
        frame_idx: int,
    ) -> None:
        for det in activos:
            tid  = det["track_id"]
            tipo = det["tipo_maquina"]

            if clf_lstm is not None:
                estado_lstm, _conf = clf_lstm.procesar(
                    frame, det["bbox"], tipo, tid, frame_idx,
                )
                if estado_lstm:
                    det["estado"] = estado_lstm
                    continue

            if clasificador is not None:
                estado_visual = clasificador.clasificar(
                    frame, det["bbox"], tipo, tid, frame_idx,
                )
                if estado_visual:
                    hist = detector._historial.get(tid)
                    es_estatico = hist.es_estatico if hist else True
                    det["estado"] = (
                        clasificador.estado_combinado(tid, tipo, es_estatico)
                        or estado_visual
                    )

    @staticmethod
    def _limpiar_track(
        msg: str,
        lector_ocr: LectorFlota | None,
        clasificador: ClasificadorEstado | None,
        clf_lstm: ClasificadorLSTM | None,
    ) -> None:
        m = re.search(r"id=(\d+)", msg)
        if not m:
            return
        tid = int(m.group(1))
        if lector_ocr is not None:
            lector_ocr.limpiar_track(tid)
        if clasificador is not None:
            clasificador.limpiar_track(tid)
        if clf_lstm is not None:
            clf_lstm.limpiar_track(tid)
