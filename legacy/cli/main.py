"""
Entry point CLI del sistema de deteccion de maquinaria minera.

Uso:
    python main.py                              # 1 video (desde .env VIDEO_PATH)
    python main.py video1.mp4                   # 1 video explícito
    python main.py video1.mp4 video2.mp4        # 2 videos simultáneos
    python main.py video1.mp4 video2.mp4 v3.mp4 # hasta 3 videos

Pipeline multihilo por video (3 hilos + sub-hilo OCR):
  Hilo LECTOR     -> lee frames a velocidad del video (~30fps)
  Hilo INFERENCIA -> YOLO + BoT-SORT + filtro estaticos (~5fps)
                     lanza sub-hilo OCR para leer numeros de flota
  Sub-hilo OCR    -> EasyOCR en background (no bloquea YOLO)
  Hilo DISPLAY    -> video fluido (30fps) + bboxes + numero de flota
                     (hilo principal — OpenCV requiere el hilo principal)

Controles:
    Q / ESC  -- salir todos los videos
    ESPACIO  -- pausar / reanudar todos
    1 / 2 / 3 -- pausar/reanudar video específico
"""
from __future__ import annotations

import os
import queue
import sys
import warnings
# Suprimir warnings de deprecación de PyTorch (quantize, etc.) — solo ruido visual
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import load_config
from src.database.db import (
    RepositorioEventos,
    RepositorioCiclos,
    RepositorioZonas,
    get_session,
    migrate,
)
from src.ai.detector import Detector
from src.ai.ocr_flota import LectorFlota
from src.ai.clasificador_estado import ClasificadorEstado
from src.ai.clasificador_lstm import ClasificadorLSTM
from src.core.geometry import GestorGeofencing, dibujar_zonas
from src.pipeline.presencia import GestorPresencia, calcular_timestamp

COLORES: dict[str, tuple[int, int, int]] = {
    # BGR — colores neón que contrastan con marrón/gris de la mina
    "camion":     (0, 255, 50),    # verde neón brillante
    "excavadora": (0, 200, 255),   # amarillo neón
    "pala":       (255, 50, 180),  # magenta/rosa neón
}
COLOR_DEFAULT = (255, 255, 255)   # blanco puro

# Frames consecutivos sin ver un ID antes de considerarlo desaparecido.
# Con QUEUE_SIZE=4 y 5fps de inferencia, el tracker procesa ~5fps reales.
# Pero frame_count viene del video (30fps) — necesita ser proporcional.
# 30fps * 6 segundos = 180 frames de video sin ver el ID → desaparicion
FRAMES_AUSENCIA = 180

# ── Rendimiento ───────────────────────────────────────────────────────────────
TRACK_IMGSZ   = 320
DISPLAY_WIDTH = 960

# ── OCR de numeros de flota ───────────────────────────────────────────────────
# True = leer numeros de camion en background (requiere EasyOCR instalado)
# False = desactivar completamente (mas rapido, sin lectura de numeros)
OCR_HABILITADO = True

# ── Clasificador de estado ────────────────────────────────────────────────────
# True = clasificar estado de cada máquina (empty/loaded, digging/idle, etc.)
# Requiere modelos entrenados: python entrenar_clasificador.py
# False = desactivar (más rápido, sin clasificación de estado)
CLASIFICADOR_HABILITADO = True

# ── Zonas ─────────────────────────────────────────────────────────────────────
# DESACTIVADAS TEMPORALMENTE — descomentar cuando coordenadas esten calibradas
# ZONAS_DEFAULT = [
#     Zona.rectangulo("zona_carga",    "carga",    50,  50, 300, 200, color_bgr=(0, 255, 0)),
#     Zona.rectangulo("zona_descarga", "descarga", 500, 50, 800, 200, color_bgr=(0, 0, 255)),
# ]
ZONAS_DEFAULT: list = []

_STOP = object()

# ── Configuracion de la queue ─────────────────────────────────────────────────
# QUEUE_SIZE controla cuantos frames puede acumular el lector esperando
# al inferidor. Valor critico para el tracking:
#
#   maxsize=1  → descarta 5 de cada 6 frames (30fps video / 5fps YOLO)
#                El tracker ve saltos de ~200ms entre frames consecutivos
#                → pierde IDs facilmente
#
#   maxsize=4  → el lector acumula hasta 4 frames antes de descartar
#                El tracker ve frames mas cercanos en tiempo
#                → IDs mas estables, a costa de ~130ms de latencia extra
#
# Para video grabado (no camara en vivo) usar maxsize=4
# Para camara RTSP en vivo usar maxsize=1 (latencia minima)
QUEUE_SIZE = 4


# ── Estado compartido entre hilos ─────────────────────────────────────────────

@dataclass
class EstadoCompartido:
    """Datos que el hilo de inferencia publica y el hilo display lee."""
    detecciones:      list[dict]          = field(default_factory=list)
    numeros_flota:    dict[int, str]      = field(default_factory=dict)  # track_id -> numero
    estados_maquina:  dict[int, str]      = field(default_factory=dict)  # track_id -> estado
    n_activos:        int                 = 0
    event_count:      int                 = 0
    ciclo_count:      int                 = 0
    fps_infer:        float               = 0.0
    frames_inferidos: int                 = 0
    pausado:          bool                = False
    salir:            bool                = False
    video_terminado:  bool                = False
    ultimo_frame:     np.ndarray | None   = None
    _lock:            threading.Lock      = field(default_factory=threading.Lock)

    def actualizar_dets(
        self,
        dets:    list[dict],
        n:       int,
        ev:      int,
        ci:      int,
        numeros: dict[int, str],
        estados: dict[int, str] | None = None,
    ) -> None:
        with self._lock:
            self.detecciones    = dets
            self.n_activos      = n
            self.event_count    = ev
            self.ciclo_count    = ci
            self.numeros_flota  = numeros
            if estados is not None:
                self.estados_maquina = estados
            self.frames_inferidos += 1

    def actualizar_frame(self, frame: np.ndarray) -> None:
        """Publica el frame mas reciente para el hilo display (sin competir con inferencia)."""
        with self._lock:
            self.ultimo_frame = frame

    def leer_frame(self) -> np.ndarray | None:
        """Retorna el ultimo frame disponible para display."""
        with self._lock:
            return self.ultimo_frame

    def leer_dets(self) -> tuple[list[dict], dict[int, str], dict[int, str]]:
        """Retorna (detecciones, numeros_flota, estados_maquina)."""
        with self._lock:
            return (
                list(self.detecciones),
                dict(self.numeros_flota),
                dict(self.estados_maquina),
            )


# ── Helpers ───────────────────────────────────────────────────────────────────
# GestorPresencia y calcular_timestamp viven en src/pipeline/presencia.py y se
# importan arriba. Se mantienen accesibles como main.GestorPresencia y
# main.calcular_timestamp para compatibilidad con los tests existentes.

def open_video(video_path: str) -> cv2.VideoCapture:
    if not os.path.exists(video_path):
        print(f"[VIDEO ERROR] Archivo no encontrado: {video_path}")
        raise SystemExit(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VIDEO ERROR] OpenCV no puede decodificar: {video_path}")
        raise SystemExit(1)
    w, h  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {w}x{h} @ {fps:.1f}fps | {total} fotogramas")
    return cap


def print_evento(det: dict) -> None:
    tid = det.get("track_id", "?")
    print(
        f"[EVENTO] fotograma={det['numero_fotograma']} | timestamp={det['timestamp']} "
        f"| id={tid} | maquina={det['tipo_maquina']} | clase={det['clase_id']} "
        f"| confianza={det['confianza']:.2f}"
    )


def dibujar_tracking(
    frame:         np.ndarray,
    detecciones:   list[dict],
    numeros_flota: dict[int, str],
    estados:       dict[int, str] | None = None,
) -> None:
    """
    Dibuja bboxes estilo SmartPSS con esquinas marcadas.
    Muestra número de flota (OCR) y estado operativo (clasificador).
    """
    # Colores neón por estado — contrastan con marrón/gris de la mina
    ESTADO_COLORES = {
        # ── Camión (nuevos estados) ───────────────────────────────────────────
        "vacio":            (200, 200, 255),   # lavanda  — tolva vacía
        "cargado":          (0,   255,  50),   # verde neón — tolva cargada ✅
        # ── Excavadora / Pala (nuevos estados) ───────────────────────────────
        "escarbando_cargando": (0, 220, 255),  # amarillo neón — en operación
        "descargando":         (0,  50, 255),  # rojo brillante — volcando
        # ── Estados legacy (ClasificadorEstado CNN) — retrocompatibilidad ─────
        "empty":            (200, 200, 200),
        "loaded":           (0,   255,  50),
        "dumping":          (0,    50, 255),
        "traveling_empty":  (180, 180, 255),
        "traveling_loaded": (0,   200,  50),
        "waiting_empty":    (150, 150, 150),
        "being_loaded":     (0,   255, 200),
        "digging":          (0,   220, 255),
        "loading_truck":    (50,  255, 255),
        "traveling":        (200, 200, 255),
        "scooping":         (255,  50, 200),
        "dumping_truck":    (255, 150,   0),
    }

    for det in detecciones:
        tipo  = det["tipo_maquina"]
        color = COLORES.get(tipo, COLOR_DEFAULT)
        x1, y1, x2, y2 = map(int, det["bbox"])
        tid   = det.get("track_id", "?")
        w_box = x2 - x1
        h_box = y2 - y1

        # Estado del dict de detección (puesto por hilo_inferencia)
        estado_maq = det.get("estado") or (estados.get(tid) if estados else None)

        # Color del bbox según estado si está disponible
        color_bbox = ESTADO_COLORES.get(estado_maq, color) if estado_maq else color

        # Bbox sólido grueso — más visible sobre fondo marrón/gris
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bbox, 3)
        # Segunda pasada semitransparente para dar profundidad
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bbox, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # Esquinas gruesas estilo SmartPSS — más visibles
        cs = min(w_box, h_box, 25)
        for (ox, oy, dx, dy) in [
            (x1, y1,  cs, 0), (x1, y1, 0,  cs),
            (x2, y1, -cs, 0), (x2, y1, 0,  cs),
            (x1, y2,  cs, 0), (x1, y2, 0, -cs),
            (x2, y2, -cs, 0), (x2, y2, 0, -cs),
        ]:
            cv2.line(frame, (ox, oy), (ox + dx, oy + dy), color_bbox, 4)

        # Etiqueta principal
        numero = numeros_flota.get(tid)
        if numero:
            label_top = f"#{numero}"
            label_bot = f"ID:{tid} {tipo} {det['confianza']:.2f}"
        else:
            label_top = None
            label_bot  = f"ID:{tid} {tipo} {det['confianza']:.2f}"

        # Etiqueta inferior — fondo sólido con borde
        (tw, th), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        # Fondo negro sólido para máximo contraste
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_bbox, 2)
        cv2.putText(frame, label_bot, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_bbox, 1, cv2.LINE_AA)

        # Badge de estado operativo — grande y sólido
        if estado_maq:
            color_est   = ESTADO_COLORES.get(estado_maq, (200, 200, 200))
            label_est   = estado_maq.upper().replace("_", " ")
            (ew, eh), _ = cv2.getTextSize(label_est, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
            ex1 = x1
            ey1 = y2 + 3
            ex2 = x1 + ew + 12
            ey2 = y2 + eh + 12
            # Fondo sólido negro + borde de color
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 0), -1)
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), color_est, 2)
            cv2.putText(frame, label_est, (ex1 + 6, ey2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_est, 2, cv2.LINE_AA)

        # Número de flota — grande y visible
        if label_top:
            (nw, nh), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            nx = x1 + (w_box - nw) // 2
            ny = y1 + (h_box + nh) // 2
            # Sombra negra gruesa para visibilidad
            cv2.putText(frame, label_top, (nx + 2, ny + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, label_top, (nx, ny),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bbox, 3, cv2.LINE_AA)


def _cargar_zonas(session) -> tuple[GestorGeofencing, bool]:
    repo_zonas = RepositorioZonas(session)
    gestor_geo = GestorGeofencing()
    zonas_db   = repo_zonas.cargar_activas()
    if zonas_db:
        for zona in zonas_db:
            gestor_geo.agregar_zona(zona)
        print(f"[INFO] Geofencing: {len(zonas_db)} zonas desde DB")
        return gestor_geo, True
    else:
        for zona in ZONAS_DEFAULT:
            gestor_geo.agregar_zona(zona)
        if ZONAS_DEFAULT:              # guardar fuera del for — evita guardar vacio
            for zona in ZONAS_DEFAULT:
                repo_zonas.guardar(zona)
        n = len(ZONAS_DEFAULT)
        print(f"[INFO] Geofencing: {'desactivado' if n == 0 else f'{n} zonas cargadas'}")
        return gestor_geo, n > 0


# ── Hilos ─────────────────────────────────────────────────────────────────────

def hilo_lector(
    cap:        cv2.VideoCapture,
    frame_q:    queue.Queue,
    display_q:  queue.Queue,          # queue dedicada para el display (maxsize=1)
    estado:     EstadoCompartido,
    fps_video:  float,
) -> None:
    """
    Lee frames y los publica en DOS queues independientes:
      frame_q   → hilo inferencia (YOLO/tracker)
      display_q → hilo display (siempre el frame más reciente)
    """
    frame_idx  = 0
    delay_read = 1.0 / fps_video if fps_video > 0 else 1 / 30

    try:
        while not estado.salir:
            if estado.pausado:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[LECTOR] Fin del video.")
                break

            # ── Queue inferencia ──────────────────────────────────────────────
            if QUEUE_SIZE == 1:
                # Cámara en vivo: descartar frame viejo si llena
                try:
                    frame_q.put_nowait((frame_idx, frame))
                except queue.Full:
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_q.put_nowait((frame_idx, frame))
                    except queue.Full:
                        pass
            else:
                # Video grabado: acumular hasta QUEUE_SIZE frames
                try:
                    frame_q.put_nowait((frame_idx, frame))
                except queue.Full:
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_q.put_nowait((frame_idx, frame))
                    except queue.Full:
                        pass

            # ── Queue display (siempre el más reciente, nunca bloquea) ────────
            try:
                display_q.put_nowait(frame)
            except queue.Full:
                try:
                    display_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    display_q.put_nowait(frame)
                except queue.Full:
                    pass

            frame_idx += 1

            elapsed = time.perf_counter() - t0
            sleep   = delay_read - elapsed
            if sleep > 0:
                time.sleep(sleep)
    finally:
        estado.video_terminado = True
        frame_q.put(_STOP)
        display_q.put(_STOP)
        print(f"[LECTOR] Terminado. Frames leidos: {frame_idx}")


def hilo_inferencia(
    frame_q:      queue.Queue,
    detector:     Detector,
    gestor:       GestorPresencia,
    gestor_geo:   GestorGeofencing,
    repo:         RepositorioEventos,
    repo_ciclos:  RepositorioCiclos,
    estado:       EstadoCompartido,
    fps_video:    float,
    lector_ocr:   LectorFlota | None,
    clasificador: ClasificadorEstado | None = None,
    clf_lstm:     ClasificadorLSTM | None  = None,
    session=None,
) -> None:
    """
    Consume frames, corre YOLO + BoT-SORT + filtro estaticos.
    Lanza OCR en sub-hilo para no bloquear la inferencia.
    Clasifica estado de cada maquina con LSTM (preferido) o CNN estática.
    """
    ev_count    = 0
    ciclo_count = 0
    t_ini_fps   = time.perf_counter()
    frames_fps  = 0

    # Sub-hilo OCR: procesa frames en background
    ocr_queue: queue.Queue = queue.Queue(maxsize=4)

    def _ocr_worker() -> None:
        """Sub-hilo: lee la cola OCR y corre EasyOCR sin bloquear YOLO."""
        while True:
            item = ocr_queue.get()
            if item is _STOP:
                break
            frame_ocr, dets_ocr, frame_num = item
            for det in dets_ocr:
                lector_ocr.leer(
                    frame_ocr, det["bbox"],
                    det["track_id"], frame_num,
                )

    if lector_ocr is not None:
        t_ocr = threading.Thread(target=_ocr_worker, daemon=True, name="ocr")
        t_ocr.start()
    else:
        t_ocr = None

    try:
        while not estado.salir:
            item = frame_q.get(timeout=2.0)
            if item is _STOP:
                frame_q.put(_STOP)
                break

            frame_idx, frame = item
            timestamp = calcular_timestamp(fps_video, frame_idx)

            try:
                detecciones = detector.track_frame(
                    frame, frame_idx, timestamp, imgsz=TRACK_IMGSZ
                )
                mensajes = gestor.procesar(
                    detecciones, frame_idx, repo, gestor_geo, repo_ciclos,
                )
                for msg in mensajes:
                    print(msg)
                    if "[APARICION]" in msg or "[DESAPARICION]" in msg:
                        ev_count += 1
                        if "[DESAPARICION]" in msg:
                            try:
                                tid_str = msg.split("id=")[1].split(" ")[0]
                                tid_int = int(tid_str)
                                if lector_ocr is not None:
                                    lector_ocr.limpiar_track(tid_int)
                                if clasificador is not None:
                                    clasificador.limpiar_track(tid_int)
                                if clf_lstm is not None:
                                    clf_lstm.limpiar_track(tid_int)
                            except Exception:
                                pass
                    elif "[ZONA]" in msg:
                        ciclo_count += 1

                activos = gestor.activos_para_dibujar()

                # ── Clasificación de estado ───────────────────────────────────
                # Prioridad: LSTM (temporal) > CNN estática > sin estado
                if activos:
                    for det in activos:
                        tid  = det["track_id"]
                        tipo = det["tipo_maquina"]

                        # 1. Intentar con LSTM temporal (más preciso)
                        if clf_lstm is not None:
                            estado_lstm, conf_lstm = clf_lstm.procesar(
                                frame, det["bbox"], tipo, tid, frame_idx
                            )
                            if estado_lstm:
                                det["estado"] = estado_lstm
                                continue   # LSTM ganó — no necesitamos CNN

                        # 2. Fallback: CNN estática (ClasificadorEstado)
                        if clasificador is not None:
                            estado_visual = clasificador.clasificar(
                                frame, det["bbox"], tipo, tid, frame_idx,
                            )
                            if estado_visual:
                                hist = detector._historial.get(tid)
                                es_estatico = hist.es_estatico if hist else True
                                estado_final = clasificador.estado_combinado(
                                    tid, tipo, es_estatico,
                                )
                                det["estado"] = estado_final or estado_visual

                    # Limpiar LSTM de tracks desaparecidos
                    if clf_lstm is not None and frame_idx % 300 == 0:
                        clf_lstm.limpiar_inactivos(frame_idx)

                # Encolar para OCR (no bloqueante)
                if lector_ocr is not None and activos:
                    try:
                        ocr_queue.put_nowait((frame.copy(), activos, frame_idx))
                    except queue.Full:
                        pass

                # Publicar al display
                numeros  = lector_ocr.todos_conocidos() if lector_ocr else {}
                # Estados: preferir LSTM, completar con CNN si hay
                if clf_lstm is not None:
                    estados = clf_lstm.todos_estados()
                elif clasificador is not None:
                    estados = clasificador.todos_estados()
                else:
                    estados = {}
                estado.actualizar_dets(
                    activos, gestor.n_activos,
                    ev_count, ciclo_count,
                    numeros, estados,
                )

            except Exception as e:
                print(f"[INFERENCIA] frame={frame_idx} error: {e}")

            # Commit periódico: la sesión pertenece a este hilo y no debe
            # mantener una transacción abierta durante todo el video.
            if session is not None and frame_idx % 60 == 0:
                try:
                    session.commit()
                except Exception:
                    session.rollback()

            # Calcular FPS de inferencia
            frames_fps += 1
            elapsed_fps = time.perf_counter() - t_ini_fps
            if elapsed_fps >= 2.0:
                estado.fps_infer = frames_fps / elapsed_fps
                frames_fps = 0
                t_ini_fps  = time.perf_counter()

    except queue.Empty:
        print("[INFERENCIA] Timeout esperando frames.")
    finally:
        if t_ocr is not None:
            ocr_queue.put(_STOP)
            t_ocr.join(timeout=5)
        print(f"[INFERENCIA] Terminado. Inferencias: {estado.frames_inferidos}")


# ── Pipeline por video ────────────────────────────────────────────────────────

@dataclass
class PipelineVideo:
    """Encapsula un pipeline completo para un único video."""
    video_path:  str
    indice:      int              # 1-based: 1, 2, 3
    ventana:     str              # nombre de la ventana OpenCV
    estado:      EstadoCompartido = field(default_factory=EstadoCompartido)
    cap:         object           = field(default=None)   # cv2.VideoCapture
    fps:         float            = 30.0
    t_lector:    object           = field(default=None)
    t_infer:     object           = field(default=None)
    frame_vis_cache: object       = field(default=None)
    ultimo_frame_id: object       = field(default=None)
    frame_count: int              = 0
    fps_display: float            = 0.0
    t_fps:       float            = field(default_factory=time.perf_counter)
    frames_disp: int              = 0
    t_inicio:    float            = field(default_factory=time.time)
    gestor_geo:  GestorGeofencing = field(default_factory=GestorGeofencing)


def _hilo_inferencia_sesion(
    frame_q:      queue.Queue,
    detector:     Detector,
    gestor:       GestorPresencia,
    gestor_geo:   GestorGeofencing,
    estado:       EstadoCompartido,
    fps_video:    float,
    lector_ocr:   LectorFlota | None,
    clasificador: ClasificadorEstado | None = None,
    clf_lstm:     ClasificadorLSTM | None  = None,
) -> None:
    """
    Corre la inferencia en su propio hilo con una sesión de DB PROPIA.

    SQLAlchemy/SQLite no permite compartir una Session entre hilos: antes se
    pasaba una sesión única a todos los videos, lo que corrompía transacciones
    al procesar 2-3 videos en paralelo.
    """
    with get_session() as session:
        hilo_inferencia(
            frame_q, detector, gestor, gestor_geo,
            RepositorioEventos(session), RepositorioCiclos(session),
            estado, fps_video, lector_ocr, clasificador, clf_lstm,
            session=session,
        )


def _iniciar_pipeline(
    pv:           PipelineVideo,
    detector:     Detector,
    lector_ocr:   LectorFlota | None,
    clasificador: ClasificadorEstado | None = None,
    clf_lstm:     ClasificadorLSTM | None   = None,
) -> None:
    """Lanza los hilos de lector e inferencia para un PipelineVideo."""
    try:
        pv.cap = open_video(pv.video_path)
    except SystemExit:
        print(f"[ERROR] No se pudo abrir video {pv.indice}: {pv.video_path}")
        pv.estado.salir = True
        return

    pv.fps   = pv.cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Cada pipeline tiene su propio GestorPresencia para no mezclar IDs
    gestor = GestorPresencia(frames_ausencia=FRAMES_AUSENCIA)

    frame_q:   queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
    display_q: queue.Queue = queue.Queue(maxsize=1)

    pv.t_lector = threading.Thread(
        target=hilo_lector,
        args=(pv.cap, frame_q, display_q, pv.estado, pv.fps),
        daemon=True, name=f"lector-v{pv.indice}",
    )
    pv.t_infer = threading.Thread(
        target=_hilo_inferencia_sesion,
        args=(frame_q, detector, gestor, pv.gestor_geo,
              pv.estado, pv.fps, lector_ocr, clasificador, clf_lstm),
        daemon=True, name=f"inferencia-v{pv.indice}",
    )
    # Guardar display_q en estado para que el display principal la lea
    pv.estado._display_q = display_q  # type: ignore[attr-defined]

    pv.t_lector.start()
    pv.t_infer.start()
    print(f"[INFO] Video {pv.indice} iniciado: {os.path.basename(pv.video_path)}")


def _tick_display(pv: PipelineVideo, lector_ocr: LectorFlota | None) -> bool:
    """
    Procesa un tick del display para un PipelineVideo.
    Retorna True si el video sigue activo, False si terminó.
    """
    if pv.estado.salir:
        return False

    display_q = getattr(pv.estado, '_display_q', None)
    if display_q is None:
        return False

    # Obtener frame más reciente para display
    frame_display = None
    try:
        item = display_q.get_nowait()
        if item is _STOP:
            return False
        frame_display = item
    except queue.Empty:
        pass

    if frame_display is None:
        if pv.estado.video_terminado and not pv.t_infer.is_alive():
            return False
        # Mostrar último frame cacheado si no hay nuevo
        if pv.frame_vis_cache is not None:
            cv2.imshow(pv.ventana, pv.frame_vis_cache)
        return True

    # Frame nuevo — procesar
    if frame_display is not pv.ultimo_frame_id:
        pv.ultimo_frame_id = frame_display
        frame_vis = frame_display.copy()

        if not pv.estado.pausado:
            detecciones, numeros_flota, estados = pv.estado.leer_dets()
            dibujar_zonas(frame_vis, pv.gestor_geo)
            dibujar_tracking(frame_vis, detecciones, numeros_flota, estados)

        # HUD
        pv.frames_disp += 1
        el_fps = time.perf_counter() - pv.t_fps
        if el_fps >= 1.0:
            pv.fps_display = pv.frames_disp / el_fps
            pv.frames_disp = 0
            pv.t_fps = time.perf_counter()

        ts_disp = calcular_timestamp(pv.fps, pv.frame_count)
        ocr_str = f"OCR:{len(pv.estado.numeros_flota)}" if lector_ocr else ""
        hud = (
            f"V{pv.indice} {ts_disp} "
            f"| act={pv.estado.n_activos} "
            f"| ev={pv.estado.event_count} "
            f"{ocr_str} "
            f"| {pv.fps_display:.0f}fps disp "
            f"| {pv.estado.fps_infer:.0f}fps inf"
        )
        cv2.putText(frame_vis, hud, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_vis, hud, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        if pv.estado.pausado:
            cv2.putText(frame_vis, f"PAUSADO V{pv.indice}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        if lector_ocr and pv.estado.numeros_flota:
            _dibujar_panel_flota(frame_vis, pv.estado.numeros_flota)

        if DISPLAY_WIDTH > 0:
            h_vis, w_vis = frame_vis.shape[:2]
            if w_vis > DISPLAY_WIDTH:
                scale     = DISPLAY_WIDTH / w_vis
                frame_vis = cv2.resize(frame_vis, (DISPLAY_WIDTH, int(h_vis * scale)))

        pv.frame_vis_cache = frame_vis
        pv.frame_count += 1

    if pv.frame_vis_cache is not None:
        cv2.imshow(pv.ventana, pv.frame_vis_cache)

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Determinar qué videos analizar ───────────────────────────────────────
    config = load_config()

    if len(sys.argv) > 1:
        # Videos pasados como argumentos: python main.py v1.mp4 v2.mp4 v3.mp4
        rutas_video = sys.argv[1:]
    else:
        # Fallback al video del .env
        rutas_video = [config.video_path]

    MAX_VIDEOS = 3
    rutas_video = rutas_video[:MAX_VIDEOS]

    if not rutas_video:
        print("[ERROR] No hay videos que procesar. Especifica rutas como argumentos.")
        return

    print(f"[INFO] Procesando {len(rutas_video)} video(s):")
    for i, ruta in enumerate(rutas_video, 1):
        print(f"  [{i}] {ruta}")

    migrate()

    with get_session() as session:
        gestor_zonas, _ = _cargar_zonas(session)
        zonas = list(gestor_zonas._zonas.values())
        # Cerrar la transacción de lectura antes de lanzar los hilos de
        # inferencia (cada uno abre su propia conexión a SQLite).
        session.commit()

        # Un solo lector OCR compartido — el modelo se carga una sola vez
        lector_ocr: LectorFlota | None = None
        if OCR_HABILITADO:
            lector_ocr = LectorFlota()
            threading.Thread(
                target=lector_ocr._init_reader,
                daemon=True, name="ocr-init",
            ).start()
            print("[INFO] OCR de flotas: ACTIVADO (cargando en background…)")
        else:
            print("[INFO] OCR de flotas: DESACTIVADO")

        # Clasificador de estado compartido — carga lazy por tipo
        clasificador: ClasificadorEstado | None = None
        if CLASIFICADOR_HABILITADO:
            clasificador = ClasificadorEstado()
            modelos_ok = [t for t, ok in clasificador._disponible.items() if ok]
            if modelos_ok:
                print(f"[INFO] Clasificador CNN (estático): ACTIVO ({modelos_ok})")
            else:
                print("[INFO] Clasificador CNN: sin modelos — ejecuta: python entrenar_clasificador.py")
        else:
            print("[INFO] Clasificador CNN: DESACTIVADO")

        # Clasificador LSTM temporal — se activa si hay modelos en weights/
        clf_lstm: ClasificadorLSTM | None = None
        if CLASIFICADOR_HABILITADO:
            clf_lstm = ClasificadorLSTM(device="cuda" if False else "cpu")
            lstm_ok  = [t for t, ok in clf_lstm._disponible.items() if ok]
            if lstm_ok:
                print(f"[INFO] Clasificador LSTM (temporal): ACTIVO ({lstm_ok})")
            else:
                print("[INFO] Clasificador LSTM: sin modelos — ejecuta: python entrenar_lstm.py")

        # ── Crear y lanzar un pipeline por video ──────────────────────────────
        pipelines: list[PipelineVideo] = []
        for i, ruta in enumerate(rutas_video, 1):
            pv = PipelineVideo(
                video_path = ruta,
                indice     = i,
                ventana    = f"Mining Dispatch — Video {i}: {os.path.basename(ruta)}",
            )
            # Geofencing independiente por video: los estados de máquina no
            # deben mezclarse entre videos.
            for zona in zonas:
                pv.gestor_geo.agregar_zona(zona)

            # Detector independiente por video: track_id y tracker son estado
            # por secuencia. Compartirlos hacía que los IDs colisionaran.
            detector = Detector(config.model_path, config.confidence_threshold)
            detector.load_model()

            _iniciar_pipeline(pv, detector, lector_ocr, clasificador, clf_lstm)
            if not pv.estado.salir:
                pipelines.append(pv)

        if not pipelines:
            print("[ERROR] Ningún video pudo iniciarse.")
            return

        print(f"\n[INFO] {len(pipelines)} pipeline(s) activo(s)")
        print("[INFO] Q/ESC=salir | ESPACIO=pausar todos | 1/2/3=pausar video específico")

        t_inicio = time.time()

        try:
            while True:
                # Tick de display para cada pipeline activo
                activos = [pv for pv in pipelines if not pv.estado.salir]
                if not activos:
                    break

                for pv in activos:
                    sigue = _tick_display(pv, lector_ocr)
                    if not sigue:
                        pv.estado.salir = True
                        cv2.destroyWindow(pv.ventana)
                        print(f"[INFO] Video {pv.indice} finalizado")

                # Teclado — afecta a todos o al video específico
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    print("[INFO] Salida por usuario.")
                    for pv in pipelines:
                        pv.estado.salir = True
                    break

                elif key == ord(" "):
                    # Pausar/reanudar TODOS
                    nuevo = not all(pv.estado.pausado for pv in activos)
                    for pv in activos:
                        pv.estado.pausado = nuevo
                    print(f"[INFO] {'PAUSADO' if nuevo else 'REANUDADO'} todos")

                elif key in (ord("1"), ord("2"), ord("3")):
                    idx = int(chr(key)) - 1
                    if idx < len(activos):
                        pv = activos[idx]
                        pv.estado.pausado = not pv.estado.pausado
                        estado_str = "PAUSADO" if pv.estado.pausado else "REANUDADO"
                        print(f"[INFO] Video {pv.indice} {estado_str}")

                time.sleep(0.001)   # ceder CPU entre ticks

        except KeyboardInterrupt:
            print("\n[INFO] Interrumpido (Ctrl+C).")
        finally:
            for pv in pipelines:
                pv.estado.salir = True
            for pv in pipelines:
                if pv.t_lector: pv.t_lector.join(timeout=3)
                if pv.t_infer:  pv.t_infer.join(timeout=8)
                if pv.cap:      pv.cap.release()
            cv2.destroyAllWindows()

    duracion = time.time() - t_inicio
    print(f"\n[RESUMEN GLOBAL] duracion={duracion:.1f}s | videos={len(pipelines)}")
    for pv in pipelines:
        flotas = lector_ocr.todos_historicos() if lector_ocr else {}
        print(
            f"  Video {pv.indice} ({os.path.basename(pv.video_path)}): "
            f"frames={pv.frame_count} | "
            f"inferencias={pv.estado.frames_inferidos} | "
            f"eventos={pv.estado.event_count} | "
            f"flotas={len(flotas)}"
        )


def _dibujar_panel_flota(frame: np.ndarray, numeros: dict[int, str]) -> None:
    """
    Dibuja un panel semitransparente en la esquina superior derecha
    con la lista de numeros de flota detectados.
    """
    if not numeros:
        return

    h, w = frame.shape[:2]
    lineas = ["FLOTA DETECTADA"] + [f"  ID:{tid} -> #{num}" for tid, num in numeros.items()]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = 0.38
    pad    = 6
    lh     = 16

    panel_h = pad * 2 + len(lineas) * lh
    panel_w = 180
    px1     = w - panel_w - 8
    py1     = 8
    px2     = w - 8
    py2     = py1 + panel_h

    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 255), 1)

    for i, linea in enumerate(lineas):
        color = (0, 200, 255) if i == 0 else (255, 255, 255)
        cv2.putText(frame, linea,
                    (px1 + pad, py1 + pad + lh * (i + 1) - 3),
                    font, fs, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
