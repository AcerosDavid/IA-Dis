"""
Entry point CLI del sistema de deteccion de maquinaria minera.
Ejecutar: python main.py

Pipeline multihilo (3 hilos + sub-hilo OCR):
  Hilo LECTOR     -> lee frames a velocidad del video (~30fps)
  Hilo INFERENCIA -> YOLO + BoT-SORT + filtro estaticos (~5fps)
                     lanza sub-hilo OCR para leer numeros de flota
  Sub-hilo OCR    -> EasyOCR en background (no bloquea YOLO)
  Hilo DISPLAY    -> video fluido (30fps) + bboxes + numero de flota

Controles:
    Q / ESC  -- salir
    ESPACIO  -- pausar / reanudar
"""
from __future__ import annotations

import os
import queue
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
from src.core.geometry import BBox, GestorGeofencing, Zona, dibujar_zonas

COLORES: dict[str, tuple[int, int, int]] = {
    "camion":     (0, 200, 255),
    "excavadora": (0, 255, 80),
    "pala":       (255, 100, 0),
}
COLOR_DEFAULT = (200, 200, 200)

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
    n_activos:        int                 = 0
    event_count:      int                 = 0
    ciclo_count:      int                 = 0
    fps_infer:        float               = 0.0
    frames_inferidos: int                 = 0
    pausado:          bool                = False
    salir:            bool                = False
    _lock:            threading.Lock      = field(default_factory=threading.Lock)

    def actualizar_dets(
        self,
        dets:    list[dict],
        n:       int,
        ev:      int,
        ci:      int,
        numeros: dict[int, str],
    ) -> None:
        with self._lock:
            self.detecciones   = dets
            self.n_activos     = n
            self.event_count   = ev
            self.ciclo_count   = ci
            self.numeros_flota = numeros
            self.frames_inferidos += 1

    def leer_dets(self) -> tuple[list[dict], dict[int, str]]:
        with self._lock:
            return list(self.detecciones), dict(self.numeros_flota)


# ── GestorPresencia ───────────────────────────────────────────────────────────

class GestorPresencia:
    def __init__(self, frames_ausencia: int = FRAMES_AUSENCIA) -> None:
        self.frames_ausencia = frames_ausencia
        self._activos: dict[int, dict] = {}
        self._cache_detecciones: list[dict] = []

    def procesar(
        self,
        detecciones: list[dict],
        frame_count: int,
        repo: RepositorioEventos,
        gestor_geo: GestorGeofencing | None = None,
        repo_ciclos: RepositorioCiclos | None = None,
    ) -> list[str]:
        self._cache_detecciones = detecciones
        mensajes: list[str] = []
        ids_vistos: set[int] = set()

        for det in detecciones:
            tid  = det["track_id"]
            tipo = det["tipo_maquina"]
            ids_vistos.add(tid)

            zona_actual: str | None = None
            if gestor_geo is not None:
                bbox = BBox.desde_yolo(det["bbox"])
                eventos_ciclo = gestor_geo.procesar(
                    maquina_id=det["maquina_id"], tipo_maquina=tipo,
                    bbox=bbox, timestamp=det["timestamp"], frame_number=frame_count,
                )
                for ec in eventos_ciclo:
                    zona_actual = ec.zona_nombre
                    mensajes.append(f"[ZONA] id={tid} | {ec.tipo_evento} -> {ec.zona_nombre}")
                    if repo_ciclos is not None:
                        try:
                            repo_ciclos.registrar_evento(ec)
                        except Exception as e:
                            print(f"[DB ERROR] ciclo: {e}")
                estado_geo = gestor_geo._estados.get(det["maquina_id"])
                if estado_geo:
                    zona_actual = estado_geo.zona_actual

            if tid not in self._activos:
                self._activos[tid] = {
                    "maquina_id":          det["maquina_id"],
                    "tipo_maquina":        tipo,
                    "clase_id":            det["clase_id"],
                    "frame_aparicion":     frame_count,
                    "timestamp_aparicion": det["timestamp"],
                    "ultimo_frame_visto":  frame_count,
                    "ultimo_timestamp":    det["timestamp"],
                    "confianza_max":       det["confianza"],
                }
                repo.insertar({
                    "tipo_maquina": tipo, "clase_id": det["clase_id"],
                    "confianza": det["confianza"], "numero_fotograma": frame_count,
                    "timestamp": det["timestamp"], "tipo_evento_ciclo": "aparicion",
                    "zona_nombre": zona_actual,
                })
                mensajes.append(
                    f"[APARICION]    id={tid} | maquina={tipo} | clase={det['clase_id']} "
                    f"| frame={frame_count} | ts={det['timestamp']} | conf={det['confianza']:.2f}"
                )
            else:
                est = self._activos[tid]
                est["ultimo_frame_visto"] = frame_count
                est["ultimo_timestamp"]   = det["timestamp"]
                if det["confianza"] > est["confianza_max"]:
                    est["confianza_max"] = det["confianza"]

        desaparecidos = [
            tid for tid, est in self._activos.items()
            if tid not in ids_vistos
            and (frame_count - est["ultimo_frame_visto"]) >= self.frames_ausencia
        ]
        for tid in desaparecidos:
            est = self._activos.pop(tid)
            fp  = est["ultimo_frame_visto"] - est["frame_aparicion"]
            repo.insertar({
                "tipo_maquina": est["tipo_maquina"], "clase_id": est["clase_id"],
                "confianza": est["confianza_max"], "numero_fotograma": est["ultimo_frame_visto"],
                "timestamp": est["ultimo_timestamp"], "tipo_evento_ciclo": "desaparicion",
            })
            mensajes.append(
                f"[DESAPARICION] id={tid} | maquina={est['tipo_maquina']} | frames={fp}"
            )
            if gestor_geo is not None:
                gestor_geo.limpiar_maquina(est["maquina_id"])

        return mensajes

    def activos_para_dibujar(self) -> list[dict]:
        ids_activos = set(self._activos.keys())
        return [d for d in self._cache_detecciones if d["track_id"] in ids_activos]

    @property
    def n_activos(self) -> int:
        return len(self._activos)

    def cerrar(self, repo: RepositorioEventos, gestor_geo: GestorGeofencing | None = None) -> None:
        for tid, est in self._activos.items():
            fp = est["ultimo_frame_visto"] - est["frame_aparicion"]
            repo.insertar({
                "tipo_maquina": est["tipo_maquina"], "clase_id": est["clase_id"],
                "confianza": est["confianza_max"], "numero_fotograma": est["ultimo_frame_visto"],
                "timestamp": est["ultimo_timestamp"], "tipo_evento_ciclo": "desaparicion",
            })
            print(f"[DESAPARICION] id={tid} | maquina={est['tipo_maquina']} | frames={fp}")
            if gestor_geo is not None:
                gestor_geo.limpiar_maquina(est["maquina_id"])
        self._activos.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def calcular_timestamp(fps: float, frame_number: int) -> str:
    segundos = frame_number / fps if fps > 0 else 0.0
    horas    = int(segundos // 3600)
    minutos  = int((segundos % 3600) // 60)
    segs     = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:06.3f}"


def print_evento(det: dict) -> None:
    tid = det.get("track_id", "?")
    print(
        f"[EVENTO] fotograma={det['numero_fotograma']} | timestamp={det['timestamp']} "
        f"| id={tid} | maquina={det['tipo_maquina']} | clase={det['clase_id']} "
        f"| confianza={det['confianza']:.2f}"
    )


def dibujar_tracking(
    frame:        np.ndarray,
    detecciones:  list[dict],
    numeros_flota: dict[int, str],
) -> None:
    """
    Dibuja bboxes estilo SmartPSS con esquinas marcadas.
    Si el numero de flota esta disponible, lo muestra en la etiqueta.
    """
    for det in detecciones:
        tipo  = det["tipo_maquina"]
        color = COLORES.get(tipo, COLOR_DEFAULT)
        x1, y1, x2, y2 = map(int, det["bbox"])
        tid   = det.get("track_id", "?")
        w_box = x2 - x1
        h_box = y2 - y1

        # Bbox fino
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Esquinas gruesas estilo SmartPSS
        cs = min(w_box, h_box, 20)
        for (ox, oy, dx, dy) in [
            (x1, y1,  cs, 0), (x1, y1, 0,  cs),
            (x2, y1, -cs, 0), (x2, y1, 0,  cs),
            (x1, y2,  cs, 0), (x1, y2, 0, -cs),
            (x2, y2, -cs, 0), (x2, y2, 0, -cs),
        ]:
            cv2.line(frame, (ox, oy), (ox + dx, oy + dy), color, 2)

        # Etiqueta principal
        numero = numeros_flota.get(tid)
        if numero:
            # Numero de flota conocido — mostrarlo prominentemente
            label_top  = f"#{numero}"
            label_bot  = f"ID:{tid} {tipo} {det['confianza']:.2f}"
        else:
            label_top  = None
            label_bot  = f"ID:{tid} {tipo} {det['confianza']:.2f}"

        # Etiqueta inferior (siempre)
        (tw, th), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label_bot, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        # Numero de flota en el centro del bbox (grande y visible)
        if label_top:
            (nw, nh), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            nx = x1 + (w_box - nw) // 2
            ny = y1 + (h_box + nh) // 2
            # Sombra negra
            cv2.putText(frame, label_top, (nx + 1, ny + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
            # Texto en color de tipo
            cv2.putText(frame, label_top, (nx, ny),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


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
            if ZONAS_DEFAULT:
                repo_zonas.guardar(zona)
        n = len(ZONAS_DEFAULT)
        print(f"[INFO] Geofencing: {'desactivado' if n == 0 else f'{n} zonas cargadas'}")
        return gestor_geo, n > 0


# ── Hilos ─────────────────────────────────────────────────────────────────────

def hilo_lector(
    cap:       cv2.VideoCapture,
    frame_q:   queue.Queue,
    estado:    EstadoCompartido,
    fps_video: float,
) -> None:
    """
    Lee frames a velocidad del video y los encola para el inferidor.

    Para VIDEO GRABADO (maxsize=4):
      Llena la queue hasta su capacidad — el tracker recibe frames
      consecutivos con pocos saltos, los IDs son estables.
      Si la queue esta llena, espera (bloquea) en lugar de descartar.

    Para CAMARA EN VIVO (maxsize=1):
      Descarta frames viejos — minimiza latencia.
      Cambiar QUEUE_SIZE=1 en la configuracion.
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

            if QUEUE_SIZE == 1:
                # Modo camara en vivo: descartar frame viejo si queue llena
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
                # Modo video grabado: bloquear hasta que haya espacio
                # Esto garantiza que el tracker no se salte frames
                try:
                    frame_q.put((frame_idx, frame), timeout=2.0)
                except queue.Full:
                    pass   # si timeout, continuar (estado.salir puede haber cambiado)

            frame_idx += 1

            elapsed = time.perf_counter() - t0
            sleep   = delay_read - elapsed
            if sleep > 0:
                time.sleep(sleep)
    finally:
        frame_q.put(_STOP)
        print(f"[LECTOR] Terminado. Frames leidos: {frame_idx}")


def hilo_inferencia(
    frame_q:    queue.Queue,
    detector:   Detector,
    gestor:     GestorPresencia,
    gestor_geo: GestorGeofencing,
    repo:       RepositorioEventos,
    repo_ciclos: RepositorioCiclos,
    estado:     EstadoCompartido,
    fps_video:  float,
    lector_ocr: LectorFlota | None,
) -> None:
    """
    Consume frames, corre YOLO + BoT-SORT + filtro estaticos.
    Lanza OCR en sub-hilo para no bloquear la inferencia.
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
                        # Limpiar cache OCR cuando un ID desaparece
                        if "[DESAPARICION]" in msg and lector_ocr is not None:
                            try:
                                tid_str = msg.split("id=")[1].split(" ")[0]
                                lector_ocr.limpiar_track(int(tid_str))
                            except Exception:
                                pass
                    elif "[ZONA]" in msg:
                        ciclo_count += 1

                activos = gestor.activos_para_dibujar()

                # Encolar para OCR (no bloqueante — si la cola esta llena, saltar)
                if lector_ocr is not None and activos:
                    try:
                        ocr_queue.put_nowait((frame.copy(), activos, frame_idx))
                    except queue.Full:
                        pass   # OCR ocupado — no bloquear YOLO

                # Publicar al display
                numeros = lector_ocr.todos_conocidos() if lector_ocr else {}
                estado.actualizar_dets(
                    activos, gestor.n_activos,
                    ev_count, ciclo_count,
                    numeros,
                )

            except Exception as e:
                print(f"[INFERENCIA] frame={frame_idx} error: {e}")

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    migrate()

    with get_session() as session:
        repo        = RepositorioEventos(session)
        repo_ciclos = RepositorioCiclos(session)
        gestor      = GestorPresencia(frames_ausencia=FRAMES_AUSENCIA)
        gestor_geo, zonas_ok = _cargar_zonas(session)

        detector = Detector(config.model_path, config.confidence_threshold)
        detector.load_model()

        # OCR de flota — pre-inicializar en background para que este listo
        # cuando aparezca el primer camion (evita el retraso de 21s)
        lector_ocr: LectorFlota | None = None
        if OCR_HABILITADO:
            lector_ocr = LectorFlota()
            # Lanzar init en background para no bloquear el arranque
            def _preload_ocr():
                lector_ocr._init_reader()
            threading.Thread(target=_preload_ocr, daemon=True, name="ocr-init").start()
            print("[INFO] OCR de numeros de flota: ACTIVADO (cargando en background…)")
        else:
            print("[INFO] OCR de numeros de flota: DESACTIVADO")

        cap      = open_video(config.video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        estado   = EstadoCompartido()
        frame_q: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)

        print("[INFO] Pipeline multihilo activo")
        print(f"[INFO] imgsz={TRACK_IMGSZ} | Tracker: BoT-SORT (botsort_mining.yaml)")
        print("[INFO] Q=salir | ESPACIO=pausar")

        # ── Lanzar hilos ──────────────────────────────────────────────────────
        t_lector = threading.Thread(
            target=hilo_lector,
            args=(cap, frame_q, estado, fps),
            daemon=True, name="lector",
        )
        t_infer = threading.Thread(
            target=hilo_inferencia,
            args=(frame_q, detector, gestor, gestor_geo,
                  repo, repo_ciclos, estado, fps, lector_ocr),
            daemon=True, name="inferencia",
        )
        t_lector.start()
        t_infer.start()

        # ── Hilo principal = Display ──────────────────────────────────────────
        t_inicio    = time.time()
        frame_count = 0
        fps_display = 0.0
        t_fps       = time.perf_counter()
        frames_disp = 0
        ultimo_frame: np.ndarray | None = None

        try:
            while not estado.salir:

                # Intentar obtener frame mas reciente del lector
                try:
                    item = frame_q.get_nowait()
                    if item is _STOP:
                        break
                    _, ultimo_frame = item
                except queue.Empty:
                    pass

                if ultimo_frame is None:
                    if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                        estado.salir = True
                        break
                    continue

                frame_vis = ultimo_frame.copy()

                if not estado.pausado:
                    detecciones, numeros_flota = estado.leer_dets()
                    dibujar_zonas(frame_vis, gestor_geo)
                    dibujar_tracking(frame_vis, detecciones, numeros_flota)

                # HUD
                elapsed = time.time() - t_inicio
                frames_disp += 1
                el_fps = time.perf_counter() - t_fps
                if el_fps >= 1.0:
                    fps_display = frames_disp / el_fps
                    frames_disp = 0
                    t_fps = time.perf_counter()

                ts_disp = calcular_timestamp(fps, frame_count)
                ocr_str = f"| OCR:{len(estado.numeros_flota)} flotas" if lector_ocr else ""
                hud = (
                    f"{ts_disp} | f={frame_count} "
                    f"| activos={estado.n_activos} "
                    f"| ev={estado.event_count} "
                    f"{ocr_str} "
                    f"| disp={fps_display:.1f}fps "
                    f"| infer={estado.fps_infer:.1f}fps"
                )
                cv2.putText(frame_vis, hud, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame_vis, hud, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                if estado.pausado:
                    cv2.putText(frame_vis, "PAUSADO", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

                # Panel lateral de numeros de flota conocidos
                if lector_ocr and estado.numeros_flota:
                    _dibujar_panel_flota(frame_vis, estado.numeros_flota)

                if DISPLAY_WIDTH > 0:
                    h_vis, w_vis = frame_vis.shape[:2]
                    if w_vis > DISPLAY_WIDTH:
                        scale     = DISPLAY_WIDTH / w_vis
                        frame_vis = cv2.resize(
                            frame_vis, (DISPLAY_WIDTH, int(h_vis * scale))
                        )

                cv2.imshow("Mining Dispatch  [Q=salir  SPACE=pausar]", frame_vis)
                frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[INFO] Salida por usuario.")
                    estado.salir = True
                    break
                elif key == ord(" "):
                    estado.pausado = not estado.pausado
                    print(f"[INFO] {'PAUSADO' if estado.pausado else 'REANUDADO'}")

                time.sleep(max(0, 1 / fps - 0.001))

        except KeyboardInterrupt:
            print("\n[INFO] Interrumpido (Ctrl+C).")
        finally:
            estado.salir = True
            t_lector.join(timeout=3)
            t_infer.join(timeout=8)
            gestor.cerrar(repo, gestor_geo)
            cap.release()
            cv2.destroyAllWindows()

    duracion = time.time() - t_inicio
    print(
        f"[RESUMEN] frames_display={frame_count} "
        f"| inferencias={estado.frames_inferidos} "
        f"| eventos={estado.event_count} "
        f"| ciclos={estado.ciclo_count} "
        f"| flotas_leidas={len(estado.numeros_flota)} "
        f"| duracion={duracion:.2f}s"
    )
    if estado.numeros_flota:
        print("[FLOTAS DETECTADAS]")
        for tid, num in estado.numeros_flota.items():
            print(f"  ID:{tid} -> Flota #{num}")


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
