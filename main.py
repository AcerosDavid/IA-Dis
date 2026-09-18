"""
Entry point CLI del sistema de deteccion de maquinaria minera.
Ejecutar: python main.py

Pipeline: config -> DB -> modelo -> video -> tracking -> eventos de presencia -> resumen

Controles:
    Q / ESC  -- salir
    ESPACIO  -- pausar / reanudar
"""
from __future__ import annotations

import os
import time

import cv2

from src.config import load_config
from src.database.db import RepositorioEventos, get_session, migrate
from src.ai.detector import Detector

COLORES: dict[str, tuple[int, int, int]] = {
    "camion": (0, 200, 255),   # naranja
    "pala":   (0, 255, 80),    # verde
}
COLOR_DEFAULT = (200, 200, 200)

# Frames consecutivos sin detectar un ID antes de considerarlo "desaparecido"
FRAMES_AUSENCIA = 30

# ── Rendimiento ───────────────────────────────────────────────────────────────
# Ajusta segun tu CPU (i5-7200U sin GPU):
#   TRACK_IMGSZ=640, TRACK_SKIP=2  ->  ~3-5 FPS   (maxima calidad, objetos pequenos)
#   TRACK_IMGSZ=480, TRACK_SKIP=2  ->  ~5-8 FPS   (buena calidad, recomendado)
#   TRACK_IMGSZ=320, TRACK_SKIP=2  ->  ~8-12 FPS  (calidad media)
#   TRACK_IMGSZ=160, TRACK_SKIP=2  ->  ~12-15 FPS (basico, pierde objetos pequenos)
TRACK_IMGSZ   = 480   # tamano de imagen para inferencia YOLO (px)
TRACK_SKIP    = 2     # ejecutar tracking 1 de cada N frames
DISPLAY_WIDTH = 960   # ancho maximo de la ventana en pixeles (0 = sin redimensionar)


class GestorPresencia:
    """
    Registra solo eventos de aparicion y desaparicion de cada maquina.
    Elimina el ruido de registrar el mismo ID en cada frame.
    """

    def __init__(self, frames_ausencia: int = FRAMES_AUSENCIA) -> None:
        self.frames_ausencia = frames_ausencia
        self._activos: dict[int, dict] = {}
        self._cache_detecciones: list[dict] = []

    def procesar(
        self,
        detecciones: list[dict],
        frame_count: int,
        repo: RepositorioEventos,
    ) -> list[str]:
        """
        Procesa las detecciones del frame actual.
        Retorna lista de mensajes de consola para los eventos generados.
        """
        self._cache_detecciones = detecciones
        mensajes: list[str] = []
        ids_vistos: set[int] = set()

        for det in detecciones:
            tid  = det["track_id"]
            tipo = det["tipo_maquina"]
            ids_vistos.add(tid)

            if tid not in self._activos:
                self._activos[tid] = {
                    "tipo_maquina":        tipo,
                    "clase_id":            det["clase_id"],
                    "frame_aparicion":     frame_count,
                    "timestamp_aparicion": det["timestamp"],
                    "ultimo_frame_visto":  frame_count,
                    "ultimo_timestamp":    det["timestamp"],
                    "confianza_max":       det["confianza"],
                }
                repo.insertar({
                    "tipo_maquina":      tipo,
                    "clase_id":          det["clase_id"],
                    "confianza":         det["confianza"],
                    "numero_fotograma":  frame_count,
                    "timestamp":         det["timestamp"],
                    "tipo_evento_ciclo": "aparicion",
                })
                mensajes.append(
                    f"[APARICION]    id={tid} | maquina={tipo} | clase={det['clase_id']} "
                    f"| frame={frame_count} | ts={det['timestamp']} | conf={det['confianza']:.2f}"
                )
            else:
                estado = self._activos[tid]
                estado["ultimo_frame_visto"] = frame_count
                estado["ultimo_timestamp"]   = det["timestamp"]
                if det["confianza"] > estado["confianza_max"]:
                    estado["confianza_max"] = det["confianza"]

        # Verificar desapariciones
        desaparecidos = [
            tid for tid, est in self._activos.items()
            if tid not in ids_vistos
            and (frame_count - est["ultimo_frame_visto"]) >= self.frames_ausencia
        ]

        for tid in desaparecidos:
            estado = self._activos.pop(tid)
            frames_presencia = estado["ultimo_frame_visto"] - estado["frame_aparicion"]
            repo.insertar({
                "tipo_maquina":      estado["tipo_maquina"],
                "clase_id":          estado["clase_id"],
                "confianza":         estado["confianza_max"],
                "numero_fotograma":  estado["ultimo_frame_visto"],
                "timestamp":         estado["ultimo_timestamp"],   # timestamp del ultimo frame visto
                "tipo_evento_ciclo": "desaparicion",
            })
            mensajes.append(
                f"[DESAPARICION] id={tid} | maquina={estado['tipo_maquina']} "
                f"| frames_en_escena={frames_presencia} "
                f"| aparecio={estado['timestamp_aparicion']}"
            )

        return mensajes

    def activos_para_dibujar(self) -> list[dict]:
        """Retorna solo las detecciones de IDs actualmente activos."""
        ids_activos = set(self._activos.keys())
        return [d for d in self._cache_detecciones if d["track_id"] in ids_activos]

    @property
    def n_activos(self) -> int:
        """Numero de maquinas actualmente en escena."""
        return len(self._activos)

    def cerrar(self, repo: RepositorioEventos) -> None:
        """Registra desaparicion de todos los IDs activos al finalizar el video."""
        for tid, estado in self._activos.items():
            frames_presencia = estado["ultimo_frame_visto"] - estado["frame_aparicion"]
            repo.insertar({
                "tipo_maquina":      estado["tipo_maquina"],
                "clase_id":          estado["clase_id"],
                "confianza":         estado["confianza_max"],
                "numero_fotograma":  estado["ultimo_frame_visto"],
                "timestamp":         estado["ultimo_timestamp"],   # timestamp correcto del ultimo frame
                "tipo_evento_ciclo": "desaparicion",
            })
            print(
                f"[DESAPARICION] id={tid} | maquina={estado['tipo_maquina']} "
                f"| frames_en_escena={frames_presencia}"
            )
        self._activos.clear()


def open_video(video_path: str) -> cv2.VideoCapture:
    if not os.path.exists(video_path):
        print(f"[VIDEO ERROR] Archivo no encontrado: {video_path}")
        raise SystemExit(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VIDEO ERROR] OpenCV no puede decodificar: {video_path}")
        raise SystemExit(1)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video cargado: {w}x{h} @ {fps:.1f}fps | {total} fotogramas")
    return cap


def calcular_timestamp(fps: float, frame_number: int) -> str:
    segundos = frame_number / fps if fps > 0 else 0.0
    horas    = int(segundos // 3600)
    minutos  = int((segundos % 3600) // 60)
    segs     = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:06.3f}"


def print_evento(det: dict) -> None:
    """Imprime un evento de deteccion con formato estandar. Usado en tests."""
    tid = det.get("track_id", "?")
    print(
        f"[EVENTO] fotograma={det['numero_fotograma']} "
        f"| timestamp={det['timestamp']} "
        f"| id={tid} "
        f"| maquina={det['tipo_maquina']} "
        f"| clase={det['clase_id']} "
        f"| confianza={det['confianza']:.2f}"
    )


def dibujar_tracking(frame, detecciones: list[dict]) -> None:
    """Dibuja bounding boxes con ID de tracking sobre el frame (in-place).
    Texto con sombra: primera pasada blanca (grosor 2), segunda negra (grosor 1).
    """
    for det in detecciones:
        tipo  = det["tipo_maquina"]
        color = COLORES.get(tipo, COLOR_DEFAULT)
        x1, y1, x2, y2 = map(int, det["bbox"])
        tid   = det.get("track_id", "?")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {tipo} | {det['clase_id']} {det['confianza']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main() -> None:
    config = load_config()

    migrate()
    with get_session() as session:
        repo    = RepositorioEventos(session)
        gestor  = GestorPresencia(frames_ausencia=FRAMES_AUSENCIA)

        detector = Detector(config.model_path, config.confidence_threshold)
        detector.load_model()

        cap   = open_video(config.video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        delay = max(1, int(1000 / fps) if fps > 0 else 30)

        t_inicio    = time.time()
        frame_count = 0
        event_count = 0
        pausado     = False

        print("[INFO] Tracking activo (ByteTrack) -- Q=salir | ESPACIO=pausar")
        print(f"[INFO] Modo rendimiento: imgsz={TRACK_IMGSZ}px | skip={TRACK_SKIP} frames")

        try:
            while True:
                if not pausado:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp = calcular_timestamp(fps, frame_count)
                    frame_vis = frame.copy()

                    try:
                        if frame_count % TRACK_SKIP == 0:
                            detecciones = detector.track_frame(
                                frame, frame_count, timestamp, imgsz=TRACK_IMGSZ
                            )
                            mensajes = gestor.procesar(detecciones, frame_count, repo)
                            for msg in mensajes:
                                print(msg)
                                event_count += 1

                        dibujar_tracking(frame_vis, gestor.activos_para_dibujar())

                    except Exception as e:
                        print(f"[ERROR] frame={frame_count} | {e}")

                    # HUD — texto con sombra (blanco grueso + negro fino)
                    elapsed  = time.time() - t_inicio
                    fps_real = frame_count / elapsed if elapsed > 0 else 0
                    hud = (
                        f"{timestamp} | frame {frame_count} "
                        f"| activos: {gestor.n_activos} | eventos: {event_count} "
                        f"| {fps_real:.1f} FPS"
                    )
                    cv2.putText(frame_vis, hud, (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_vis, hud, (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                    if DISPLAY_WIDTH > 0:
                        h_vis, w_vis = frame_vis.shape[:2]
                        if w_vis > DISPLAY_WIDTH:
                            scale     = DISPLAY_WIDTH / w_vis
                            frame_vis = cv2.resize(
                                frame_vis, (DISPLAY_WIDTH, int(h_vis * scale))
                            )

                    cv2.imshow("Mining Dispatch -- Tracking YOLO26  [Q=salir  SPACE=pausar]", frame_vis)
                    frame_count += 1

                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), 27):
                    print("[INFO] Salida por usuario.")
                    break
                elif key == ord(" "):
                    pausado = not pausado
                    print(f"[INFO] {'PAUSADO' if pausado else 'REANUDADO'}")

        finally:
            gestor.cerrar(repo)
            cap.release()
            cv2.destroyAllWindows()

    duracion = time.time() - t_inicio
    print(
        f"[RESUMEN] fotogramas_procesados={frame_count} "
        f"| eventos_registrados={event_count} "
        f"| duracion_segundos={duracion:.2f}"
    )


if __name__ == "__main__":
    main()
