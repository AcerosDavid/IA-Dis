"""
Deteccion y tracking en tiempo real desde camara IP (DroidCam u otra fuente).
Ejecutar: python camera.py

Configura CAMERA_SOURCE en .env:
    DroidCam:    http://192.168.1.X:4747/video
    Webcam local: 0
    RTSP:        rtsp://usuario:clave@ip/stream

Controles: Q/ESC = salir | ESPACIO = pausar
"""
from __future__ import annotations

import time

import cv2

from src.config import load_config
from src.database.db import RepositorioEventos, get_session, migrate
from src.ai.detector import Detector
from main import (
    GestorPresencia,
    COLORES,
    COLOR_DEFAULT,
    FRAMES_AUSENCIA,
    dibujar_tracking,
    calcular_timestamp,
)

# ── Rendimiento (ajusta segun tu CPU) ────────────────────────────────────────
# INFERENCE_IMGSZ  480px / INFERENCE_SKIP 2  ->  ~5-8 FPS   (recomendado)
# INFERENCE_IMGSZ  320px / INFERENCE_SKIP 3  ->  ~10-15 FPS (buena calidad)
# INFERENCE_IMGSZ  160px / INFERENCE_SKIP 4  ->  ~20-30 FPS (basico)
INFERENCE_IMGSZ = 480
INFERENCE_SKIP  = 2

DISPLAY_WIDTH = 960


def get_camera_source(config) -> str | int:
    source = getattr(config, "camera_source", None)
    if source is None:
        print("[WARN] CAMERA_SOURCE no definido -- usando webcam local (0)")
        return 0
    if str(source).strip().isdigit():
        return int(source)
    return str(source)


def open_camera(source: str | int) -> cv2.VideoCapture:
    print(f"[INFO] Conectando a: {source}")
    cap = cv2.VideoCapture(source)
    if isinstance(source, str):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(2)
    if not cap.isOpened():
        print(f"[CAMERA ERROR] No se pudo conectar a: {source}")
        raise SystemExit(1)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Camara conectada: {w}x{h} @ {fps:.0f}fps")
    return cap


def main() -> None:
    config = load_config()

    migrate()
    with get_session() as session:
        repo   = RepositorioEventos(session)
        gestor = GestorPresencia(frames_ausencia=FRAMES_AUSENCIA)

        detector = Detector(config.model_path, config.confidence_threshold)
        detector.load_model()

        source = get_camera_source(config)
        cap    = open_camera(source)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        print("[INFO] Tracking en tiempo real activo (ByteTrack)")
        print(f"[INFO] Modo rendimiento: imgsz={INFERENCE_IMGSZ}px | skip={INFERENCE_SKIP} frames")
        print("[INFO] Q/ESC = salir | ESPACIO = pausar")

        frame_count = 0
        event_count = 0
        pausado     = False
        t_inicio    = time.time()

        try:
            while True:
                if not pausado:
                    ret, frame = cap.read()
                    if not ret:
                        print("[WARN] Frame perdido -- reintentando...")
                        time.sleep(0.1)
                        continue

                    # Usa calcular_timestamp igual que main.py para formato consistente
                    timestamp = calcular_timestamp(fps, frame_count)
                    frame_vis = frame.copy()

                    try:
                        if frame_count % INFERENCE_SKIP == 0:
                            detecciones = detector.track_frame(
                                frame, frame_count, timestamp, imgsz=INFERENCE_IMGSZ
                            )
                            mensajes = gestor.procesar(detecciones, frame_count, repo)
                            for msg in mensajes:
                                print(msg)
                                if "[APARICION]" in msg or "[DESAPARICION]" in msg:
                                    event_count += 1

                        dibujar_tracking(frame_vis, gestor.activos_para_dibujar(), {})

                    except Exception as e:
                        print(f"[ERROR] frame={frame_count} | {e}")

                    # HUD — texto con sombra
                    elapsed  = time.time() - t_inicio
                    fps_real = frame_count / elapsed if elapsed > 0 else 0
                    hud = (
                        f"CAMARA EN VIVO | {timestamp} | frame {frame_count} "
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

                    cv2.imshow("Mining Dispatch -- Camara Tracking  [Q=salir  SPACE=pausar]", frame_vis)
                    frame_count += 1

                key = cv2.waitKey(1) & 0xFF
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
        f"[RESUMEN] frames={frame_count} "
        f"| eventos={event_count} "
        f"| duracion={duracion:.0f}s"
    )


if __name__ == "__main__":
    main()
