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
from src.database.database import RepositorioEventos, get_session_factory, init_db
from src.ai.detector import CLASE_MAP, Detector

# ── Rendimiento (ajusta segun tu CPU) ────────────────────────────────────────
# INFERENCE_IMGSZ  256px / INFERENCE_SKIP 4  ->  ~12-18 FPS (recomendado i5)
# INFERENCE_IMGSZ  160px / INFERENCE_SKIP 5  ->  ~20-30 FPS (basico)
INFERENCE_IMGSZ = 256
INFERENCE_SKIP  = 4

COLORES: dict[str, tuple[int, int, int]] = {
    "camion": (0, 200, 255),
    "pala":   (0, 255, 80),
}
COLOR_DEFAULT = (200, 200, 200)


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


def dibujar_tracking(frame, detecciones: list[dict]) -> None:
    """Dibuja bounding boxes con ID de tracking (in-place)."""
    for det in detecciones:
        tipo  = det["tipo_maquina"]
        color = COLORES.get(tipo, COLOR_DEFAULT)
        x1, y1, x2, y2 = map(int, det["bbox"])
        tid   = det.get("track_id", "?")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {tipo} {det['confianza']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main() -> None:
    config = load_config()

    init_db(config.db_path)
    SessionFactory = get_session_factory(config.db_path)
    session        = SessionFactory()
    repo           = RepositorioEventos(session)

    detector = Detector(config.model_path, config.confidence_threshold)
    detector.load_model()

    source = get_camera_source(config)
    cap    = open_camera(source)

    print("[INFO] Tracking en tiempo real activo (ByteTrack)")
    print("[INFO] Q/ESC = salir | ESPACIO = pausar")

    frame_count = 0
    event_count = 0
    pausado     = False
    t_inicio    = time.time()
    ultimas_detecciones: list[dict] = []

    try:
        while True:
            if not pausado:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Frame perdido -- reintentando...")
                    time.sleep(0.1)
                    continue

                frame_vis = frame.copy()

                try:
                    # Inferencia + tracking cada INFERENCE_SKIP frames
                    if frame_count % INFERENCE_SKIP == 0:
                        ts = time.strftime("%H:%M:%S")
                        ultimas_detecciones = detector.track_frame(
                            frame, frame_count, ts, imgsz=INFERENCE_IMGSZ
                        )
                        # Guardar en DB cada INFERENCE_SKIP * 3 frames
                        if frame_count % (INFERENCE_SKIP * 3) == 0:
                            for det in ultimas_detecciones:
                                repo.insertar({
                                    "tipo_maquina":     det["tipo_maquina"],
                                    "clase_id":         det["clase_id"],
                                    "confianza":        det["confianza"],
                                    "numero_fotograma": frame_count,
                                    "timestamp":        det["timestamp"],
                                })
                                print(
                                    f"[EVENTO] frame={frame_count} "
                                    f"| id={det['track_id']} "
                                    f"| maquina={det['tipo_maquina']} "
                                    f"| conf={det['confianza']:.2f}"
                                )
                                event_count += 1

                    # Siempre dibujar las ultimas detecciones conocidas
                    dibujar_tracking(frame_vis, ultimas_detecciones)

                except Exception as e:
                    print(f"[ERROR] frame={frame_count} | {e}")

                # HUD
                elapsed  = time.time() - t_inicio
                fps_real = frame_count / elapsed if elapsed > 0 else 0
                hud = (f"CAMARA EN VIVO | frame {frame_count} "
                       f"| eventos: {event_count} | {fps_real:.1f} fps")
                cv2.putText(frame_vis, hud, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame_vis, hud, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

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
        cap.release()
        cv2.destroyAllWindows()
        session.close()

    duracion = time.time() - t_inicio
    print(
        f"[RESUMEN] frames={frame_count} "
        f"| eventos={event_count} "
        f"| duracion={duracion:.0f}s"
    )


if __name__ == "__main__":
    main()
