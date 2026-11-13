"""
inferencia_lstm.py — Inferencia en tiempo real: YOLO + ByteTrack + LSTM estados.

Pipeline completo:
  ┌──────────┐   ┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │  Frame   │──▶│ YOLO + Track   │──▶│ Ventana desliz.  │──▶│  LSTM → Estado   │
  │ (OpenCV) │   │ (ByteTrack ID) │   │ Features (N frm) │   │  operacional     │
  └──────────┘   └────────────────┘   └──────────────────┘   └──────────────────┘

Estados reconocidos:
  Excavadora / Pala : quieta | escarbando_cargando | descargando
  Camión            : vacio_andando | lleno_andando | quieto_en_cargue_descargue

Uso:
    python inferencia_lstm.py --video  data/mina.mp4
    python inferencia_lstm.py --rtsp   rtsp://192.168.1.100:554/stream
    python inferencia_lstm.py --camara 0
    python inferencia_lstm.py --video  data/mina.mp4 --yolo weights/best.pt --device cuda

Controles en pantalla:
    Q / ESC  → salir
    ESPACIO  → pausar / reanudar
    S        → guardar captura de pantalla
    I        → imprimir estadísticas en consola
    +  / -   → aumentar / reducir velocidad de reproducción
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Paleta visual ─────────────────────────────────────────────────────────────
# BGR — colores neón que contrastan con marrón/gris de la mina

_COLOR_TIPO: dict[str, tuple[int, int, int]] = {
    "camion":     (0,   255,  50),   # verde neón
    "excavadora": (0,   200, 255),   # amarillo neón
    "pala":       (255,  50, 180),   # magenta neón
}
_COLOR_ESTADO: dict[str, tuple[int, int, int]] = {
    "quieta":                      (180, 180, 180),
    "escarbando_cargando":         (0,   220, 255),
    "descargando":                 (0,    50, 255),
    "vacio_andando":               (200, 200, 255),
    "lleno_andando":               (0,   200,  50),
    "quieto_en_cargue_descargue":  (0,   255, 200),
}
_ICONO: dict[str, str] = {
    "quieta":                      "[ ]",
    "escarbando_cargando":         "[/]",
    "descargando":                 "[v]",
    "vacio_andando":               "->o",
    "lleno_andando":               "->■",
    "quieto_en_cargue_descargue":  "[=]",
}
_CLASE_MAP: dict[str, str] = {
    "dump_truck":   "camion",
    "excavator":    "excavadora",
    "wheel_loader": "pala",
    "dump":         "camion",
    "truck":        "camion",
}

IMGSZ       = 320
CONF_YOLO   = 0.45
IOU_YOLO    = 0.50
WIN_W       = 1280   # ancho de display


# ── Pipeline ─────────────────────────────────────────────────────────────────

class PipelineLSTM:
    """
    Combina YOLO + ByteTrack + ClasificadorLSTM para inferencia cuadro a cuadro.
    """

    def __init__(self, yolo_weights: str, device: str = "cpu") -> None:
        from ultralytics import YOLO
        from src.ai.clasificador_lstm import ClasificadorLSTM

        print(f"[INFO] Cargando YOLO: {yolo_weights}")
        self.yolo    = YOLO(yolo_weights)
        self.clf     = ClasificadorLSTM(device=device)
        self._device = device

        # Buffer FPS (media móvil 30 frames)
        self._fps_buf:   deque[float] = deque(maxlen=30)
        self._t_ultimo:  float        = time.perf_counter()
        # Caché de estados para display suave (no parpadea cada frame)
        self._cache_estados: dict[int, tuple[str, float]] = {}

        print("[INFO] Pipeline LSTM listo.\n")

    def procesar(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Detecta, clasifica y dibuja. Retorna el frame anotado."""

        # ── YOLO + ByteTrack ──────────────────────────────────────────────────
        resultados = self.yolo.track(
            frame,
            persist      = True,
            verbose      = False,
            imgsz        = IMGSZ,
            conf         = CONF_YOLO,
            iou          = IOU_YOLO,
            tracker      = "bytetrack.yaml",
            device       = self._device,
            max_det      = 30,
            agnostic_nms = True,
        )

        detecciones: list[dict] = []
        for res in resultados:
            if res.boxes is None or res.boxes.id is None:
                continue
            for i, box in enumerate(res.boxes):
                clase = res.names[int(box.cls)]
                if clase not in _CLASE_MAP:
                    continue
                detecciones.append({
                    "track_id":  int(res.boxes.id[i]),
                    "tipo":      _CLASE_MAP[clase],
                    "confianza": float(box.conf),
                    "bbox":      box.xyxy[0].tolist(),
                })

        # ── LSTM por ID ───────────────────────────────────────────────────────
        ids_vistos: set[int] = set()
        for det in detecciones:
            tid = det["track_id"]
            ids_vistos.add(tid)

            estado, conf = self.clf.procesar(
                frame, det["bbox"], det["tipo"], tid, frame_idx
            )
            if estado:
                self._cache_estados[tid] = (estado, conf)

            det["estado"], det["conf_est"] = self._cache_estados.get(tid, (None, 0.0))

        # Limpiar cache de IDs que ya no están en pantalla
        for tid in [t for t in self._cache_estados if t not in ids_vistos]:
            del self._cache_estados[tid]
            self.clf.limpiar_track(tid)

        # Purgar ventanas inactivas cada 300 frames
        if frame_idx % 300 == 0:
            self.clf.limpiar_inactivos(frame_idx)

        # ── Dibujar ───────────────────────────────────────────────────────────
        return self._dibujar(frame.copy(), detecciones, frame_idx)

    # ── Dibujado ──────────────────────────────────────────────────────────────

    def _dibujar(self, frame: np.ndarray, dets: list[dict], frame_idx: int) -> np.ndarray:
        for det in dets:
            tid    = det["track_id"]
            tipo   = det["tipo"]
            estado = det.get("estado")
            conf_d = det["confianza"]
            conf_e = det.get("conf_est", 0.0)
            x1, y1, x2, y2 = map(int, det["bbox"])
            w_box = x2 - x1
            h_box = y2 - y1

            color = (
                _COLOR_ESTADO.get(estado, _COLOR_TIPO.get(tipo, (255, 255, 255)))
                if estado else _COLOR_TIPO.get(tipo, (255, 255, 255))
            )

            # Bbox con relleno semitransparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Esquinas estilo SmartPSS
            cs = min(w_box, h_box, 20)
            for ox, oy, dx, dy in [
                (x1, y1,  cs, 0), (x1, y1, 0,  cs),
                (x2, y1, -cs, 0), (x2, y1, 0,  cs),
                (x1, y2,  cs, 0), (x1, y2, 0, -cs),
                (x2, y2, -cs, 0), (x2, y2, 0, -cs),
            ]:
                cv2.line(frame, (ox, oy), (ox + dx, oy + dy), color, 3)

            # Etiqueta ID + tipo + confianza detección
            lbl_id = f"ID:{tid} {tipo} {conf_d:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl_id, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ly = max(y1 - 5, th + 5)
            cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, 1)
            cv2.putText(frame, lbl_id, (x1 + 4, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

            # Badge de estado LSTM
            if estado:
                icono    = _ICONO.get(estado, "")
                lbl_est  = f"{icono} {estado.upper().replace('_', ' ')}"
                if conf_e > 0:
                    lbl_est += f" {conf_e:.0%}"
                (ew, eh), _ = cv2.getTextSize(lbl_est, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                ey1 = y2 + 4
                ey2 = ey1 + eh + 10
                cv2.rectangle(frame, (x1, ey1), (x1 + ew + 12, ey2), (0, 0, 0), -1)
                cv2.rectangle(frame, (x1, ey1), (x1 + ew + 12, ey2), color, 2)
                cv2.putText(frame, lbl_est, (x1 + 6, ey2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

                # Barra de progreso de la ventana temporal
                n_frames_vent = self.clf._ventanas.n_frames(tid)
                pct  = min(n_frames_vent / 16, 1.0)
                bx1, by1 = x1, ey2 + 3
                bx2 = x1 + w_box
                cv2.rectangle(frame, (bx1, by1), (bx2, by1 + 4), (40, 40, 40), -1)
                cv2.rectangle(frame, (bx1, by1), (bx1 + int(w_box * pct), by1 + 4), color, -1)

        # HUD global
        self._hud(frame, len(dets), frame_idx)
        return frame

    def _hud(self, frame: np.ndarray, n: int, frame_idx: int) -> None:
        ahora = time.perf_counter()
        self._fps_buf.append(1.0 / max(ahora - self._t_ultimo, 1e-6))
        self._t_ultimo = ahora
        fps = sum(self._fps_buf) / len(self._fps_buf)

        for i, linea in enumerate([
            f"FPS: {fps:5.1f}", f"Frame: {frame_idx}", f"Activos: {n}", "[LSTM States]"
        ]):
            (tw, th), _ = cv2.getTextSize(linea, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            y = 20 + i * (th + 8)
            cv2.rectangle(frame, (8, y - th - 2), (8 + tw + 6, y + 2), (0, 0, 0), -1)
            cv2.putText(frame, linea, (11, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1, cv2.LINE_AA)


# ── Loop principal ────────────────────────────────────────────────────────────

def run(fuente: str, yolo_weights: str, device: str, guardar: str | None) -> None:

    try:
        fuente_cv: str | int = int(fuente)
    except (ValueError, TypeError):
        fuente_cv = fuente

    cap = cv2.VideoCapture(fuente_cv)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir: {fuente}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_orig    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    escala    = WIN_W / max(w_orig, 1)
    w_disp, h_disp = int(w_orig * escala), int(h_orig * escala)
    print(f"[INFO] Video: {w_orig}×{h_orig} @ {fps_video:.1f}fps | {total} frames")

    pipeline = PipelineLSTM(yolo_weights, device)

    writer = None
    if guardar:
        writer = cv2.VideoWriter(
            guardar, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w_disp, h_disp)
        )

    # Velocidad de reproducción: 1.0 = normal, 2.0 = doble, etc.
    velocidad  = 1.0
    delay_base = 1.0 / fps_video
    pausado    = False
    frame_idx  = 0
    frame_disp = None

    print("Controles: Q/ESC=salir | ESPACIO=pausa | +/-=velocidad | S=captura | I=stats\n")

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fin del video.")
                break

            frame_out  = pipeline.procesar(frame, frame_idx)
            frame_disp = cv2.resize(frame_out, (w_disp, h_disp))
            frame_idx += 1

            if writer:
                writer.write(frame_disp)

        if frame_disp is not None:
            # Indicador de velocidad en esquina inferior derecha
            vel_txt = f"x{velocidad:.1f}"
            cv2.putText(frame_disp, vel_txt, (w_disp - 70, h_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.imshow("LSTM Estados — Maquinaria Minera", frame_disp)

        # Delay ajustado por velocidad
        delay_ms = max(1, int(delay_base / velocidad * 1000))
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            pausado = not pausado
            print(f"[INFO] {'PAUSADO' if pausado else 'REANUDADO'}")
        elif key == ord("+") or key == ord("="):
            velocidad = min(velocidad + 0.25, 8.0)
            print(f"[INFO] Velocidad: x{velocidad:.2f}")
        elif key == ord("-"):
            velocidad = max(velocidad - 0.25, 0.25)
            print(f"[INFO] Velocidad: x{velocidad:.2f}")
        elif key == ord("s") and frame_disp is not None:
            nombre = f"captura_{frame_idx:06d}.jpg"
            cv2.imwrite(nombre, frame_disp)
            print(f"[INFO] Captura: {nombre}")
        elif key == ord("i"):
            print(pipeline.clf.estadisticas())

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Finalizado. Frames procesados: {frame_idx}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Inferencia LSTM estados operacionales en tiempo real"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video",  help="Ruta al video grabado")
    g.add_argument("--rtsp",   help="URL RTSP de la cámara IP")
    g.add_argument("--camara", help="Índice de cámara (0, 1, ...)")
    p.add_argument("--yolo",    default="models/best_2.pt",
                   help="Pesos YOLO (default: models/best_2.pt)")
    p.add_argument("--device",  default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--guardar", default=None,
                   help="Guardar video anotado en esta ruta")
    args = p.parse_args()

    run(
        fuente       = args.video or args.rtsp or args.camara,
        yolo_weights = args.yolo,
        device       = args.device,
        guardar      = args.guardar,
    )


if __name__ == "__main__":
    main()
