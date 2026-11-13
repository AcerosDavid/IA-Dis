"""
etiquetar_video.py — Etiquetador de video en vivo (multi-máquina, multi-estado).

═══════════════════════════════════════════════════════════════
  FLUJO DE USO
═══════════════════════════════════════════════════════════════

  El video tiene excavadoras, palas y camiones mezclados.
  Tú decides en tiempo real qué estás viendo y el script
  guarda los recortes en la carpeta correcta sin que toques
  una sola imagen.

  PASO 1 — Selecciona la MÁQUINA (se queda activa hasta que cambies):
    E → Excavadora
    P → Pala mecánica
    C → Camión

  PASO 2 — Presiona el ESTADO de lo que ves EN ESE MOMENTO:
    ┌─────────────────────────────────────────────────────────┐
    │ Excavadora / Pala          │ Camión                     │
    ├─────────────────────────────────────────────────────────┤
    │ 1 → escarbando_cargando    │ 1 → vacio_andando          │
    │ 2 → descargando            │ 2 → lleno_andando          │
    │ 3 → quieta                 │ 3 → quieto_en_cargue_desc  │
    └─────────────────────────────────────────────────────────┘

  El recorte se guarda EN EL INSTANTE en:
    data/recortes_estados/{maquina}/{estado}/frame_XXXXXXX_idN.jpg

═══════════════════════════════════════════════════════════════
  CONTROLES DE REPRODUCCIÓN
═══════════════════════════════════════════════════════════════
    ESPACIO → pausar / reanudar
    + / =   → aumentar velocidad (×1.25 cada vez, máx ×8)
    -       → reducir velocidad  (×0.8  cada vez, mín ×0.25)
    ←       → retroceder 5 segundos
    →       → avanzar 5 segundos
    G       → saltar a un frame exacto (escribe el número)
    S       → captura de pantalla (sin guardar como dataset)
    0       → LIMPIAR selección activa (dejar de etiquetar)
    Q / ESC → guardar .npz y salir

═══════════════════════════════════════════════════════════════
  USO EN CONSOLA
═══════════════════════════════════════════════════════════════
    python etiquetar_video.py --video data/raw/video.mp4
    python etiquetar_video.py --video data/raw/video.mp4 --yolo weights/best.pt
    python etiquetar_video.py --video data/raw/video.mp4 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ══════════════════════════════════════════════════════════════════════════════
#  TAXONOMÍA CENTRAL — máquinas, estados y carpetas de destino
# ══════════════════════════════════════════════════════════════════════════════

# Tecla de selección de máquina (minúscula / mayúscula ambas aceptadas)
MAQUINAS: dict[str, str] = {
    "e": "excavadora",
    "p": "pala",
    "c": "camion",
}

# Estado que activa cada tecla numérica, según la máquina seleccionada
# El tracker detecta movimiento. El clasificador solo necesita estado de carga.
ESTADOS: dict[str, dict[str, str]] = {
    "excavadora": {
        "1": "escarbando_cargando",
        "2": "descargando",
    },
    "pala": {
        "1": "escarbando_cargando",
        "2": "descargando",
    },
    "camion": {
        "1": "vacio",
        "2": "cargado",
    },
}

# Alias del detector YOLO → tipo canónico (para el bbox tracker)
_YOLO_A_TIPO: dict[str, str] = {
    "dump_truck":   "camion",
    "excavator":    "excavadora",
    "wheel_loader": "pala",
    "dump":         "camion",
    "truck":        "camion",
}

# ── Paleta visual ─────────────────────────────────────────────────────────────
_COLOR_MAQUINA: dict[str, tuple[int, int, int]] = {
    "camion":     (0,   255,  50),
    "excavadora": (0,   200, 255),
    "pala":       (255,  50, 180),
}
_COLOR_ESTADO: dict[str, tuple[int, int, int]] = {
    "escarbando_cargando":  (0,   220, 255),
    "descargando":          (0,    50, 255),
    "vacio":                (200, 200, 255),
    "cargado":              (0,   200,  50),
}

# ── Parámetros ────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 16     # frames en la ventana LSTM
FEATURE_DIM  = 576    # dim del vector de características
CADA_N_GUARD = 3      # guardar un recorte cada N frames (evita redundancia)
IMGSZ        = 320    # resolución de inferencia YOLO
CONF_YOLO    = 0.40   # umbral mínimo de detección
WIN_W        = 1280   # ancho de la ventana de display
MARGEN_ROI   = 10     # píxeles extra alrededor del bbox


# ══════════════════════════════════════════════════════════════════════════════
#  BUFFER DE SECUENCIAS POR TRACK
# ══════════════════════════════════════════════════════════════════════════════

class _TrackBuffer:
    """Acumula vectores de características de un track_id concreto."""

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self.vectores: deque = deque(maxlen=window_size)
        self.window_size     = window_size

    def agregar(self, vec: np.ndarray) -> None:
        self.vectores.append(vec.astype(np.float32))

    def secuencia(self) -> np.ndarray | None:
        """(window_size, feature_dim) con padding de ceros si faltan frames."""
        if not self.vectores:
            return None
        frames = list(self.vectores)
        dim    = frames[0].shape[0]
        if len(frames) < self.window_size:
            pad    = [np.zeros(dim, dtype=np.float32)] * (self.window_size - len(frames))
            frames = pad + frames
        return np.stack(frames, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  ETIQUETADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class Etiquetador:
    """
    Reproductor-etiquetador en vivo.

    Estado de la sesión:
      - _maquina_activa : máquina que el usuario ha seleccionado (E/P/C)
      - _estado_activo  : estado que está etiquetando ahora (1/2/3)
      - _grabando       : True mientras hay máquina + estado seleccionados
    """

    def __init__(
        self,
        video_path:   str,
        yolo_weights: str  = "weights/best.pt",
        output_dir:   str  = "data/recortes_estados",
        device:       str  = "cpu",
    ) -> None:
        from ultralytics import YOLO
        from src.ai.feature_extractor import FeatureExtractor

        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.device     = device

        print(f"[INFO] Cargando YOLO: {yolo_weights}")
        self.yolo      = YOLO(yolo_weights)
        self.extractor = FeatureExtractor(device=device)

        # Estado de la sesión
        self._maquina_activa: str | None = None
        self._estado_activo:  str | None = None
        self._frames_desde_guard: int    = 0

        # Buffers de features por track_id
        self._buffers: dict[int, _TrackBuffer] = {}

        # Dataset en memoria: {tipo → {clases, X[], y[]}}
        self._dataset: dict[str, dict] = {
            tipo: {"clases": list(mapa_estados.values()), "X": [], "y": []}
            for tipo, mapa_estados in ESTADOS.items()
        }

        # Contadores por (máquina, estado)
        self._conteo: dict[str, dict[str, int]] = {
            tipo: {estado: 0 for estado in mapa.values()}
            for tipo, mapa in ESTADOS.items()
        }
        self._seq_conteo: dict[str, int] = {tipo: 0 for tipo in ESTADOS}

        # Crear todas las carpetas de salida de una vez
        self._crear_carpetas()
        self._mostrar_guia()

    # ── Ejecución ─────────────────────────────────────────────────────────────

    def ejecutar(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] No se puede abrir: {self.video_path}")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_orig    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        escala    = WIN_W / max(w_orig, 1)
        w_disp    = int(w_orig * escala)
        h_disp    = int(h_orig * escala)

        print(f"[INFO] {w_orig}×{h_orig} @ {fps_video:.1f}fps | {total_fr} frames")

        velocidad   = 1.0
        delay_base  = 1.0 / fps_video
        pausado     = False
        frame_idx   = 0
        frame_disp  = None
        dets_cache: list[dict] = []

        while True:
            if not pausado:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Fin del video.")
                    break

                # ── Detección + features ───────────────────────────────────────
                dets_cache = self._detectar(frame)
                self._actualizar_buffers(frame, dets_cache)

                # ── Guardar recortes si hay selección activa ───────────────────
                if self._maquina_activa and self._estado_activo:
                    self._frames_desde_guard += 1
                    if self._frames_desde_guard >= CADA_N_GUARD:
                        self._frames_desde_guard = 0
                        self._guardar_recortes(frame, dets_cache, frame_idx)
                        self._acumular_secuencias(dets_cache)

                # ── Dibujado ───────────────────────────────────────────────────
                frame_out  = self._dibujar(frame.copy(), dets_cache,
                                           frame_idx, total_fr, velocidad)
                frame_disp = cv2.resize(frame_out, (w_disp, h_disp))
                frame_idx += 1

            if frame_disp is not None:
                cv2.imshow("Etiquetador — Maquinaria Minera", frame_disp)

            # ── Teclado ────────────────────────────────────────────────────────
            delay_ms = max(1, int(delay_base / velocidad * 1000))
            key      = cv2.waitKey(delay_ms) & 0xFF
            key_char = chr(key).lower() if 32 <= key <= 126 else ""

            if key in (ord("q"), 27):
                break

            # Reproducción
            elif key == ord(" "):
                pausado = not pausado
                if pausado:
                    self._estado_activo = None   # dejar de grabar al pausar
                print(f"  {'⏸  PAUSADO' if pausado else '▶  REANUDADO'}")

            elif key_char in ("+", "="):
                velocidad = min(velocidad * 1.25, 8.0)
                print(f"  Velocidad: ×{velocidad:.2f}")

            elif key_char == "-":
                velocidad = max(velocidad * 0.8, 0.25)
                print(f"  Velocidad: ×{velocidad:.2f}")

            elif key in (81, 2, ord("a")):   # ← o A  → retroceder 5s
                pos = max(0, frame_idx - int(fps_video * 5))
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                frame_idx = pos
                self._estado_activo = None

            elif key in (83, 3, ord("d")):   # → o D  → avanzar 5s
                pos = min(total_fr - 1, frame_idx + int(fps_video * 5))
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                frame_idx = pos
                self._estado_activo = None

            elif key_char == "g":             # Ir a frame específico
                cv2.destroyAllWindows()
                cap.release()
                num_str = input("\n  Número de frame al que saltar: ").strip()
                cap = cv2.VideoCapture(self.video_path)
                try:
                    pos       = int(num_str)
                    frame_idx = max(0, min(pos, total_fr - 1))
                except ValueError:
                    pass
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                print(f"  → Frame {frame_idx}")

            elif key_char == "s":             # Captura sin etiquetar
                if frame_disp is not None:
                    nombre = f"captura_{frame_idx:07d}.jpg"
                    cv2.imwrite(nombre, frame_disp)
                    print(f"  Captura: {nombre}")

            elif key_char == "0":             # Limpiar selección
                self._maquina_activa = None
                self._estado_activo  = None
                print("  [0] Selección limpiada — sin etiqueta activa")

            # ── Selección de MÁQUINA (E / P / C) ──────────────────────────────
            elif key_char in MAQUINAS:
                self._maquina_activa = MAQUINAS[key_char]
                self._estado_activo  = None   # resetear estado al cambiar máquina
                self._frames_desde_guard = 0
                print(f"\n  [{key_char.upper()}] Máquina → {self._maquina_activa.upper()}")
                print(f"       Ahora presiona 1, 2 o 3 para el estado.")

            # ── Selección de ESTADO (1 / 2 / 3) ──────────────────────────────
            elif key_char in ("1", "2", "3"):
                if self._maquina_activa is None:
                    print("  ⚠  Primero selecciona la máquina: E, P o C")
                else:
                    mapa_est      = ESTADOS[self._maquina_activa]
                    nuevo_estado  = mapa_est.get(key_char)
                    if nuevo_estado:
                        self._estado_activo      = nuevo_estado
                        self._frames_desde_guard = CADA_N_GUARD   # guardar ya
                        destino = self.output_dir / self._maquina_activa / nuevo_estado
                        n_actual = self._conteo[self._maquina_activa][nuevo_estado]
                        print(
                            f"  [{key_char}] {self._maquina_activa}/{nuevo_estado} "
                            f"→ {destino}  ({n_actual} guardados)"
                        )

        cap.release()
        cv2.destroyAllWindows()
        self._guardar_npz()
        self._resumen_final()

    # ── Detección YOLO ────────────────────────────────────────────────────────

    def _detectar(self, frame: np.ndarray) -> list[dict]:
        resultados = self.yolo.track(
            frame,
            persist      = True,
            verbose      = False,
            imgsz        = IMGSZ,
            conf         = CONF_YOLO,
            tracker      = "bytetrack.yaml",
            device       = self.device,
            max_det      = 20,
            agnostic_nms = True,
        )
        dets: list[dict] = []
        for res in resultados:
            if res.boxes is None or res.boxes.id is None:
                continue
            for i, box in enumerate(res.boxes):
                clase = res.names[int(box.cls)]
                tipo  = _YOLO_A_TIPO.get(clase)
                if tipo is None:
                    continue
                dets.append({
                    "track_id":  int(res.boxes.id[i]),
                    "tipo":      tipo,
                    "confianza": float(box.conf),
                    "bbox":      box.xyxy[0].tolist(),
                })
        return dets

    # ── Buffers de features ────────────────────────────────────────────────────

    def _actualizar_buffers(self, frame: np.ndarray, dets: list[dict]) -> None:
        for det in dets:
            tid = det["track_id"]
            if tid not in self._buffers:
                self._buffers[tid] = _TrackBuffer(WINDOW_SIZE)
            roi = self._recortar(frame, det["bbox"])
            if roi is not None:
                vec = self.extractor.extract(roi)
                self._buffers[tid].agregar(vec)

    # ── Guardar recortes ───────────────────────────────────────────────────────

    def _guardar_recortes(
        self,
        frame:     np.ndarray,
        dets:      list[dict],
        frame_idx: int,
    ) -> None:
        """
        Guarda el recorte de CADA máquina detectada del tipo activo en la
        carpeta data/recortes_estados/{maquina}/{estado}/
        """
        if not self._maquina_activa or not self._estado_activo:
            return

        ts      = time.strftime("%H%M%S")
        carpeta = self.output_dir / self._maquina_activa / self._estado_activo
        carpeta.mkdir(parents=True, exist_ok=True)

        guardados = 0
        for det in dets:
            # Guardar recortes de TODAS las máquinas del tipo activo
            # (si hay 2 excavadoras en pantalla, guarda las 2)
            if det["tipo"] != self._maquina_activa:
                continue
            roi = self._recortar(frame, det["bbox"])
            if roi is None:
                continue
            tid    = det["track_id"]
            nombre = f"frame_{frame_idx:07d}_id{tid}_{ts}.jpg"
            cv2.imwrite(str(carpeta / nombre), roi)
            self._conteo[self._maquina_activa][self._estado_activo] += 1
            guardados += 1

        # Si el tipo activo no tiene detecciones pero el usuario está etiquetando,
        # guardar un crop del frame completo centrado como fallback opcional
        # (desactivado por defecto — descomentar si quieres capturar sin bbox)
        # if guardados == 0:
        #     h, w = frame.shape[:2]
        #     nombre = f"frame_{frame_idx:07d}_full_{ts}.jpg"
        #     cv2.imwrite(str(carpeta / nombre), frame)

    # ── Acumular secuencias para el .npz ──────────────────────────────────────

    def _acumular_secuencias(self, dets: list[dict]) -> None:
        """
        Si el buffer de un track tiene WINDOW_SIZE vectores y hay
        un estado activo, guarda la secuencia en el dataset en memoria.
        """
        if not self._maquina_activa or not self._estado_activo:
            return

        tipo   = self._maquina_activa
        estado = self._estado_activo
        clases = self._dataset[tipo]["clases"]
        if estado not in clases:
            return
        idx_clase = clases.index(estado)

        for det in dets:
            if det["tipo"] != tipo:
                continue
            buf = self._buffers.get(det["track_id"])
            if buf is None or len(buf.vectores) < WINDOW_SIZE:
                continue
            seq = buf.secuencia()
            if seq is not None:
                self._dataset[tipo]["X"].append(seq)
                self._dataset[tipo]["y"].append(idx_clase)
                self._seq_conteo[tipo] += 1

    # ── Guardar .npz ──────────────────────────────────────────────────────────

    def _guardar_npz(self) -> None:
        Path("data").mkdir(exist_ok=True)
        for tipo, datos in self._dataset.items():
            if not datos["X"]:
                continue
            X      = np.stack(datos["X"]).astype(np.float32)
            y      = np.array(datos["y"], dtype=np.int64)
            clases = np.array(datos["clases"])
            ruta   = Path("data") / f"secuencias_{tipo}.npz"

            # Si ya existe uno previo, concatenar (no sobreescribir)
            if ruta.exists():
                prev = np.load(str(ruta), allow_pickle=True)
                X    = np.concatenate([prev["X"], X], axis=0)
                y    = np.concatenate([prev["y"], y], axis=0)

            np.savez_compressed(str(ruta), X=X, y=y, clases=clases)
            print(f"  [.npz] {ruta} → {len(X)} secuencias")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _recortar(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1 - MARGEN_ROI)
        y1 = max(0, y1 - MARGEN_ROI)
        x2 = min(w, x2 + MARGEN_ROI)
        y2 = min(h, y2 + MARGEN_ROI)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    def _crear_carpetas(self) -> None:
        for tipo, mapa in ESTADOS.items():
            for estado in mapa.values():
                (self.output_dir / tipo / estado).mkdir(parents=True, exist_ok=True)

    # ── Dibujado ──────────────────────────────────────────────────────────────

    def _dibujar(
        self,
        frame:     np.ndarray,
        dets:      list[dict],
        frame_idx: int,
        total_fr:  int,
        velocidad: float,
    ) -> np.ndarray:
        h_fr, w_fr = frame.shape[:2]

        # ── Bboxes detectados ──────────────────────────────────────────────────
        for det in dets:
            tid  = det["track_id"]
            tipo = det["tipo"]
            x1, y1, x2, y2 = map(int, det["bbox"])
            w_box = x2 - x1
            h_box = y2 - y1

            # Color: rojo vivo si es la máquina activa, color de tipo si no
            es_activa = (tipo == self._maquina_activa)
            color_base = _COLOR_MAQUINA.get(tipo, (255, 255, 255))
            if es_activa and self._estado_activo:
                color = _COLOR_ESTADO.get(self._estado_activo, color_base)
            elif es_activa:
                color = (0, 255, 255)   # cyan — seleccionada pero sin estado
            else:
                # Máquinas no activas: semitransparentes / apagadas
                color = tuple(int(c * 0.5) for c in color_base)

            # Bbox + relleno semitransparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            alpha = 0.15 if es_activa else 0.05
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            grosor_bbox = 3 if es_activa else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor_bbox)

            # Esquinas estilo SmartPSS
            cs = min(w_box, h_box, 20)
            if es_activa:
                for ox, oy, dx, dy in [
                    (x1, y1,  cs, 0), (x1, y1, 0,  cs),
                    (x2, y1, -cs, 0), (x2, y1, 0,  cs),
                    (x1, y2,  cs, 0), (x1, y2, 0, -cs),
                    (x2, y2, -cs, 0), (x2, y2, 0, -cs),
                ]:
                    cv2.line(frame, (ox, oy), (ox + dx, oy + dy), color, 3)

            # Etiqueta ID + tipo + progreso ventana
            buf    = self._buffers.get(tid)
            n_vec  = len(buf.vectores) if buf else 0
            lbl    = f"ID:{tid} {tipo} [{n_vec}/{WINDOW_SIZE}]"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ly = max(y1 - 5, th + 6)
            cv2.rectangle(frame, (x1, ly - th - 5), (x1 + tw + 6, ly + 2), (0, 0, 0), -1)
            cv2.putText(frame, lbl, (x1 + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Si hay estado activo y es la máquina activa → badge de estado
            if es_activa and self._estado_activo:
                est_lbl = f"→ {self._estado_activo.upper().replace('_', ' ')}"
                (ew, eh), _ = cv2.getTextSize(est_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                ey1 = y2 + 4
                ey2 = ey1 + eh + 10
                cv2.rectangle(frame, (x1, ey1), (x1 + ew + 10, ey2), (0, 0, 0), -1)
                cv2.rectangle(frame, (x1, ey1), (x1 + ew + 10, ey2), color, 2)
                cv2.putText(frame, est_lbl, (x1 + 5, ey2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        # ── Panel de control (lado izquierdo) ─────────────────────────────────
        self._panel(frame, frame_idx, total_fr, velocidad)

        # ── Barra de progreso del video ────────────────────────────────────────
        if total_fr > 0:
            pct = frame_idx / total_fr
            cv2.rectangle(frame, (0, h_fr - 6), (w_fr, h_fr), (30, 30, 30), -1)
            cv2.rectangle(frame, (0, h_fr - 6), (int(w_fr * pct), h_fr), (0, 200, 100), -1)
            # Marca de posición en tiempo
            seg_total = total_fr / max(cap_fps := 30.0, 1)   # aprox
            pos_txt   = f"{int(frame_idx/30//60):02d}:{int(frame_idx/30%60):02d} / {int(total_fr/30//60):02d}:{int(total_fr/30%60):02d}"
            cv2.putText(frame, pos_txt, (w_fr // 2 - 60, h_fr - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def _panel(
        self,
        frame:     np.ndarray,
        frame_idx: int,
        total_fr:  int,
        velocidad: float,
    ) -> None:
        """Panel HUD en esquina superior izquierda con toda la información."""

        maquina = self._maquina_activa or "—"
        estado  = self._estado_activo  or "—"

        # Encabezado de estado activo
        lineas: list[tuple[str, tuple[int, int, int]]] = [
            (f"Frame {frame_idx}/{total_fr}",     (200, 255, 200)),
            (f"Vel: ×{velocidad:.2f}",             (200, 255, 200)),
            ("─" * 34,                             (60,  60,  60)),
            # Máquina activa
            (f"MÁQUINA: {maquina.upper()}",
             _COLOR_MAQUINA.get(maquina, (255, 255, 100)) if maquina != "—" else (120, 120, 120)),
            (f"ESTADO:  {estado.upper().replace('_',' ')}",
             _COLOR_ESTADO.get(estado, (255, 255, 100)) if estado != "—" else (120, 120, 120)),
            ("─" * 34,                             (60,  60,  60)),
            # Guía de teclas — máquina
            ("[E] Excavadora  [P] Pala  [C] Camión", (180, 180, 100)),
        ]

        # Teclas de estado según máquina activa
        if maquina != "—":
            for tecla, est in ESTADOS[maquina].items():
                marca = "► " if est == self._estado_activo else "  "
                n     = self._conteo[maquina][est]
                lineas.append((
                    f"{marca}[{tecla}] {est:<30} {n:4d}",
                    _COLOR_ESTADO.get(est, (200, 200, 200))
                ))
        else:
            lineas += [
                ("  [1] Estado 1", (120, 120, 120)),
                ("  [2] Estado 2", (120, 120, 120)),
                ("  [3] Estado 3", (120, 120, 120)),
            ]

        lineas += [
            ("─" * 34,                                         (60,  60,  60)),
            ("[0] Limpiar  [SPC] Pausa  [+/-] Vel",            (140, 140, 140)),
            ("[←/→] ±5s   [G] Ir a frame  [Q] Salir",         (140, 140, 140)),
            ("─" * 34,                                         (60,  60,  60)),
            # Totales globales
            (f"Total recortes: {sum(sum(v.values()) for v in self._conteo.values())}",
             (200, 255, 200)),
            (f"Total secuencias: {sum(self._seq_conteo.values())}",
             (200, 200, 255)),
        ]

        # Dibujar panel
        y = 18
        for texto, color in lineas:
            if texto.startswith("─"):
                # Línea separadora
                y += 4
                continue
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
            cv2.rectangle(frame, (4, y - th - 2), (4 + tw + 8, y + 2), (0, 0, 0), -1)
            cv2.putText(frame, texto, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
            y += th + 6

    # ── Resumen de sesión ─────────────────────────────────────────────────────

    def _mostrar_guia(self) -> None:
        print("\n" + "═" * 62)
        print("  ETIQUETADOR DE VIDEO — Maquinaria Minera")
        print("═" * 62)
        for tipo, mapa in ESTADOS.items():
            tecla_maq = next(k for k, v in MAQUINAS.items() if v == tipo)
            print(f"\n  [{tecla_maq.upper()}] {tipo.upper()}")
            for tecla, est in mapa.items():
                carpeta = self.output_dir / tipo / est
                print(f"    [{tecla}] → {est:<35} → {carpeta}")
        print("\n  [0] Limpiar selección")
        print("  [ESPACIO] Pausar  [+/-] Velocidad  [←/→] ±5s")
        print("  [G] Ir a frame  [S] Captura  [Q/ESC] Guardar y salir")
        print("═" * 62 + "\n")

    def _resumen_final(self) -> None:
        print("\n" + "═" * 62)
        print("  SESIÓN COMPLETADA")
        print("═" * 62)
        total_global = 0
        for tipo, conteo in self._conteo.items():
            total_tipo = sum(conteo.values())
            if total_tipo == 0:
                continue
            print(f"\n  {tipo.upper()} — {self._seq_conteo[tipo]} secuencias")
            for estado, n in conteo.items():
                barra = "█" * min(n // max(total_tipo // 20, 1), 30)
                print(f"    {estado:<35} {n:5d}  {barra}")
            total_global += total_tipo
        print(f"\n  Total recortes guardados : {total_global}")
        print(f"  Recortes en : {self.output_dir.resolve()}")
        print(f"  .npz en     : data/secuencias_{{tipo}}.npz")
        print(f"\n  SIGUIENTE PASO:")
        print(f"    python entrenar_lstm.py --tipo excavadora")
        print(f"    python entrenar_lstm.py --tipo camion")
        print("═" * 62 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _leer_env() -> dict[str, str]:
    """Lee variables clave del .env sin depender de python-dotenv."""
    env: dict[str, str] = {}
    ruta_env = Path(".env")
    if not ruta_env.exists():
        return env
    for linea in ruta_env.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        env[clave.strip()] = valor.strip()
    return env


def main() -> None:
    # Leer .env para usar como defaults si el usuario no pasa argumentos
    env = _leer_env()
    default_video = env.get("VIDEO_PATH", "")
    default_yolo  = env.get("MODEL_PATH", "models/best_2.pt")

    p = argparse.ArgumentParser(
        description="Etiquetador de video en vivo — estados operacionales maquinaria"
    )
    p.add_argument(
        "--video",
        default=default_video or None,
        required=not bool(default_video),
        help=f"Ruta al video (default desde .env: {default_video or 'obligatorio'})",
    )
    p.add_argument(
        "--yolo",
        default=default_yolo,
        help=f"Pesos YOLO (default desde .env: {default_yolo})",
    )
    p.add_argument(
        "--output",
        default="data/recortes_estados",
        help="Carpeta raíz de recortes (default: data/recortes_estados)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo de inferencia (default: cpu)",
    )
    args = p.parse_args()

    # Validar video
    if not args.video:
        print("[ERROR] Debes pasar --video o definir VIDEO_PATH en el .env")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"[ERROR] Video no encontrado: {args.video}")
        sys.exit(1)

    # Validar modelo
    if not os.path.exists(args.yolo):
        print(f"[ERROR] Modelo no encontrado: {args.yolo}")
        print(f"        Comprueba MODEL_PATH en .env o pasa --yolo <ruta>")
        sys.exit(1)

    print(f"[INFO] Video : {args.video}")
    print(f"[INFO] Modelo: {args.yolo}")

    Etiquetador(
        video_path   = args.video,
        yolo_weights = args.yolo,
        output_dir   = args.output,
        device       = args.device,
    ).ejecutar()


if __name__ == "__main__":
    main()
