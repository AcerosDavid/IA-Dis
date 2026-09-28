"""
Modulo de deteccion y tracking de maquinaria minera.

Optimizaciones implementadas:
  - BoT-SORT con config tuneada para minería (cfg/botsort_mining.yaml)
  - Filtro de objetos estáticos: descarta IDs sin movimiento tras N frames
  - ONNX Runtime con paralelismo OpenMP para CPU
  - Exportación automática a ONNX (dynamic=True)
  - Resize explícito antes de inferencia
  - Fallback a PyTorch si ONNX falla
"""
from __future__ import annotations

import os
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# ── Mapeo de clases ───────────────────────────────────────────────────────────
CLASE_MAP: dict[str, str] = {
    "dump_truck":   "camion",
    "excavator":    "excavadora",
    "wheel_loader": "pala",
    "dump":         "camion",
    "truck":        "camion",
}

# Tamaño de entrada — benchmark confirmó 320 como óptimo para CPU
DEFAULT_IMGSZ = 320

# Ruta al config del tracker (relativa al CWD = raíz del proyecto)
TRACKER_CFG = os.path.join("cfg", "botsort_mining.yaml")

# ── Filtro de objetos estáticos ───────────────────────────────────────────────
# Después de STATIC_FRAMES_MIN frames sin moverse más de STATIC_PIX_THRESH px,
# el objeto se considera estático y se descarta de las detecciones activas.
STATIC_FRAMES_MIN  = 30    # ~6s a 5fps de inferencia
STATIC_PIX_THRESH  = 8.0   # píxeles de desplazamiento del centroide


@dataclass
class _HistorialTrack:
    """Historial de posición de un track para detectar si está quieto."""
    centroides: list[tuple[float, float]] = field(default_factory=list)
    frames_estatico: int = 0
    es_estatico: bool    = False

    def actualizar(self, cx: float, cy: float) -> None:
        self.centroides.append((cx, cy))
        if len(self.centroides) > STATIC_FRAMES_MIN:
            self.centroides.pop(0)

        if len(self.centroides) >= STATIC_FRAMES_MIN:
            # Calcular desplazamiento máximo en los últimos N frames
            xs = [p[0] for p in self.centroides]
            ys = [p[1] for p in self.centroides]
            rango = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)

            if rango < STATIC_PIX_THRESH:
                self.frames_estatico += 1
                if self.frames_estatico >= 3:   # confirmación en 3 actualizaciones
                    self.es_estatico = True
            else:
                self.frames_estatico = 0
                self.es_estatico     = False


def _onnx_path_for(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".onnx"


def export_to_onnx(model_path: str, imgsz: int = DEFAULT_IMGSZ) -> str:
    """Exporta .pt → ONNX con dynamic=True (acepta cualquier imgsz)."""
    from ultralytics import YOLO

    onnx_path = _onnx_path_for(model_path)
    print(f"[INFO] Exportando {os.path.basename(model_path)} → ONNX (imgsz={imgsz})…")
    model = YOLO(model_path)
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=True,
        dynamic=True,
    )
    print(f"[INFO] ONNX listo: {onnx_path}")
    return onnx_path


# ── Clase principal ───────────────────────────────────────────────────────────

class Detector:
    """
    Detector y tracker de maquinaria minera.

    Backends (selección automática en load_model):
      - PyTorch + CUDA  si hay GPU disponible
      - ONNX Runtime    en CPU (más rápido que PyTorch CPU)
      - PyTorch CPU     fallback

    Tracker: BoT-SORT con cfg/botsort_mining.yaml
      - track_buffer=75 → 15s de memoria a 5fps (sobrevive oclusiones)
      - new_track_thresh=0.55 → no crea IDs por sombras/polvo
      - appearance_thresh=0.65 → re-identifica tras nube de polvo

    Filtro de estáticos:
      - Objetos sin movimiento en STATIC_FRAMES_MIN frames → descartados
      - Evita que contenedores, tolvas y pilas de material aparezcan como camiones
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5) -> None:
        self.model_path           = model_path
        self.confidence_threshold = confidence_threshold
        self.model                = None
        self._device              = "cpu"
        self._use_onnx            = False
        self._ort_session         = None
        self._class_names: dict[int, str] = {}
        # Historial por track_id para filtro de estáticos
        self._historial: dict[int, _HistorialTrack] = defaultdict(_HistorialTrack)
        # Tracker config path (absoluto para evitar problemas de CWD)
        self._tracker_cfg = os.path.abspath(TRACKER_CFG) if os.path.exists(TRACKER_CFG) else None

    # ── Carga ─────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        cuda = self._cuda_available()
        if self._tracker_cfg:
            print(f"[INFO] Tracker: BoT-SORT (cfg/botsort_mining.yaml)")
        else:
            print(f"[WARN] cfg/botsort_mining.yaml no encontrado — usando botsort default")

        if cuda:
            self._load_pytorch(device="cuda")
        else:
            try:
                import onnxruntime  # noqa: F401
                self._load_onnx()
            except ImportError:
                print("[WARN] onnxruntime no instalado — usando PyTorch CPU")
                self._load_pytorch(device="cpu")
            except Exception as e:
                print(f"[WARN] ONNX falló ({e}) — usando PyTorch CPU")
                self._load_pytorch(device="cpu")

    def _load_pytorch(self, device: str) -> None:
        from ultralytics import YOLO
        try:
            self.model     = YOLO(self.model_path)
            self._device   = device
            self._use_onnx = False
            print(f"[INFO] Modelo {os.path.basename(self.model_path)} — PyTorch/{device}")
        except Exception as e:
            print(f"[MODEL ERROR] {self.model_path}: {e}")
            raise SystemExit(1) from e

    def _load_onnx(self) -> None:
        import onnxruntime as rt
        onnx_path = _onnx_path_for(self.model_path)
        if not os.path.exists(onnx_path):
            onnx_path = export_to_onnx(self.model_path, imgsz=DEFAULT_IMGSZ)

        opts = rt.SessionOptions()
        opts.intra_op_num_threads  = os.cpu_count() or 4
        opts.inter_op_num_threads  = os.cpu_count() or 4
        opts.execution_mode        = rt.ExecutionMode.ORT_PARALLEL
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._ort_session = rt.InferenceSession(
            onnx_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        from ultralytics import YOLO
        tmp = YOLO(self.model_path)
        self._class_names = tmp.names
        del tmp

        self._use_onnx = True
        self._device   = "cpu"
        print(f"[INFO] Modelo ONNX — {os.cpu_count()} hilos CPU")

    # ── Inferencia principal ──────────────────────────────────────────────────

    def track_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> list[dict]:
        """
        Inferencia + BoT-SORT + filtro de estáticos.

        Returns:
            Lista de detecciones ACTIVAS (excluyendo objetos estáticos).
            Cada dict: maquina_id, track_id, tipo_maquina, clase_id,
                       confianza, numero_fotograma, timestamp, bbox,
                       es_estatico (siempre False — los estáticos se descartan).
        """
        if self._use_onnx:
            raw = self._track_onnx(frame, frame_number, timestamp, imgsz)
        else:
            raw = self._track_pytorch(frame, frame_number, timestamp, imgsz)

        return self._filtrar_estaticos(raw, frame_number)

    def _filtrar_estaticos(
        self,
        detecciones: list[dict],
        frame_number: int,
    ) -> list[dict]:
        """
        Actualiza el historial de movimiento de cada track_id y descarta
        los que llevan STATIC_FRAMES_MIN frames sin moverse más de
        STATIC_PIX_THRESH píxeles.

        También limpia el historial de IDs que ya no aparecen para no
        acumular memoria indefinidamente.
        """
        ids_vistos: set[int] = set()
        activas: list[dict]  = []

        for det in detecciones:
            tid = det["track_id"]
            ids_vistos.add(tid)

            # Calcular centroide del bbox
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            hist = self._historial[tid]
            hist.actualizar(cx, cy)

            if hist.es_estatico:
                # Log solo la primera vez que se marca como estático
                if hist.frames_estatico == 3:
                    print(
                        f"[FILTRO] ID:{tid} {det['tipo_maquina']} marcado como "
                        f"ESTÁTICO en frame {frame_number} — ignorando"
                    )
                continue   # descartar de las detecciones activas

            activas.append(det)

        # Limpiar historial de IDs que desaparecieron (evitar memory leak)
        ids_a_borrar = [tid for tid in self._historial if tid not in ids_vistos]
        for tid in ids_a_borrar:
            del self._historial[tid]

        return activas

    def reset_historial(self) -> None:
        """Limpia el historial de movimiento (llamar al reiniciar el video)."""
        self._historial.clear()

    # ── Backends ──────────────────────────────────────────────────────────────

    def _track_pytorch(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
        imgsz: int,
    ) -> list[dict]:
        import cv2
        if self.model is None:
            raise RuntimeError("Modelo no cargado.")

        frame_r = cv2.resize(frame, (imgsz, imgsz))
        h_orig, w_orig = frame.shape[:2]
        scale_x = w_orig / imgsz
        scale_y = h_orig / imgsz

        # Pasar tracker config — si no existe, BoT-SORT con defaults
        tracker_kwarg = {}
        if self._tracker_cfg and os.path.exists(self._tracker_cfg):
            tracker_kwarg["tracker"] = self._tracker_cfg

        resultados = self.model.track(
            frame_r,
            persist=True,
            verbose=False,
            imgsz=imgsz,
            conf=self.confidence_threshold,
            device=self._device,
            iou=0.7,
            max_det=20,
            agnostic_nms=True,
            **tracker_kwarg,
        )

        detecciones: list[dict] = []
        for resultado in resultados:
            if resultado.boxes is None:
                continue
            boxes = resultado.boxes
            ids   = boxes.id
            if ids is None:
                continue

            for i, box in enumerate(boxes):
                clase_nombre = resultado.names[int(box.cls)]
                if clase_nombre not in CLASE_MAP:
                    continue

                tipo     = CLASE_MAP[clase_nombre]
                track_id = int(ids[i])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detecciones.append({
                    "maquina_id":       f"{tipo}_{track_id}",
                    "track_id":         track_id,
                    "tipo_maquina":     tipo,
                    "clase_id":         clase_nombre,
                    "confianza":        float(box.conf),
                    "numero_fotograma": frame_number,
                    "timestamp":        timestamp,
                    "bbox": [
                        x1 * scale_x, y1 * scale_y,
                        x2 * scale_x, y2 * scale_y,
                    ],
                })

        return detecciones

    def _track_onnx(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
        imgsz: int,
    ) -> list[dict]:
        """ONNX con ByteTrack fallback (BoT-SORT requiere PyTorch para ReID)."""
        import cv2

        if self._ort_session is None:
            raise RuntimeError("Sesión ONNX no inicializada.")

        h_orig, w_orig = frame.shape[:2]
        img = cv2.resize(frame, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        input_name = self._ort_session.get_inputs()[0].name
        outputs    = self._ort_session.run(None, {input_name: img})
        preds      = outputs[0][0]   # (N, 7+)

        mask  = preds[:, 4] >= self.confidence_threshold
        preds = preds[mask]
        if preds.shape[0] == 0:
            return []

        scale_x = w_orig / imgsz
        scale_y = h_orig / imgsz
        boxes_scaled = preds[:, :4].copy()
        boxes_scaled[:, [0, 2]] *= scale_x
        boxes_scaled[:, [1, 3]] *= scale_y

        # Necesitamos PyTorch para el tracker
        import torch
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

        boxes_t = torch.tensor(boxes_scaled, dtype=torch.float32)
        confs_t = torch.tensor(preds[:, 4],  dtype=torch.float32)
        cls_t   = torch.tensor(preds[:, 5],  dtype=torch.float32)

        from ultralytics.engine.results import Results
        orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result   = Results(
            orig_img=orig_rgb, path="",
            names=self._class_names,
            boxes=torch.cat([boxes_t, confs_t.unsqueeze(1), cls_t.unsqueeze(1)], dim=1),
        )

        tracked = None
        if hasattr(self.model, 'predictor') and self.model.predictor:
            try:
                tracked = self.model.predictor.trackers[0].update(
                    result.boxes.data.numpy(), frame
                )
            except Exception:
                pass

        if tracked is None:
            return self._track_pytorch(frame, frame_number, timestamp, imgsz)

        detecciones: list[dict] = []
        for row in tracked:
            x1, y1, x2, y2, track_id, conf, cls_id = row[:7]
            clase_nombre = self._class_names.get(int(cls_id), "")
            if clase_nombre not in CLASE_MAP:
                continue
            tipo = CLASE_MAP[clase_nombre]
            detecciones.append({
                "maquina_id":       f"{tipo}_{int(track_id)}",
                "track_id":         int(track_id),
                "tipo_maquina":     tipo,
                "clase_id":         clase_nombre,
                "confianza":        float(conf),
                "numero_fotograma": frame_number,
                "timestamp":        timestamp,
                "bbox":             [float(x1), float(y1), float(x2), float(y2)],
            })

        return detecciones

    # ── Legacy ────────────────────────────────────────────────────────────────

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
    ) -> list[dict]:
        """Inferencia simple sin tracking (modo legacy/tests)."""
        import cv2
        if self.model is None:
            if self._use_onnx:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            else:
                raise RuntimeError("Modelo no cargado.")

        frame_r    = cv2.resize(frame, (DEFAULT_IMGSZ, DEFAULT_IMGSZ))
        resultados = self.model(
            frame_r, verbose=False,
            conf=self.confidence_threshold,
            iou=0.7, max_det=20, agnostic_nms=True,
        )
        eventos: list[dict] = []
        for resultado in resultados:
            if resultado.boxes is None:
                continue
            for box in resultado.boxes:
                clase_nombre = resultado.names[int(box.cls)]
                confianza    = float(box.conf)
                if clase_nombre not in CLASE_MAP:
                    continue
                if confianza < self.confidence_threshold:
                    continue
                eventos.append({
                    "tipo_maquina":     CLASE_MAP[clase_nombre],
                    "clase_id":         clase_nombre,
                    "confianza":        confianza,
                    "numero_fotograma": frame_number,
                    "timestamp":        timestamp,
                })
        return eventos

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
