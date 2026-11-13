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
      - track_buffer=150 → 30s de memoria a 5fps (sobrevive oclusiones largas)
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
        # Tracker propio para el backend ONNX (no depende de YOLO.predictor)
        self._tracker = None

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

    def reset(self) -> None:
        """
        Reinicia TODO el estado entre videos: historial de estáticos y tracker.

        Imprescindible al reutilizar un Detector cacheado: sin esto los track_id
        del video anterior se filtran al siguiente.
        """
        self._historial.clear()
        self._tracker = None
        # En el backend PyTorch, el tracker vive dentro del predictor de YOLO.
        if self.model is not None and getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None

    def _init_tracker(self):
        """Crea un BYTETracker/BoT-SORT a partir de cfg/botsort_mining.yaml."""
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace

        defaults = {
            "tracker_type":      "botsort",
            "track_high_thresh": 0.45,
            "track_low_thresh":  0.20,
            "new_track_thresh":  0.55,
            "track_buffer":      150,
            "match_thresh":      0.70,
            "fuse_score":        True,
            "gmc_method":        "sparseOptFlow",
            "proximity_thresh":  0.30,
            "appearance_thresh": 0.65,
            "with_reid":         False,
            "model":             "auto",
        }
        if self._tracker_cfg and os.path.exists(self._tracker_cfg):
            try:
                defaults.update(YAML.load(self._tracker_cfg) or {})
            except Exception as e:
                print(f"[WARN] No se pudo leer {self._tracker_cfg}: {e}")

        cfg = IterableSimpleNamespace(**defaults)
        cfg.device = "cpu"
        tracker_cls = TRACKER_MAP.get(cfg.tracker_type, TRACKER_MAP["botsort"])
        return tracker_cls(args=cfg)

    # ── Backends ──────────────────────────────────────────────────────────────

    def _track_pytorch(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
        imgsz: int,
    ) -> list[dict]:
        if self.model is None:
            # Puede ocurrir si la sesión ONNX existía pero el tracker falló.
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

        # Pasar tracker config — si no existe, BoT-SORT con defaults
        tracker_kwarg = {}
        if self._tracker_cfg and os.path.exists(self._tracker_cfg):
            tracker_kwarg["tracker"] = self._tracker_cfg

        # YOLO hace su propio letterbox y devuelve las cajas YA escaladas al
        # frame original. No redimensionar antes ni volver a escalar después.
        resultados = self.model.track(
            frame,
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
                    "bbox":             [x1, y1, x2, y2],
                })

        return detecciones

    @staticmethod
    def _letterbox(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
        """
        Redimensiona manteniendo aspecto y rellena hasta (imgsz, imgsz).

        Returns:
            (imagen, ratio, pad_x, pad_y) — para deshacer la transformación.
        """
        import cv2
        h, w = frame.shape[:2]
        ratio = min(imgsz / w, imgsz / h)
        new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (imgsz - new_w) // 2
        pad_y = (imgsz - new_h) // 2
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, ratio, pad_x, pad_y

    def _track_onnx(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
        imgsz: int,
    ) -> list[dict]:
        """
        Inferencia ONNX + NMS + BoT-SORT.

        El ONNX exportado (dynamic=True) devuelve la salida cruda del modelo,
        por lo que hay que hacer NMS aquí. Se soportan dos formatos:
          - (1, 4+nc, N)  YOLOv8/v11/v26 clásico (cx, cy, w, h, scores…)
          - (1, N, 6)     YOLO26 end-to-end (x1, y1, x2, y2, conf, cls)
        """
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.utils.nms import non_max_suppression

        if self._ort_session is None:
            raise RuntimeError("Sesión ONNX no inicializada.")

        h_orig, w_orig = frame.shape[:2]
        canvas, ratio, pad_x, pad_y = self._letterbox(frame, imgsz)

        img = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)[None]

        input_name = self._ort_session.get_inputs()[0].name
        outputs    = self._ort_session.run(None, {input_name: img})
        pred       = outputs[0]
        pred       = pred[0] if pred.ndim == 3 else pred

        # clone(): non_max_suppression modifica las coordenadas in-place.
        # Sin copia, si el backend reutiliza el buffer de salida, el segundo
        # frame interpretaría cajas xyxy como si fueran cxcywh.
        t = torch.from_numpy(np.asarray(pred)).clone()
        if t.ndim != 2:
            return []

        nc = len(self._class_names) or 1
        # Detectar layout: (C, N) clásico vs (N, 6) end-to-end.
        if t.shape[1] > t.shape[0] and t.shape[0] <= 4 + nc:
            t = t.T                       # → (N, C)
            end2end = False
        else:
            end2end = t.shape[1] == 6

        if t.numel() == 0:
            return []

        if end2end:
            batch = t.unsqueeze(0)        # (1, N, 6)
        else:
            batch = t.T.unsqueeze(0)      # (1, C, N)

        det = non_max_suppression(
            batch,
            conf_thres=self.confidence_threshold,
            iou_thres=0.7,
            agnostic=True,
            max_det=20,
            end2end=end2end,
        )[0]

        if det is None or len(det) == 0:
            return []

        # Volver de coordenadas del canvas (letterbox) al frame original.
        boxes = det[:, :4].clone()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w_orig)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h_orig)

        confs = det[:, 4]
        clss  = det[:, 5].long()

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
        if len(boxes) == 0:
            return []

        # Boxes espera (N, 6) = xyxy + conf + cls y deriva .xywh/.conf/.cls.
        data = torch.cat([boxes, confs[:, None], clss[:, None].float()], dim=1)
        result = Results(orig_img=frame, path="", names=self._class_names, boxes=data)

        try:
            if self._tracker is None:
                self._tracker = self._init_tracker()
            # El tracker espera un objeto con .conf/.xywh/.cls indexable:
            # Boxes cumple eso. Results NO expone .conf directamente.
            tracked = self._tracker.update(result.boxes, img=frame)
        except Exception as e:
            # Desactivar ONNX permanentemente para no repetir la inferencia
            # (y el fallback) en cada frame.
            print(f"[WARN] Tracker ONNX falló ({e}) — usando PyTorch desde ahora")
            self._use_onnx = False
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
