"""
Modulo de deteccion y tracking de maquinaria minera usando YOLO26 + ByteTrack.

Responsabilidad: ejecutar inferencia/tracking YOLO sobre un fotograma
y retornar detecciones validas con ID de tracking persistente.
"""
from __future__ import annotations

import numpy as np

# Mapeo de clases YOLO a nombres del dominio minero
CLASE_MAP: dict[str, str] = {
    # Dataset equipment.yolo26 (3 clases)
    "dump_truck":   "camion",
    "excavator":    "excavadora",
    "wheel_loader": "pala",
    # Dataset anterior (por compatibilidad)
    "dump":         "camion",
    # Clases COCO genericas (yolo26n.pt pretrained)
    "truck":        "camion",
}

# ID centinela cuando ByteTrack no ha asignado IDs aun
_TRACK_ID_PENDIENTE = -1


class Detector:
    """
    Detector y tracker de maquinaria minera usando YOLO26 + ByteTrack.

    Modos de operacion:
      - track_frame(): inferencia + tracking (ID persistente por maquina)
      - detect_frame(): inferencia simple sin tracking (compatible legacy)

    Attributes:
        model_path:           ruta al archivo .pt del modelo YOLO26
        confidence_threshold: umbral minimo de confianza
        model:                instancia YOLO (cargada con load_model())
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5) -> None:
        self.model_path           = model_path
        self.confidence_threshold = confidence_threshold
        self.model                = None

    def load_model(self) -> None:
        """
        Carga el modelo YOLO26. Falla rapido si la ruta no existe.

        Raises:
            SystemExit: si el modelo no puede cargarse.
        """
        import os
        from ultralytics import YOLO

        try:
            self.model = YOLO(self.model_path)
            device     = "cuda" if self._cuda_available() else "cpu"
            nombre     = os.path.basename(self.model_path)
            print(f"[INFO] Modelo {nombre} cargado en {device}")
        except Exception as e:
            print(f"[MODEL ERROR] No se pudo cargar el modelo: {self.model_path} | {e}")
            raise SystemExit(1) from e

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def track_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
        imgsz: int = 320,
    ) -> list[dict]:
        """
        Ejecuta inferencia + ByteTrack sobre un fotograma.
        Asigna un ID persistente a cada maquina mientras permanece en escena.

        Si ByteTrack aun no ha asignado IDs (primeros frames), las detecciones
        sin ID se omiten para evitar colisiones silenciosas entre objetos distintos.

        Args:
            frame:        imagen BGR (numpy array OpenCV).
            frame_number: indice del fotograma (>= 0).
            timestamp:    "HH:MM:SS.mmm".
            imgsz:        tamano de imagen para inferencia (px).

        Returns:
            Lista de dicts:
            {
                "maquina_id":       str,   # "camion_4", "pala_7" (ID ByteTrack)
                "track_id":         int,   # ID numerico ByteTrack (>= 0)
                "tipo_maquina":     str,   # "camion" | "pala"
                "clase_id":         str,   # nombre clase YOLO original
                "confianza":        float,
                "numero_fotograma": int,
                "timestamp":        str,
                "bbox":             list,  # [x1, y1, x2, y2] en pixeles
            }
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama load_model() primero.")

        resultados = self.model.track(
            frame,
            persist=True,   # mantiene IDs entre frames
            verbose=False,
            imgsz=imgsz,
            conf=self.confidence_threshold,
        )

        detecciones: list[dict] = []

        for resultado in resultados:
            if resultado.boxes is None:
                continue

            boxes = resultado.boxes
            ids   = boxes.id  # tensor de IDs o None

            # Si el tracker aun no asigno IDs, omitir este frame completo
            # para evitar que varios objetos compartan track_id=0
            if ids is None:
                continue

            for i, box in enumerate(boxes):
                clase_nombre = resultado.names[int(box.cls)]

                if clase_nombre not in CLASE_MAP:
                    continue

                tipo       = CLASE_MAP[clase_nombre]
                track_id   = int(ids[i])
                maquina_id = f"{tipo}_{track_id}"
                confianza  = float(box.conf)
                xyxy       = box.xyxy[0].tolist()

                detecciones.append({
                    "maquina_id":       maquina_id,
                    "track_id":         track_id,
                    "tipo_maquina":     tipo,
                    "clase_id":         clase_nombre,
                    "confianza":        confianza,
                    "numero_fotograma": frame_number,
                    "timestamp":        timestamp,
                    "bbox":             xyxy,
                })

        return detecciones

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
    ) -> list[dict]:
        """
        Inferencia simple sin tracking (modo legacy, sin IDs).
        Mantiene compatibilidad con los tests existentes.

        Retorna dicts con campos: tipo_maquina, clase_id, confianza,
        numero_fotograma, timestamp. No incluye bbox ni track_id.
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama load_model() primero.")

        resultados = self.model(frame, verbose=False, conf=self.confidence_threshold)
        eventos: list[dict] = []

        for resultado in resultados:
            if resultado.boxes is None:
                continue
            for box in resultado.boxes:
                clase_nombre = resultado.names[int(box.cls)]
                confianza    = float(box.conf)

                if clase_nombre not in CLASE_MAP:
                    continue
                # Filtrado manual necesario: en tests el mock no respeta conf=
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
