"""
feature_extractor.py — Extracción de vectores de características desde recortes de máquinas.

Convierte un recorte (crop) BGR de la ROI de una máquina en un vector numérico
1-D que describe su apariencia en ese instante. Ese vector es la "entrada por frame"
que la LSTM recibe en secuencia para inferir el estado operacional.

Backbone: MobileNetV3-Small pretrained en ImageNet (sin cabeza clasificadora).
Salida  : vector float32 de 576 dimensiones, normalizado L2.

Diseño:
  - Lazy loading: el backbone se carga en el primer uso, no en el import.
  - Caché de transformación: se construye una sola vez.
  - Thread-safe: protegido con Lock para usarse desde múltiples hilos.
  - Fallback: si PyTorch no está, retorna vector de ceros sin crashear.

Uso:
    extractor = FeatureExtractor(device="cpu")
    vec = extractor.extract(roi_bgr)   # np.ndarray (576,)
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

IMG_SIZE    = 128   # resolución interna — 128×128 equilibra velocidad y calidad
FEATURE_DIM = 576   # dim de salida del backbone MobileNetV3-Small


class FeatureExtractor:
    """
    Extrae un vector de características denso a partir de un recorte BGR.

    Args:
        device: 'cpu' | 'cuda'. Si 'cuda' no está disponible se usa CPU.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device_str = device
        self.feature_dim = FEATURE_DIM
        self._model      = None   # lazy
        self._transform  = None   # lazy
        self._device     = None   # lazy
        self._lock       = threading.Lock()

    # ── API pública ───────────────────────────────────────────────────────────

    def extract(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        BGR crop → vector float32 (576,) normalizado L2.
        Retorna vector de ceros si el recorte es inválido o PyTorch no está disponible.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        self._inicializar()

        try:
            import torch
            tensor = self._preprocesar(roi_bgr).to(self._device)   # (1, 3, H, W)
            with torch.no_grad():
                feat = self._model(tensor)                          # (1, C, 1, 1)
                feat = feat.view(feat.size(0), -1)                  # (1, C)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            return np.zeros(self.feature_dim, dtype=np.float32)

    # ── Internos ──────────────────────────────────────────────────────────────

    def _inicializar(self) -> None:
        """Carga el backbone la primera vez, thread-safe."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            device = torch.device(
                "cuda" if self._device_str == "cuda" and torch.cuda.is_available()
                else "cpu"
            )
            self._device = device

            # MobileNetV3-Small: solo features + avgpool → (B, 576, 1, 1)
            base     = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            backbone = nn.Sequential(base.features, base.avgpool)
            backbone.to(device).eval()
            self._model = backbone

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def _preprocesar(self, roi_bgr: np.ndarray):
        """BGR ndarray → tensor (1, 3, H, W)."""
        h, w = roi_bgr.shape[:2]
        if h < 10 or w < 10:
            roi_bgr = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        return self._transform(rgb).unsqueeze(0)

    def __repr__(self) -> str:
        estado = "cargado" if self._model is not None else "lazy"
        return f"FeatureExtractor(device={self._device_str}, dim={self.feature_dim}, {estado})"
