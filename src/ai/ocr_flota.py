"""
ocr_flota.py — Lectura de números de flota en maquinaria minera.

Estrategia:
  1. Recortar el ROI superior del bbox del camión (donde suele estar el número)
  2. Pre-procesar la imagen para mejorar contraste y nitidez
  3. Correr EasyOCR filtrando solo dígitos
  4. Cachear el resultado por track_id para no re-leer en cada frame

Rendimiento:
  - OCR corre solo cada OCR_INTERVAL frames por track_id (~cada 3s a 5fps)
  - El lector EasyOCR se inicializa una sola vez (lazy, primer uso)
  - En CPU: ~150-300ms por ROI → se corre en el hilo de inferencia

Uso:
    ocr = LectorFlota()
    numero = ocr.leer(frame, bbox=[x1,y1,x2,y2], track_id=3, frame_number=150)
    # Retorna "C-47" | "047" | None si no detecta nada legible
"""
from __future__ import annotations

import re
import time
import cv2
import numpy as np

# Intervalo de frames entre lecturas OCR por track_id
OCR_INTERVAL = 25   # ~5s a 5fps de inferencia

# Confianza mínima de EasyOCR para aceptar el texto
OCR_MIN_CONF = 0.55

# Fracción del bbox que se recorta para buscar el número
# 0.0-0.35 = tercio superior (cabina del camión, donde suele estar el número)
ROI_Y_INICIO = 0.0
ROI_Y_FIN    = 0.40

# Ampliar el ROI horizontalmente para no cortar números en los bordes
ROI_X_MARGEN = 0.10   # 10% extra a cada lado

# Regex: acepta secuencias de 2-5 dígitos, con guión o letra opcional
# Ejemplos válidos: "047", "C-47", "CA-12", "1047", "C047"
_PATRON_NUMERO = re.compile(r"[A-Z]{0,2}[-]?\d{2,5}", re.IGNORECASE)


def _preprocesar_roi(roi: np.ndarray) -> np.ndarray:
    """
    Pre-procesa el ROI para mejorar la legibilidad de los números.
    Pipeline: escalar → grises → CLAHE → umbral adaptativo → dilatar
    """
    # Escalar a altura fija para OCR consistente
    h, w = roi.shape[:2]
    if h < 40:
        scale = 60 / h
        roi = cv2.resize(roi, (int(w * scale), 60), interpolation=cv2.INTER_CUBIC)

    # Convertir a grises
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE: mejora contraste local (muy útil en escenas con polvo/sombra)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # Umbral adaptativo — robusto a iluminación variable
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )

    # Dilatar ligeramente para engordar los trazos de los números
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Volver a BGR para EasyOCR
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


class LectorFlota:
    """
    Lee números de flota de maquinaria minera usando EasyOCR.

    Diseñado para ser instanciado una sola vez y reutilizado
    durante toda la sesión de video.
    """

    def __init__(self) -> None:
        self._reader        = None   # lazy init — carga el modelo la 1ª vez
        self._cache:  dict[int, str | None] = {}     # track_id → número leído
        self._ultimo: dict[int, int]         = {}    # track_id → último frame OCR
        self._disponible: bool | None        = None  # None=sin intentar

    def _init_reader(self) -> bool:
        """Inicializa EasyOCR (descarga modelos en la 1ª ejecución ~30s)."""
        if self._disponible is not None:
            return self._disponible
        try:
            import easyocr
            print("[OCR] Iniciando EasyOCR (puede tardar ~30s la 1ª vez)…")
            t0 = time.perf_counter()
            # Solo inglés + español para números; gpu=False para CPU
            self._reader = easyocr.Reader(
                ["en"],
                gpu=self._gpu_disponible(),
                verbose=False,
            )
            print(f"[OCR] EasyOCR listo en {time.perf_counter()-t0:.1f}s")
            self._disponible = True
        except Exception as e:
            print(f"[OCR] EasyOCR no disponible: {e}")
            print("[OCR] Instala con: pip install easyocr")
            self._disponible = False
        return self._disponible

    def _gpu_disponible(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def leer(
        self,
        frame:        np.ndarray,
        bbox:         list[float],
        track_id:     int,
        frame_number: int,
    ) -> str | None:
        """
        Lee el número de flota del camión en el bbox dado.

        Args:
            frame:        frame BGR completo del video.
            bbox:         [x1, y1, x2, y2] en píxeles del frame original.
            track_id:     ID del track (para cachear resultado).
            frame_number: número de frame actual.

        Returns:
            String con el número leído (ej: "C-47", "047") o None si no
            se detecta texto numérico válido con suficiente confianza.
        """
        # Usar cache si el track fue leído recientemente
        ultimo = self._ultimo.get(track_id, -OCR_INTERVAL - 1)
        if frame_number - ultimo < OCR_INTERVAL:
            return self._cache.get(track_id)

        # Marcar como intentado aunque falle
        self._ultimo[track_id] = frame_number

        if not self._init_reader():
            return None

        # ── Recortar ROI ──────────────────────────────────────────────────────
        x1, y1, x2, y2 = map(int, bbox)
        h_bbox = y2 - y1
        w_bbox = x2 - x1

        # Margen horizontal para no cortar números en los bordes
        margen_x = int(w_bbox * ROI_X_MARGEN)
        rx1 = max(0, x1 - margen_x)
        rx2 = min(frame.shape[1], x2 + margen_x)

        # Zona vertical: tercio superior del bbox
        ry1 = y1 + int(h_bbox * ROI_Y_INICIO)
        ry2 = y1 + int(h_bbox * ROI_Y_FIN)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            self._cache[track_id] = None
            return None

        # ── Pre-procesar ──────────────────────────────────────────────────────
        roi_proc = _preprocesar_roi(roi)

        # ── OCR ───────────────────────────────────────────────────────────────
        try:
            resultados = self._reader.readtext(
                roi_proc,
                detail=1,          # devuelve (bbox, texto, confianza)
                paragraph=False,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-",
                min_size=10,
                text_threshold=OCR_MIN_CONF,
                low_text=0.3,
                width_ths=0.8,
            )
        except Exception as e:
            print(f"[OCR] Error en track_id={track_id}: {e}")
            self._cache[track_id] = None
            return None

        # ── Filtrar y seleccionar el mejor resultado ──────────────────────────
        candidatos: list[tuple[float, str]] = []

        for (_bbox_ocr, texto, conf) in resultados:
            texto_clean = texto.strip().upper().replace(" ", "")
            if conf < OCR_MIN_CONF:
                continue
            # Verificar que parece un número de flota
            if _PATRON_NUMERO.search(texto_clean):
                candidatos.append((conf, texto_clean))

        if not candidatos:
            self._cache[track_id] = None
            return None

        # Tomar el de mayor confianza
        candidatos.sort(key=lambda x: x[0], reverse=True)
        numero = candidatos[0][1]

        # Actualizar cache
        prev = self._cache.get(track_id)
        if prev and prev != numero:
            # Si ya teníamos un número diferente, quedarnos con el anterior
            # (los números de flota no cambian — evitar lecturas erróneas)
            self._cache[track_id] = prev
            return prev

        self._cache[track_id] = numero
        print(f"[OCR] Flota detectada: track_id={track_id} → '{numero}' (conf={candidatos[0][0]:.2f})")
        return numero

    def limpiar_track(self, track_id: int) -> None:
        """Elimina el cache de un track que desapareció."""
        self._cache.pop(track_id, None)
        self._ultimo.pop(track_id, None)

    def numero_conocido(self, track_id: int) -> str | None:
        """Retorna el número cacheado sin correr OCR."""
        return self._cache.get(track_id)

    def todos_conocidos(self) -> dict[int, str]:
        """Retorna todos los números de flota conocidos {track_id: numero}."""
        return {k: v for k, v in self._cache.items() if v is not None}
