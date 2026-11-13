"""
clasificador_lstm.py — Orquestador del pipeline LSTM temporal (inferencia en tiempo real).

Combina los tres módulos del sistema en una sola interfaz:
  1. FeatureExtractor  → recorte BGR → vector (576,)
  2. GestorVentanas    → acumula vectores por track_id (ventana deslizante)
  3. LSTMEstados       → secuencia (16, 576) → estado operacional

Diseño:
  - Un ClasificadorLSTM gestiona TODOS los tipos y TODOS los IDs activos.
  - Lazy loading de los modelos .pt por tipo de máquina.
  - Thread-safe con Lock por tipo.
  - Fallback gracioso: si no hay modelo, retorna None sin crashear.
  - Se integra directamente en hilo_inferencia del main.py existente.

Uso desde hilo_inferencia:
    clf = ClasificadorLSTM(device="cpu")
    for det in detecciones:
        estado, conf = clf.procesar(
            frame, det["bbox"], det["tipo_maquina"],
            det["track_id"], frame_idx
        )
        if estado:
            det["estado_lstm"] = estado
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

# Directorio de pesos LSTM entrenados
WEIGHTS_DIR = Path("weights")

# Nombre de archivo por tipo canónico
_MODELO_POR_TIPO: dict[str, str] = {
    "camion":     "lstm_camion.pt",
    "excavadora": "lstm_excavadora.pt",
    "pala":       "lstm_pala.pt",
}

_ALIAS: dict[str, str] = {
    "dump_truck":   "camion",
    "excavator":    "excavadora",
    "wheel_loader": "pala",
    "truck":        "camion",
    "shovel":       "pala",
}

CONF_MIN   = 0.55   # probabilidad mínima para publicar un estado
MARGEN_ROI = 10     # píxeles extra alrededor del bbox


class ClasificadorLSTM:
    """
    Pipeline completo YOLO-crop → features → ventana → LSTM → estado.

    Args:
        device:      'cpu' | 'cuda'.
        weights_dir: Carpeta donde buscar los .pt entrenados.
        window_size: Tamaño de la ventana temporal (frames).
        conf_min:    Confianza mínima para publicar el estado.
    """

    def __init__(
        self,
        device:      str       = "cpu",
        weights_dir: str | Path = WEIGHTS_DIR,
        window_size: int       = 16,
        conf_min:    float     = CONF_MIN,
    ) -> None:
        from src.ai.feature_extractor import FeatureExtractor
        from src.ai.ventana_temporal  import GestorVentanas

        self._device      = self._resolver_device(device)
        self._weights_dir = Path(weights_dir)
        self._conf_min    = conf_min

        # Módulos compartidos para todos los tracks
        self._extractor = FeatureExtractor(device=self._device)
        self._ventanas  = GestorVentanas(window_size=window_size)

        # Modelos LSTM por tipo — carga lazy
        self._modelos:    dict[str, object]         = {}
        self._locks:      dict[str, threading.Lock] = {}
        self._disponible: dict[str, bool]           = {}

        for tipo, nombre in _MODELO_POR_TIPO.items():
            ruta = self._weights_dir / nombre
            self._disponible[tipo] = ruta.exists()
            self._locks[tipo]      = threading.Lock()

        disponibles = [t for t, ok in self._disponible.items() if ok]
        if disponibles:
            print(f"[ClasificadorLSTM] Modelos disponibles: {disponibles}")
        else:
            print(
                "[ClasificadorLSTM] Sin modelos entrenados. "
                "Ejecuta: python entrenar_lstm.py"
            )

    # ── API principal ─────────────────────────────────────────────────────────

    def procesar(
        self,
        frame:     np.ndarray,
        bbox:      list[float],
        tipo:      str,
        track_id:  int,
        frame_num: int,
    ) -> tuple[str | None, float]:
        """
        Procesa un track en el frame actual y retorna (estado, confianza).

        Flujo:
          1. Recorta la ROI del frame.
          2. Extrae vector de características (MobileNetV3).
          3. Añade el vector a la ventana del track_id.
          4. Si la ventana está lista → corre la LSTM.
          5. Si no → retorna el último estado conocido.

        Returns:
            (estado, confianza) o (None, 0.0) si no hay datos suficientes.
        """
        tipo_base = _normalizar_tipo(tipo)

        # 1. Recorte
        roi = self._recortar(frame, bbox)
        if roi is None:
            return self._ventanas.estado_actual(track_id)

        # 2. Features
        vector = self._extractor.extract(roi)

        # 3. Acumular en ventana
        self._ventanas.agregar(track_id, vector, frame_num)

        # 4. ¿Lista para inferir?
        if not self._ventanas.lista(track_id):
            return self._ventanas.estado_actual(track_id)

        if not self._disponible.get(tipo_base, False):
            return self._ventanas.estado_actual(track_id)

        # 5. LSTM
        secuencia = self._ventanas.obtener_secuencia(track_id)
        if secuencia is None:
            return self._ventanas.estado_actual(track_id)

        estado, conf = self._inferir(tipo_base, secuencia)

        if conf >= self._conf_min:
            self._ventanas.marcar_inferencia(track_id, estado, conf)
            return estado, conf
        else:
            # Confianza baja → mantener estado anterior (más estable en pantalla)
            return self._ventanas.estado_actual(track_id)

    def limpiar_track(self, track_id: int) -> None:
        """Llama esto cuando el GestorPresencia registra una desaparición."""
        self._ventanas.limpiar_track(track_id)

    def limpiar_inactivos(self, frame_actual: int) -> None:
        eliminados = self._ventanas.limpiar_inactivos(frame_actual)
        if eliminados:
            print(f"[ClasificadorLSTM] Ventanas purgadas: {eliminados}")

    def todos_estados(self) -> dict[int, str]:
        """Retorna {track_id: estado} para todos los tracks activos."""
        result = {}
        for tid in self._ventanas.ids_activos():
            estado, _ = self._ventanas.estado_actual(tid)
            if estado:
                result[tid] = estado
        return result

    def estadisticas(self) -> dict:
        return {
            "modelos_disponibles": [t for t, ok in self._disponible.items() if ok],
            "device":   self._device,
            "tracks":   len(self._ventanas),
        }

    # ── Internos ──────────────────────────────────────────────────────────────

    def _inferir(self, tipo: str, secuencia: np.ndarray) -> tuple[str, float]:
        modelo = self._cargar_modelo(tipo)
        if modelo is None:
            return "desconocido", 0.0
        try:
            import torch
            tensor = torch.from_numpy(secuencia).float().unsqueeze(0)
            tensor = tensor.to(torch.device(self._device))
            return modelo.predecir(tensor)
        except Exception as e:
            print(f"[ClasificadorLSTM] Error inferencia '{tipo}': {e}")
            return "desconocido", 0.0

    def _cargar_modelo(self, tipo: str):
        """Lazy loading thread-safe."""
        if tipo in self._modelos:
            return self._modelos[tipo]
        with self._locks.get(tipo, threading.Lock()):
            if tipo in self._modelos:
                return self._modelos[tipo]
            ruta = self._weights_dir / _MODELO_POR_TIPO.get(tipo, "")
            if not ruta.exists():
                self._disponible[tipo] = False
                return None
            try:
                from src.ai.lstm_estados import LSTMEstados
                modelo = LSTMEstados.cargar(ruta, device=self._device)
                self._modelos[tipo] = modelo
                return modelo
            except Exception as e:
                print(f"[ClasificadorLSTM] Error cargando {ruta}: {e}")
                self._disponible[tipo] = False
                return None

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

    def _resolver_device(self, device: str) -> str:
        if device == "cuda":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device


def _normalizar_tipo(tipo: str) -> str:
    return _ALIAS.get(tipo.lower(), tipo.lower())
