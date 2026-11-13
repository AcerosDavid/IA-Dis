"""
ventana_temporal.py — Ventana deslizante de vectores de características por track_id.

Problema:
  El tracker asigna IDs únicos. Necesitamos los últimos N vectores de cada ID
  para alimentar la LSTM sin realocar memoria en cada frame.

Solución:
  Un deque(maxlen=N) por track_id. Inserción O(1), acceso O(N).
  Cuando tiene ≥ MIN_FRAMES vectores y han pasado CADA_N_FRAMES desde la
  última inferencia, la ventana está "lista" para correr la LSTM.

Uso:
    gv = GestorVentanas(window_size=16)
    gv.agregar(track_id=3, vector=vec, frame_number=42)
    if gv.lista(3):
        seq = gv.obtener_secuencia(3)   # (16, 576)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

WINDOW_SIZE    = 16    # frames en la ventana  (~3 s a 5 fps de inferencia)
MIN_FRAMES     = 8     # mínimo para primera inferencia
CADA_N_FRAMES  = 5     # reclasificar cada N frames nuevos (ahorra cómputo)
TIMEOUT_FRAMES = 150   # frames sin actividad antes de purgar el track


@dataclass
class _Ventana:
    frames:              deque       = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    ultimo_frame:        int         = 0
    frames_desde_infer:  int         = 0
    ultimo_estado:       str | None  = None
    ultima_confianza:    float       = 0.0


class GestorVentanas:
    """
    Administra una ventana deslizante de vectores por track_id.

    Args:
        window_size:    Tamaño de la ventana (frames que recibe la LSTM).
        min_frames:     Mínimo de frames acumulados para activar inferencia.
        cada_n_frames:  Inferir cada cuántos frames nuevos.
        timeout_frames: Purgar track tras N frames sin actividad.
    """

    def __init__(
        self,
        window_size:    int = WINDOW_SIZE,
        min_frames:     int = MIN_FRAMES,
        cada_n_frames:  int = CADA_N_FRAMES,
        timeout_frames: int = TIMEOUT_FRAMES,
    ) -> None:
        self.window_size    = window_size
        self.min_frames     = min_frames
        self.cada_n_frames  = cada_n_frames
        self.timeout_frames = timeout_frames
        self._ventanas: dict[int, _Ventana] = {}

    # ── API ───────────────────────────────────────────────────────────────────

    def agregar(self, track_id: int, vector: np.ndarray, frame_number: int = 0) -> None:
        """Añade un vector al historial del track_id."""
        if track_id not in self._ventanas:
            self._ventanas[track_id] = _Ventana(frames=deque(maxlen=self.window_size))
        v = self._ventanas[track_id]
        v.frames.append(vector.astype(np.float32))
        v.ultimo_frame       = frame_number
        v.frames_desde_infer += 1

    def lista(self, track_id: int) -> bool:
        """True si hay suficientes frames Y han pasado los N frames de espera."""
        v = self._ventanas.get(track_id)
        if v is None:
            return False
        return len(v.frames) >= self.min_frames and v.frames_desde_infer >= self.cada_n_frames

    def obtener_secuencia(self, track_id: int) -> np.ndarray | None:
        """
        Retorna array (window_size, feature_dim).
        Si hay menos frames que window_size, rellena con ceros por la izquierda.
        """
        v = self._ventanas.get(track_id)
        if v is None or len(v.frames) == 0:
            return None
        frames   = list(v.frames)
        feat_dim = frames[0].shape[0]
        if len(frames) < self.window_size:
            pad    = [np.zeros(feat_dim, dtype=np.float32)] * (self.window_size - len(frames))
            frames = pad + frames
        return np.stack(frames, axis=0)   # (window_size, feature_dim)

    def marcar_inferencia(self, track_id: int, estado: str, confianza: float) -> None:
        """Registra el resultado y reinicia el contador de frames."""
        v = self._ventanas.get(track_id)
        if v:
            v.frames_desde_infer = 0
            v.ultimo_estado      = estado
            v.ultima_confianza   = confianza

    def estado_actual(self, track_id: int) -> tuple[str | None, float]:
        """Último estado conocido sin correr inferencia."""
        v = self._ventanas.get(track_id)
        if v is None:
            return None, 0.0
        return v.ultimo_estado, v.ultima_confianza

    def limpiar_track(self, track_id: int) -> None:
        """Elimina la ventana de un track que desapareció."""
        self._ventanas.pop(track_id, None)

    def limpiar_inactivos(self, frame_actual: int) -> list[int]:
        """Purga tracks sin actividad reciente. Retorna los IDs eliminados."""
        a_borrar = [
            tid for tid, v in self._ventanas.items()
            if (frame_actual - v.ultimo_frame) >= self.timeout_frames
        ]
        for tid in a_borrar:
            del self._ventanas[tid]
        return a_borrar

    def n_frames(self, track_id: int) -> int:
        v = self._ventanas.get(track_id)
        return len(v.frames) if v else 0

    def ids_activos(self) -> list[int]:
        return list(self._ventanas.keys())

    def __len__(self) -> int:
        return len(self._ventanas)

    def __repr__(self) -> str:
        return f"GestorVentanas(window={self.window_size}, tracks={len(self._ventanas)})"
