"""
clasificador_estado.py — Clasificador secundario de estado para maquinaria minera.

Recibe el recorte del bbox detectado por YOLO y determina el estado operativo:
  camion:     empty / loaded
  excavadora: digging / loading / idle
  pala:       scooping / dumping / idle

Diseño:
  - Lazy loading: los modelos se cargan la primera vez que se usan
  - Cache por track_id: no reclasifica la misma máquina en cada frame
  - Fallback gracioso: si no hay modelo entrenado, retorna None sin crashear
  - Thread-safe: usa un lock por tipo de máquina

Uso:
    clf = ClasificadorEstado()
    estado = clf.clasificar(frame, bbox=[x1,y1,x2,y2], tipo="camion", track_id=3, frame_num=150)
    # Retorna "loaded" | "empty" | None
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ── Configuración ─────────────────────────────────────────────────────────────

MODELS_DIR     = Path("models")
IMG_SIZE       = 224
CONF_MIN       = 0.65   # confianza mínima para aceptar clasificación
INTERVALO_FRAMES = 20   # reclasificar cada N frames por track_id
MARGEN_PX      = 15     # píxeles extra alrededor del bbox

# Mapeo tipo YOLO → nombre del modelo
MODELO_POR_TIPO: dict[str, str] = {
    "camion":     "estado_camion.pt",
    "excavadora": "estado_excavadora.pt",
    "pala":       "estado_pala.pt",
    # Aliases
    "dump_truck":   "estado_camion.pt",
    "excavator":    "estado_excavadora.pt",
    "wheel_loader": "estado_pala.pt",
}

# Estados visuales por tipo (detectados por el clasificador de imagen)
# Solo estados que se pueden distinguir en una foto estática
# El movimiento lo detecta el tracker — no es un estado visual
ESTADOS_VISUALES: dict[str, list[str]] = {
    "camion":     ["empty", "loaded", "dumping"],
    "excavadora": ["digging", "loading_truck"],   # sin waiting — tracker lo detecta
    "pala":       ["scooping", "dumping_truck"],  # sin waiting — tracker lo detecta
}

# Estados que indican trabajo activo (para el sistema de ciclos)
ESTADOS_ACTIVOS: dict[str, set[str]] = {
    "camion":     {"loaded", "dumping"},
    "excavadora": {"digging", "loading_truck"},
    "pala":       {"scooping", "dumping_truck"},
}


# ── Estado cacheado por track ─────────────────────────────────────────────────

@dataclass
class _CacheTrack:
    estado:      str | None = None
    confianza:   float      = 0.0
    ultimo_frame: int       = -999
    lecturas:    list[str]  = field(default_factory=list)  # historial para votación

    def actualizar(self, estado: str, confianza: float, frame_num: int) -> str:
        """Acumula lecturas y retorna el estado por votación de mayoría."""
        self.ultimo_frame = frame_num
        self.lecturas.append(estado)
        if len(self.lecturas) > 5:   # ventana deslizante de 5
            self.lecturas.pop(0)

        # Votación: el estado más frecuente
        from collections import Counter
        ganador, votos = Counter(self.lecturas).most_common(1)[0]
        if votos >= 2:   # necesita al menos 2 votos para confirmar
            self.estado    = ganador
            self.confianza = confianza
        return self.estado or estado


# ── Clasificador principal ────────────────────────────────────────────────────

class ClasificadorEstado:
    """
    Clasificador secundario de estado para maquinaria minera.

    Se instancia una sola vez y se reutiliza durante toda la sesión.
    Los modelos se cargan de forma lazy (solo cuando se necesitan).
    """

    def __init__(self, models_dir: str | Path = MODELS_DIR) -> None:
        self._models_dir  = Path(models_dir)
        self._modelos:    dict[str, object]        = {}   # tipo → modelo cargado
        self._clases:     dict[str, list[str]]     = {}   # tipo → lista de clases
        self._cache:      dict[int, _CacheTrack]   = {}   # track_id → cache
        self._locks:      dict[str, threading.Lock] = {}  # tipo → lock
        self._transform   = None   # lazy
        self._disponible: dict[str, bool] = {}    # tipo → si el modelo existe

        # Pre-verificar qué modelos existen
        for tipo, nombre in MODELO_POR_TIPO.items():
            tipo_base = self._tipo_base(tipo)
            if tipo_base not in self._disponible:
                ruta = self._models_dir / nombre
                self._disponible[tipo_base] = ruta.exists()
                self._locks[tipo_base]      = threading.Lock()

        disponibles = [t for t, ok in self._disponible.items() if ok]
        if disponibles:
            print(f"[ClasificadorEstado] Modelos disponibles: {disponibles}")
        else:
            print(f"[ClasificadorEstado] Sin modelos entrenados — "
                  f"ejecuta: python entrenar_clasificador.py")

    # ── API pública ───────────────────────────────────────────────────────────

    def estado_combinado(
        self,
        track_id:    int,
        tipo:        str,
        es_estatico: bool,
    ) -> str | None:
        """
        Combina el estado visual (clasificador) con el movimiento (tracker)
        para obtener el estado operativo final del camión/máquina.

        Tabla de combinación para camión:
          visual=empty  + quieto      → waiting_empty    (esperando turno)
          visual=empty  + moviéndose  → traveling_empty  (regresando a carga)
          visual=loaded + quieto      → being_loaded      (siendo cargado aún)
          visual=loaded + moviéndose  → traveling_loaded  (yendo a descarga)
          visual=dumping + quieto     → dumping           (descargando)

        Para excavadora y pala el movimiento también distingue:
          visual=waiting + moviéndose → traveling
          visual=waiting + quieto     → waiting

        Args:
            track_id:    ID del track.
            tipo:        tipo de máquina.
            es_estatico: True si el tracker detectó que no se mueve.

        Returns:
            Estado combinado o el estado visual si no hay combinación definida.
        """
        estado_visual = self.estado_conocido(track_id)
        if estado_visual is None:
            return None

        tipo_base = self._tipo_base(tipo)
        moviendose = not es_estatico

        # ── Camión ────────────────────────────────────────────────────────────
        if tipo_base == "camion":
            if estado_visual == "empty":
                return "traveling_empty"  if moviendose else "waiting_empty"
            elif estado_visual == "loaded":
                return "traveling_loaded" if moviendose else "being_loaded"
            elif estado_visual == "dumping":
                return "dumping"   # siempre quieto al descargar

        # ── Excavadora ────────────────────────────────────────────────────────
        elif tipo_base == "excavadora":
            if moviendose:
                return "traveling"

        # ── Pala ──────────────────────────────────────────────────────────────
        elif tipo_base == "pala":
            if moviendose:
                return "traveling"

        return estado_visual   # fallback al estado visual puro

    def clasificar(
        self,
        frame:      np.ndarray,
        bbox:       list[float],
        tipo:       str,
        track_id:   int,
        frame_num:  int,
    ) -> str | None:
        """
        Clasifica el estado de una máquina a partir del recorte de su bbox.

        Args:
            frame:     frame BGR completo del video.
            bbox:      [x1, y1, x2, y2] en píxeles del frame original.
            tipo:      tipo de máquina ("camion", "excavadora", "pala", etc.)
            track_id:  ID del track (para cache).
            frame_num: número de frame actual.

        Returns:
            Estado clasificado ("loaded", "empty", "digging", etc.) o None
            si no hay modelo disponible o la confianza es baja.
        """
        tipo_base = self._tipo_base(tipo)

        if not self._disponible.get(tipo_base, False):
            return None

        # ── Cache: no reclasificar demasiado seguido ─────────────────────────
        cache = self._cache.setdefault(track_id, _CacheTrack())
        if frame_num - cache.ultimo_frame < INTERVALO_FRAMES:
            return cache.estado

        # ── Recortar ROI ──────────────────────────────────────────────────────
        roi = self._recortar_roi(frame, bbox)
        if roi is None:
            return cache.estado

        # ── Inferencia ────────────────────────────────────────────────────────
        modelo, clases = self._cargar_modelo(tipo_base)
        if modelo is None:
            return None

        try:
            import torch
            tensor = self._preprocesar(roi)
            with torch.no_grad():
                logits  = modelo(tensor)
                probs   = torch.softmax(logits, dim=1)[0]
                conf, idx = probs.max(0)
                confianza = float(conf)
                estado    = clases[int(idx)]

            if confianza < CONF_MIN:
                return cache.estado   # confianza baja → mantener estado anterior

            return cache.actualizar(estado, confianza, frame_num)

        except Exception as e:
            # No crashear si hay error — el sistema sigue sin clasificación
            return cache.estado

    def estado_conocido(self, track_id: int) -> str | None:
        """Retorna el último estado confirmado sin correr inferencia."""
        return self._cache.get(track_id, _CacheTrack()).estado

    def es_activo(self, track_id: int, tipo: str) -> bool:
        """
        Retorna True si la máquina está en un estado de trabajo activo.
        Útil para el sistema de ciclos: si un camión está loaded probablemente
        acaba de cargar.
        """
        estado    = self.estado_conocido(track_id)
        tipo_base = self._tipo_base(tipo)
        if estado is None:
            return False
        return estado in ESTADOS_ACTIVOS.get(tipo_base, set())

    def limpiar_track(self, track_id: int) -> None:
        """Elimina el cache de un track que desapareció."""
        self._cache.pop(track_id, None)

    def todos_estados(self) -> dict[int, str]:
        """Retorna todos los estados conocidos {track_id: estado}."""
        return {tid: c.estado for tid, c in self._cache.items() if c.estado}

    def estadisticas(self) -> dict:
        """Retorna estadísticas del clasificador."""
        return {
            "modelos_disponibles": [t for t, ok in self._disponible.items() if ok],
            "tracks_activos":      len(self._cache),
            "estados_actuales":    self.todos_estados(),
        }

    # ── Internos ──────────────────────────────────────────────────────────────

    def _tipo_base(self, tipo: str) -> str:
        """Normaliza alias al tipo base (camion/excavadora/pala)."""
        mapa = {
            "dump_truck":   "camion",
            "excavator":    "excavadora",
            "wheel_loader": "pala",
        }
        return mapa.get(tipo, tipo)

    def _recortar_roi(
        self,
        frame: np.ndarray,
        bbox:  list[float],
    ) -> np.ndarray | None:
        """Recorta el bbox con margen extra para dar contexto al clasificador."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1 - MARGEN_PX)
        y1 = max(0, y1 - MARGEN_PX)
        x2 = min(w, x2 + MARGEN_PX)
        y2 = min(h, y2 + MARGEN_PX)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi

    def _preprocesar(self, roi: np.ndarray):
        """BGR → RGB → tensor normalizado."""
        import torch
        from torchvision import transforms

        if self._transform is None:
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])

        rgb    = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0)  # (1, 3, H, W)
        return tensor

    def _cargar_modelo(self, tipo: str):
        """
        Carga el modelo de forma lazy y thread-safe.
        Retorna (modelo, clases) o (None, None) si no está disponible.
        """
        if tipo in self._modelos:
            return self._modelos[tipo], self._clases[tipo]

        lock = self._locks.get(tipo)
        if lock is None:
            return None, None

        with lock:
            # Double-check dentro del lock
            if tipo in self._modelos:
                return self._modelos[tipo], self._clases[tipo]

            nombre_archivo = MODELO_POR_TIPO.get(tipo)
            if nombre_archivo is None:
                return None, None

            ruta = self._models_dir / nombre_archivo
            if not ruta.exists():
                self._disponible[tipo] = False
                return None, None

            try:
                import torch
                from torchvision import models

                ckpt  = torch.load(str(ruta), map_location="cpu", weights_only=False)
                clases = ckpt["clases"]
                n_cls  = len(clases)

                # Recrear arquitectura
                modelo = models.mobilenet_v3_small(weights=None)
                import torch.nn as nn
                in_features = modelo.classifier[-1].in_features
                modelo.classifier[-1] = nn.Linear(in_features, n_cls)
                modelo.load_state_dict(ckpt["modelo_state"])
                modelo.eval()

                self._modelos[tipo] = modelo
                self._clases[tipo]  = clases

                print(
                    f"[ClasificadorEstado] {tipo} cargado: "
                    f"clases={clases} acc={ckpt.get('acc_val', 0):.3f}"
                )
                return modelo, clases

            except Exception as e:
                print(f"[ClasificadorEstado] Error cargando {ruta}: {e}")
                self._disponible[tipo] = False
                return None, None
