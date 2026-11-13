"""
Gestión de presencia de máquinas a partir de las detecciones del tracker.

Módulo independiente de la interfaz: lo reutiliza el motor de análisis
(`src/pipeline/runner.py`) sin cargar Tkinter ni modelos al importarlo.
"""
from __future__ import annotations

from src.core.geometry import BBox, GestorGeofencing
from src.database.db import RepositorioCiclos, RepositorioEventos

# Frames consecutivos sin ver un ID antes de considerarlo desaparecido.
FRAMES_AUSENCIA_DEFAULT = 90


def calcular_timestamp(fps: float, frame_number: int) -> str:
    """Convierte un número de frame a 'HH:MM:SS.mmm'."""
    segundos = frame_number / fps if fps > 0 else 0.0
    horas    = int(segundos // 3600)
    minutos  = int((segundos % 3600) // 60)
    segs     = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:06.3f}"


class GestorPresencia:
    """
    Registra apariciones/desapariciones de máquinas y alimenta el geofencing.

    No gestiona transacciones: recibe un repositorio ya ligado a una sesión.
    La sesión debe pertenecer al mismo hilo que llama a `procesar`.
    """

    def __init__(self, frames_ausencia: int = FRAMES_AUSENCIA_DEFAULT) -> None:
        self.frames_ausencia = frames_ausencia
        self._activos: dict[int, dict] = {}
        self._cache_detecciones: list[dict] = []

    def procesar(
        self,
        detecciones: list[dict],
        frame_count: int,
        repo: RepositorioEventos,
        gestor_geo: GestorGeofencing | None = None,
        repo_ciclos: RepositorioCiclos | None = None,
    ) -> list[str]:
        self._cache_detecciones = detecciones
        mensajes: list[str] = []
        ids_vistos: set[int] = set()

        for det in detecciones:
            tid  = det["track_id"]
            tipo = det["tipo_maquina"]
            ids_vistos.add(tid)

            zona_actual: str | None = None
            if gestor_geo is not None:
                bbox = BBox.desde_yolo(det["bbox"])
                eventos_ciclo = gestor_geo.procesar(
                    maquina_id=det["maquina_id"], tipo_maquina=tipo,
                    bbox=bbox, timestamp=det["timestamp"], frame_number=frame_count,
                )
                for ec in eventos_ciclo:
                    zona_actual = ec.zona_nombre
                    mensajes.append(f"[ZONA] id={tid} | {ec.tipo_evento} -> {ec.zona_nombre}")
                    if repo_ciclos is not None:
                        try:
                            repo_ciclos.registrar_evento(ec)
                        except Exception as e:
                            print(f"[DB ERROR] ciclo: {e}")
                estado_geo = gestor_geo._estados.get(det["maquina_id"])
                if estado_geo:
                    zona_actual = estado_geo.zona_actual

            if tid not in self._activos:
                self._activos[tid] = {
                    "maquina_id":          det["maquina_id"],
                    "tipo_maquina":        tipo,
                    "clase_id":            det["clase_id"],
                    "frame_aparicion":     frame_count,
                    "timestamp_aparicion": det["timestamp"],
                    "ultimo_frame_visto":  frame_count,
                    "ultimo_timestamp":    det["timestamp"],
                    "confianza_max":       det["confianza"],
                }
                repo.insertar({
                    "tipo_maquina": tipo, "clase_id": det["clase_id"],
                    "confianza": det["confianza"], "numero_fotograma": frame_count,
                    "timestamp": det["timestamp"], "tipo_evento_ciclo": "aparicion",
                    "zona_nombre": zona_actual,
                })
                mensajes.append(
                    f"[APARICION]    id={tid} | maquina={tipo} | clase={det['clase_id']} "
                    f"| frame={frame_count} | ts={det['timestamp']} | conf={det['confianza']:.2f}"
                )
            else:
                est = self._activos[tid]
                est["ultimo_frame_visto"] = frame_count
                est["ultimo_timestamp"]   = det["timestamp"]
                if det["confianza"] > est["confianza_max"]:
                    est["confianza_max"] = det["confianza"]

        desaparecidos = [
            tid for tid, est in self._activos.items()
            if tid not in ids_vistos
            and (frame_count - est["ultimo_frame_visto"]) >= self.frames_ausencia
        ]
        for tid in desaparecidos:
            est = self._activos.pop(tid)
            fp  = est["ultimo_frame_visto"] - est["frame_aparicion"]
            repo.insertar({
                "tipo_maquina": est["tipo_maquina"], "clase_id": est["clase_id"],
                "confianza": est["confianza_max"], "numero_fotograma": est["ultimo_frame_visto"],
                "timestamp": est["ultimo_timestamp"], "tipo_evento_ciclo": "desaparicion",
            })
            mensajes.append(
                f"[DESAPARICION] id={tid} | maquina={est['tipo_maquina']} | frames={fp}"
            )
            if gestor_geo is not None:
                gestor_geo.limpiar_maquina(est["maquina_id"])

        return mensajes

    def activos_para_dibujar(self) -> list[dict]:
        ids_activos = set(self._activos.keys())
        return [d for d in self._cache_detecciones if d["track_id"] in ids_activos]

    @property
    def n_activos(self) -> int:
        return len(self._activos)

    def cerrar(
        self,
        repo: RepositorioEventos,
        gestor_geo: GestorGeofencing | None = None,
    ) -> list[str]:
        """Registra desapariciones pendientes y retorna los mensajes generados."""
        mensajes: list[str] = []
        for tid, est in self._activos.items():
            fp = est["ultimo_frame_visto"] - est["frame_aparicion"]
            repo.insertar({
                "tipo_maquina": est["tipo_maquina"], "clase_id": est["clase_id"],
                "confianza": est["confianza_max"], "numero_fotograma": est["ultimo_frame_visto"],
                "timestamp": est["ultimo_timestamp"], "tipo_evento_ciclo": "desaparicion",
            })
            msg = f"[DESAPARICION] id={tid} | maquina={est['tipo_maquina']} | frames={fp}"
            print(msg)
            mensajes.append(msg)
            if gestor_geo is not None:
                gestor_geo.limpiar_maquina(est["maquina_id"])
        self._activos.clear()
        return mensajes
