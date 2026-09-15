"""
Modulo de geometria y geofencing para operaciones mineras.

Detecta cuando una maquina entra o sale de zonas virtuales definidas
usando las coordenadas de los bounding boxes de YOLO26.

Conceptos:
  - Zona:    poligono o rectangulo que representa un area operativa
             (zona de carga, zona de descarga, parqueo, etc.)
  - BBox:    bounding box de YOLO [x1, y1, x2, y2] en pixeles
  - Centroide: punto central del bounding box — se usa para
               determinar si una maquina esta dentro de una zona

Uso basico:
    zona_carga = Zona("carga", [(100,100), (400,100), (400,300), (100,300)])
    bbox = BBox(x1=150, y1=120, x2=250, y2=250)
    if zona_carga.contiene(bbox.centroide):
        print("Camion en zona de carga")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


# ── Tipos basicos ─────────────────────────────────────────────────────────────

class Punto(NamedTuple):
    """Coordenada 2D en pixeles."""
    x: float
    y: float


class BBox(NamedTuple):
    """
    Bounding box de una deteccion YOLO.
    Coordenadas en pixeles: esquina superior-izquierda (x1,y1) y
    esquina inferior-derecha (x2,y2).
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroide(self) -> Punto:
        """Punto central del bounding box."""
        return Punto(
            x=(self.x1 + self.x2) / 2,
            y=(self.y1 + self.y2) / 2,
        )

    @property
    def ancho(self) -> float:
        return self.x2 - self.x1

    @property
    def alto(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.ancho * self.alto

    @classmethod
    def desde_yolo(cls, xyxy: list | tuple) -> "BBox":
        """Crea un BBox desde el formato xyxy de Ultralytics."""
        return cls(x1=float(xyxy[0]), y1=float(xyxy[1]),
                   x2=float(xyxy[2]), y2=float(xyxy[3]))


# ── Zona virtual ──────────────────────────────────────────────────────────────

@dataclass
class Zona:
    """
    Zona operativa definida por un poligono convexo.

    Attributes:
        nombre:    identificador de la zona (ej: "zona_carga_1")
        tipo:      tipo operativo: "carga" | "descarga" | "parqueo" | "transito"
        vertices:  lista de puntos (x, y) que definen el poligono en orden
        color_bgr: color para visualizacion OpenCV (B, G, R)
        activa:    si la zona esta activa para deteccion
    """
    nombre:    str
    tipo:      str                               # "carga"|"descarga"|"parqueo"|"transito"
    vertices:  list[tuple[float, float]]
    color_bgr: tuple[int, int, int] = (0, 255, 255)   # amarillo por defecto
    activa:    bool = True

    # Tipos validos de zona
    TIPOS_VALIDOS = {"carga", "descarga", "parqueo", "transito"}

    def __post_init__(self) -> None:
        if self.tipo not in self.TIPOS_VALIDOS:
            raise ValueError(f"Tipo de zona invalido: {self.tipo!r}. "
                             f"Debe ser uno de: {self.TIPOS_VALIDOS}")
        if len(self.vertices) < 3:
            raise ValueError("Una zona necesita al menos 3 vertices.")

    def contiene(self, punto: Punto) -> bool:
        """
        Determina si un punto esta dentro del poligono usando
        el algoritmo ray-casting.

        Args:
            punto: coordenada (x, y) a evaluar.

        Returns:
            True si el punto esta dentro o en el borde del poligono.
        """
        x, y = punto.x, punto.y
        n = len(self.vertices)
        dentro = False
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                dentro = not dentro
            j = i
        return dentro

    def contiene_bbox(self, bbox: BBox, modo: str = "centroide") -> bool:
        """
        Determina si un bounding box esta dentro de la zona.

        Args:
            bbox:  bounding box de la deteccion.
            modo:  "centroide" (default) — usa el centro del bbox
                   "parcial"   — True si cualquier esquina esta dentro
                   "completo"  — True solo si todo el bbox esta dentro

        Returns:
            True si la maquina esta en la zona segun el modo elegido.
        """
        if modo == "centroide":
            return self.contiene(bbox.centroide)

        esquinas = [
            Punto(bbox.x1, bbox.y1),
            Punto(bbox.x2, bbox.y1),
            Punto(bbox.x2, bbox.y2),
            Punto(bbox.x1, bbox.y2),
        ]
        resultados = [self.contiene(e) for e in esquinas]

        if modo == "completo":
            return all(resultados)
        elif modo == "parcial":
            return any(resultados)
        else:
            raise ValueError(f"Modo invalido: {modo!r}. Usa 'centroide', 'parcial' o 'completo'.")

    @classmethod
    def rectangulo(
        cls,
        nombre: str,
        tipo: str,
        x1: float, y1: float,
        x2: float, y2: float,
        color_bgr: tuple[int, int, int] = (0, 255, 255),
    ) -> "Zona":
        """
        Crea una zona rectangular (atajo conveniente).

        Args:
            nombre, tipo: identificacion de la zona.
            x1, y1: esquina superior-izquierda.
            x2, y2: esquina inferior-derecha.
        """
        return cls(
            nombre=nombre,
            tipo=tipo,
            vertices=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            color_bgr=color_bgr,
        )


# ── Detector de eventos de ciclo ──────────────────────────────────────────────

@dataclass
class EstadoMaquina:
    """
    Seguimiento del estado de una maquina individual entre frames.

    Attributes:
        maquina_id:    identificador unico (ej: "camion_0", "pala_1")
        tipo_maquina:  "camion" | "pala"
        zona_actual:   nombre de la zona donde esta ahora (o None)
        zona_anterior: nombre de la zona donde estaba antes (o None)
        frames_en_zona: cuantos frames consecutivos lleva en la zona actual
    """
    maquina_id:     str
    tipo_maquina:   str
    zona_actual:    str | None = None
    zona_anterior:  str | None = None
    frames_en_zona: int = 0


@dataclass
class EventoCiclo:
    """
    Evento operativo generado cuando una maquina entra o sale de una zona.

    Attributes:
        maquina_id:   identificador de la maquina
        tipo_maquina: "camion" | "pala"
        tipo_evento:  "entrada_carga" | "salida_carga" |
                      "entrada_descarga" | "salida_descarga" |
                      "entrada_parqueo" | "salida_parqueo"
        zona_nombre:  nombre de la zona involucrada
        timestamp:    marca de tiempo del evento
        frame_number: numero de fotograma
        bbox:         bounding box en el momento del evento
    """
    maquina_id:   str
    tipo_maquina: str
    tipo_evento:  str
    zona_nombre:  str
    timestamp:    str
    frame_number: int
    bbox:         BBox


class GestorGeofencing:
    """
    Gestiona las zonas virtuales y detecta eventos de ciclo
    cuando las maquinas entran o salen de ellas.

    Uso:
        gestor = GestorGeofencing()
        gestor.agregar_zona(Zona.rectangulo("carga_norte", "carga", 100, 100, 400, 300))

        # En cada frame, para cada deteccion:
        eventos = gestor.procesar(
            maquina_id="camion_0",
            tipo_maquina="camion",
            bbox=BBox.desde_yolo(box.xyxy[0]),
            timestamp="00:01:23.456",
            frame_number=1234,
        )
        for evento in eventos:
            print(evento.tipo_evento, evento.zona_nombre)
    """

    # Minimo de frames consecutivos dentro de una zona para confirmar entrada
    # Evita falsos positivos por detecciones inestables
    FRAMES_MINIMOS_CONFIRMACION = 3

    def __init__(self) -> None:
        self._zonas:   dict[str, Zona]          = {}
        self._estados: dict[str, EstadoMaquina] = {}

    def agregar_zona(self, zona: Zona) -> None:
        """Registra una zona virtual en el gestor."""
        self._zonas[zona.nombre] = zona

    def eliminar_zona(self, nombre: str) -> None:
        """Elimina una zona virtual por nombre."""
        self._zonas.pop(nombre, None)

    @property
    def zonas(self) -> dict[str, Zona]:
        """Diccionario de zonas registradas (solo lectura)."""
        return dict(self._zonas)

    def _zona_para_bbox(self, bbox: BBox) -> str | None:
        """Retorna el nombre de la primera zona activa que contiene el bbox, o None."""
        for zona in self._zonas.values():
            if zona.activa and zona.contiene_bbox(bbox):
                return zona.nombre
        return None

    def procesar(
        self,
        maquina_id:   str,
        tipo_maquina: str,
        bbox:         BBox,
        timestamp:    str,
        frame_number: int,
    ) -> list[EventoCiclo]:
        """
        Procesa la posicion de una maquina en un frame y genera
        EventoCiclo si entra o sale de alguna zona.

        Args:
            maquina_id:   identificador unico de la maquina en este frame.
            tipo_maquina: "camion" | "pala".
            bbox:         bounding box de la deteccion.
            timestamp:    timestamp del frame.
            frame_number: indice del frame.

        Returns:
            Lista de EventoCiclo generados (vacia si no hay cambios de zona).
        """
        eventos: list[EventoCiclo] = []

        # Obtener o crear estado de la maquina
        if maquina_id not in self._estados:
            self._estados[maquina_id] = EstadoMaquina(
                maquina_id=maquina_id,
                tipo_maquina=tipo_maquina,
            )
        estado = self._estados[maquina_id]

        # Zona donde esta ahora el centroide del bbox
        zona_detectada = self._zona_para_bbox(bbox)

        if zona_detectada == estado.zona_actual:
            # Sigue en la misma zona — incrementar contador
            estado.frames_en_zona += 1
        else:
            # Cambio de zona detectado
            if zona_detectada is not None:
                estado.frames_en_zona += 1
                # Confirmar entrada solo si lleva suficientes frames
                if estado.frames_en_zona >= self.FRAMES_MINIMOS_CONFIRMACION:
                    # Evento de salida de zona anterior (si habia)
                    if estado.zona_actual is not None:
                        tipo_salida = f"salida_{self._zonas[estado.zona_actual].tipo}"
                        eventos.append(EventoCiclo(
                            maquina_id=maquina_id,
                            tipo_maquina=tipo_maquina,
                            tipo_evento=tipo_salida,
                            zona_nombre=estado.zona_actual,
                            timestamp=timestamp,
                            frame_number=frame_number,
                            bbox=bbox,
                        ))
                    # Evento de entrada a nueva zona
                    tipo_entrada = f"entrada_{self._zonas[zona_detectada].tipo}"
                    eventos.append(EventoCiclo(
                        maquina_id=maquina_id,
                        tipo_maquina=tipo_maquina,
                        tipo_evento=tipo_entrada,
                        zona_nombre=zona_detectada,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        bbox=bbox,
                    ))
                    estado.zona_anterior = estado.zona_actual
                    estado.zona_actual   = zona_detectada
            else:
                # Salio de todas las zonas
                if estado.zona_actual is not None:
                    tipo_salida = f"salida_{self._zonas[estado.zona_actual].tipo}"
                    eventos.append(EventoCiclo(
                        maquina_id=maquina_id,
                        tipo_maquina=tipo_maquina,
                        tipo_evento=tipo_salida,
                        zona_nombre=estado.zona_actual,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        bbox=bbox,
                    ))
                estado.zona_anterior  = estado.zona_actual
                estado.zona_actual    = None
                estado.frames_en_zona = 0

        return eventos


# ── Utilidades de visualizacion ───────────────────────────────────────────────

def dibujar_zonas(frame, gestor: GestorGeofencing) -> None:
    """
    Dibuja todas las zonas activas sobre un frame de OpenCV (in-place).

    Args:
        frame:  imagen BGR de OpenCV (numpy array).
        gestor: instancia de GestorGeofencing con las zonas registradas.
    """
    import cv2
    import numpy as np

    for zona in gestor.zonas.values():
        if not zona.activa:
            continue
        pts = np.array(zona.vertices, dtype=np.int32)
        # Relleno semitransparente
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], zona.color_bgr)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        # Borde
        cv2.polylines(frame, [pts], isClosed=True, color=zona.color_bgr, thickness=2)
        # Etiqueta
        cx = int(sum(v[0] for v in zona.vertices) / len(zona.vertices))
        cy = int(sum(v[1] for v in zona.vertices) / len(zona.vertices))
        cv2.putText(frame, f"{zona.nombre} [{zona.tipo}]",
                    (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, zona.color_bgr, 1, cv2.LINE_AA)
