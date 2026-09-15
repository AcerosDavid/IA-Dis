# Design Document

## Overview

El **Módulo 1 – MVP de Detección de Maquinaria** es una prueba de concepto de línea de comandos que valida el pipeline completo: ingesta de video → detección YOLO → persistencia SQLite → consulta via API REST.

El sistema opera **exclusivamente en modo consola/terminal**. No existe interfaz gráfica. El usuario ejecuta `python main.py` desde la raíz del proyecto, y el sistema procesa el video de forma secuencial, imprimiendo eventos en tiempo real y exponiendo una API FastAPI para consulta posterior.

**Alcance del MVP:**
- Procesamiento de un único video local a la vez (`.mp4`, `.avi`, `.mov`)
- Detección de dos tipos de maquinaria: `camion` (mapeado desde `truck`) y `pala` (mapeado desde `excavator`)
- Modelo placeholder: YOLO26n pretrained (`yolo26n.pt`) hasta que `best.pt` esté entrenado
- Dataset de entrenamiento: Roboflow (descarga y colocación manual por el usuario en `data/processed/`)
- Persistencia local en SQLite (sin servidor de base de datos externo)
- API REST minimalista sin autenticación ni UI personalizada

**Fuera de alcance:**
- Procesamiento en tiempo real de múltiples cámaras
- Interfaz gráfica
- Integración con sistemas externos
- Entrenamiento del modelo YOLO
- Descarga automatizada del dataset (el usuario lo gestiona manualmente desde Roboflow)

---

## Architecture

El sistema sigue una arquitectura de **pipeline secuencial** con tres capas claramente separadas, más un punto de entrada CLI y una API independiente.

```mermaid
flowchart TD
    A[main.py\nPunto de entrada CLI] --> B[src/config.py\nCarga de configuración]
    B --> C[src/database/database.py\nInicialización DB]
    B --> D[src/ai/detector.py\nCarga modelo YOLO]
    C --> E[src/database/models.py\nORM - Tabla eventos]
    D --> F{Procesamiento\nde fotogramas}
    F --> G[detect_frame\nInferencia YOLO por frame]
    G --> H{¿Detección válida?\nclase ∈ camion/pala\nconfianza ≥ umbral}
    H -->|Sí| I[Evento generado]
    H -->|No| F
    I --> J[Persistencia SQLite\nRepositorio_Eventos]
    I --> K[Log consola\n[EVENTO] ...]
    F -->|Todos los frames procesados| L[Resumen final\n[RESUMEN] ...]

    subgraph API [Proceso separado - FastAPI / Uvicorn]
        M[src/api/main.py] --> N[src/api/routers/eventos.py]
        N --> O[GET /eventos]
        N --> P[GET /eventos/id]
        N --> Q[GET /health]
        O & P --> E
    end
```

**Decisiones de diseño clave:**

1. **Procesamiento síncrono y secuencial**: El MVP no requiere concurrencia. El pipeline lee fotogramas uno a uno con `cv2.VideoCapture`, lo que simplifica el manejo de errores y la trazabilidad.

2. **Separación CLI / API**: El procesamiento del video (`main.py`) y la API REST son procesos independientes. El usuario primero ejecuta el procesamiento, y luego puede iniciar la API para consultar los resultados. Esto evita complejidad de concurrencia en el MVP.

3. **SQLAlchemy como ORM**: Se usa SQLAlchemy Core/ORM sobre `sqlite3` puro para desacoplar la lógica de negocio del driver de base de datos, facilitando migración futura a PostgreSQL u otro motor.

4. **YOLO26n como placeholder**: Se usa el modelo nano pretrained de Ultralytics para el desarrollo. El sistema es agnóstico al peso del modelo: basta cambiar `MODEL_PATH` en `.env` para usar `best.pt` cuando esté disponible.

---

## Components and Interfaces

### 1. `main.py` — Punto de entrada CLI

Orquesta el pipeline completo. No contiene lógica de negocio.

```python
# Flujo de ejecución
def main():
    config = load_config()           # Carga y valida .env
    init_db(config.db_path)          # Crea tabla si no existe
    detector = Detector(config.model_path, config.confidence_threshold)
    detector.load_model()
    cap = open_video(config.video_path)  # Valida y abre el video
    repo = RepositorioEventos(session)

    t_inicio = time.time()
    frame_count = 0
    event_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = calcular_timestamp(cap, frame_count)
        try:
            eventos = detector.detect_frame(frame, frame_count, timestamp)
            for evento in eventos:
                repo.insertar(evento)
                print_evento(evento)
                event_count += 1
        except Exception as e:
            print(f"[ERROR] fotograma={frame_count} | detalle={e}")
        frame_count += 1

    duracion = time.time() - t_inicio
    print(f"[RESUMEN] fotogramas_procesados={frame_count} | "
          f"eventos_registrados={event_count} | duracion_segundos={duracion:.2f}")
    cap.release()
```

### 2. `src/config.py` — Configuración

Carga parámetros desde variables de entorno o archivo `.env`. Falla rápido si faltan variables obligatorias.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    video_path: str            # Obligatorio
    model_path: str            # Obligatorio
    db_path: str               # Obligatorio
    confidence_threshold: float = 0.5
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    class Config:
        env_file = ".env"
```

Las variables `VIDEO_PATH`, `MODEL_PATH` y `DB_PATH` son **obligatorias**; si no están definidas, pydantic-settings lanza `ValidationError` y el sistema detiene la ejecución con un mensaje descriptivo.

### 3. `src/ai/detector.py` — Clase Detector

Responsabilidad única: ejecutar inferencia YOLO sobre un fotograma y retornar eventos filtrados.

```python
CLASE_MAP = {
    "truck": "camion",
    "excavator": "pala",
}

class Detector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None

    def load_model(self) -> None:
        """Carga el modelo YOLO. Falla rápido si la ruta no existe o el archivo está corrupto."""
        ...

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
    ) -> list[dict]:
        """
        Ejecuta inferencia sobre un fotograma y retorna solo detecciones válidas.

        Retorna una lista de dicts con estructura:
        {
            "tipo_maquina": str,    # "camion" | "pala"
            "clase_id": str,        # nombre de clase YOLO original
            "confianza": float,     # valor en [umbral, 1.0]
            "numero_fotograma": int,
            "timestamp": str,
        }

        Solo incluye detecciones donde:
        - clase YOLO ∈ {"truck", "excavator"}
        - confianza >= self.confidence_threshold
        """
        ...
```

**Lógica de filtrado en `detect_frame`:**

```
Para cada detección d en resultados YOLO:
    Si d.clase ∉ CLASE_MAP  →  descartar
    Si d.confianza < umbral  →  descartar
    Si no  →  construir evento y agregar a lista
```

### 4. `src/database/models.py` — ORM

```python
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class Evento(Base):
    __tablename__ = "eventos"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    timestamp        = Column(String, nullable=False)
    tipo_maquina     = Column(String, nullable=False)   # "camion" | "pala"
    clase_id         = Column(String, nullable=False)   # nombre clase YOLO
    confianza        = Column(Float, nullable=False)
    numero_fotograma = Column(Integer, nullable=False)
```

### 5. `src/database/database.py` — Gestión de conexión

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

def init_db(db_path: str) -> None:
    """Crea el archivo SQLite y la tabla eventos si no existen."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)

def get_session_factory(db_path: str):
    engine = get_engine(db_path)
    return sessionmaker(bind=engine)
```

### 6. `RepositorioEventos` — Persistencia (dentro de `database.py` o módulo propio)

```python
class RepositorioEventos:
    def insertar(self, evento: dict) -> None:
        """Inserta un evento en una transacción atómica. Loguea y no relanza en caso de error."""

    def obtener_todos(self) -> list[Evento]:
        """Retorna todos los eventos ordenados por timestamp ASC."""

    def obtener_por_id(self, id: int) -> Evento | None:
        """Retorna el evento con el id dado, o None si no existe."""
```

### 7. `src/api/main.py` — Aplicación FastAPI

```python
from fastapi import FastAPI
from .routers import eventos

app = FastAPI(title="Mining Dispatch API", version="0.1.0")
app.include_router(eventos.router)
```

### 8. `src/api/routers/eventos.py` — Endpoints

| Endpoint | Método | Descripción | Respuesta exitosa | Error |
|---|---|---|---|---|
| `/health` | GET | Verificación de servicio | `{"status": "ok"}` 200 | — |
| `/eventos` | GET | Lista todos los eventos | `[EventoSchema]` 200 | 503 si DB no disponible |
| `/eventos/{id}` | GET | Evento por ID | `EventoSchema` 200 | 404 si no existe, 503 si DB no disponible |

### 9. `src/api/schemas.py` — Pydantic schemas

```python
from pydantic import BaseModel

class EventoSchema(BaseModel):
    id: int
    timestamp: str
    tipo_maquina: str
    clase_id: str
    confianza: float
    numero_fotograma: int

    class Config:
        from_attributes = True
```

---

## Data Models

### Tabla `eventos` (SQLite)

```sql
CREATE TABLE eventos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,   -- ISO 8601: "HH:MM:SS.mmm"
    tipo_maquina     TEXT    NOT NULL,   -- CHECK: 'camion' | 'pala'
    clase_id         TEXT    NOT NULL,   -- nombre de clase YOLO (ej: "truck")
    confianza        REAL    NOT NULL,   -- rango: [confidence_threshold, 1.0]
    numero_fotograma INTEGER NOT NULL    -- ≥ 0, índice del fotograma en el video
);
```

**Invariantes de datos:**
- `tipo_maquina` ∈ `{'camion', 'pala'}` — garantizado por el filtro en `Detector`
- `confianza` ∈ `[confidence_threshold, 1.0]` — garantizado por el filtro en `detect_frame`
- `numero_fotograma` ≥ 0 — índice secuencial comenzando en 0
- `timestamp` es un string en formato `"HH:MM:SS.mmm"` calculado a partir del frame rate del video

### Cálculo del timestamp por fotograma

```python
def calcular_timestamp(cap: cv2.VideoCapture, frame_number: int) -> str:
    fps = cap.get(cv2.CAP_PROP_FPS)
    segundos = frame_number / fps if fps > 0 else 0.0
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:06.3f}"
```

### Formato de log en consola

```
[EVENTO]  fotograma={numero_fotograma} | timestamp={timestamp} | maquina={tipo_maquina} | clase={clase_id} | confianza={confianza:.2f}
[ERROR]   fotograma={numero_fotograma} | detalle={descripcion_error}
[RESUMEN] fotogramas_procesados={n} | eventos_registrados={m} | duracion_segundos={s:.2f}
```

### Archivo `.env.example`

```env
# Rutas obligatorias
VIDEO_PATH=data/raw/video_prueba.mp4
MODEL_PATH=models/best.pt
DB_PATH=data/processed/eventos.db

# Opcionales (con defaults)
CONFIDENCE_THRESHOLD=0.5
API_HOST=127.0.0.1
API_PORT=8000
```

---

### Estructura del dataset Roboflow (`data/processed/`)

El usuario descarga el dataset desde [Roboflow](https://roboflow.com) en formato **YOLOv8 / YOLO26** y lo coloca manualmente en `data/processed/`. La estructura esperada tras la descarga es:

```
data/processed/
└── <nombre-dataset>/          # Ej: "mining-machinery-v1"
    ├── data.yaml              # Configuración de clases y rutas
    ├── train/
    │   ├── images/            # Imágenes de entrenamiento (.jpg, .png)
    │   └── labels/            # Anotaciones YOLO (.txt, una por imagen)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/                  # Opcional
        ├── images/
        └── labels/
```

El archivo `data.yaml` debe declarar al menos las clases `truck` y `excavator` (u otros nombres que mapeen a `camion` y `pala` en `CLASE_MAP`). El código del MVP no accede a este dataset directamente — es usado exclusivamente para el entrenamiento futuro del modelo `best.pt`.

---

## Correctness Properties

*Una propiedad es una característica o comportamiento que debe ser verdadero en todas las ejecuciones válidas del sistema — esencialmente, una declaración formal sobre lo que el sistema debe hacer. Las propiedades sirven como puente entre las especificaciones legibles por humanos y las garantías de corrección verificables automáticamente.*

Después del análisis de prework, se identificaron las siguientes propiedades universales que son adecuadas para property-based testing. Se descartaron criterios de smoke/example (inicialización, logs puntuales, estructura de proyecto) por no aportar valor con 100+ iteraciones.

**Reflexión sobre redundancias:**
- El criterio 4.6 (los objetos de `obtener_todos()` tienen los campos del esquema) es un subconjunto de la Propiedad 4 (que ya verifica campos y valores). Se fusionan.
- La Propiedad 7 (round-trip `GET /eventos/{id}`) y la Propiedad 6 (ordenamiento de `GET /eventos`) son complementarias y no redundantes.
- La Propiedad 3 (filtro por confianza) y la Propiedad 2 (filtro por clase) son ortogonales y se mantienen separadas.

---

### Property 1: Completitud del procesamiento de fotogramas

*Para todo* video válido con N fotogramas, el método de procesamiento del pipeline debe invocar `detect_frame` exactamente N veces, sin omitir fotogramas intermedios independientemente de si se generan detecciones o no.

**Validates: Requirements 3.1**

---

### Property 2: Filtrado de clases — solo camion y pala

*Para todo* fotograma con detecciones YOLO de clases arbitrarias, `detect_frame` debe retornar únicamente las detecciones cuya clase esté en `{"truck", "excavator"}` (mapeadas a `"camion"` y `"pala"` respectivamente). Cualquier otra clase debe ser descartada silenciosamente.

**Validates: Requirements 3.2**

---

### Property 3: Filtrado por umbral de confianza

*Para toda* detección cuyo valor de confianza sea estrictamente menor que `confidence_threshold`, `detect_frame` no debe incluirla en la lista de eventos retornada, sin importar la clase de la detección.

**Validates: Requirements 3.3**

---

### Property 4: Estructura e invariantes del evento generado

*Para toda* detección válida (clase ∈ `{"truck", "excavator"}` y confianza ≥ `confidence_threshold`), el dict retornado por `detect_frame` debe contener exactamente las claves `{tipo_maquina, clase_id, confianza, numero_fotograma, timestamp}`, donde `tipo_maquina` ∈ `{"camion", "pala"}` y `confianza` ∈ `[confidence_threshold, 1.0]`.

**Validates: Requirements 3.4**

---

### Property 5: Round-trip de persistencia de eventos

*Para todo* evento válido (con `tipo_maquina` ∈ `{"camion", "pala"}` y `confianza` ∈ `[0.5, 1.0]`), insertar el evento vía `RepositorioEventos.insertar()` y luego consultar con `obtener_todos()` debe retornar una lista que contenga un objeto con los mismos valores de `tipo_maquina`, `clase_id`, `confianza`, `numero_fotograma` y `timestamp`, con todos los campos del esquema presentes.

**Validates: Requirements 4.3, 4.5, 4.6**

---

### Property 6: Ordenamiento de `obtener_todos()` por timestamp

*Para todo* conjunto de N eventos con timestamps arbitrarios insertados en cualquier orden, `obtener_todos()` debe retornar exactamente N eventos y la secuencia resultante debe estar ordenada por `timestamp` de forma ascendente (es decir, `eventos[i].timestamp <= eventos[i+1].timestamp` para todo i).

**Validates: Requirements 4.5**

---

### Property 7: Formato de línea de log de evento

*Para todo* evento válido procesado, la línea impresa en consola debe seguir el patrón exacto `[EVENTO] fotograma={numero_fotograma} | timestamp={timestamp} | maquina={tipo_maquina} | clase={clase_id} | confianza={confianza:.2f}`, donde todos los campos del evento aparecen en las posiciones correctas.

**Validates: Requirements 5.1**

---

### Property 8: `GET /eventos` retorna todos los eventos ordenados

*Para todo* conjunto de N eventos insertados directamente en la base de datos con timestamps arbitrarios, una solicitud `GET /eventos` debe retornar exactamente N objetos JSON con código HTTP 200, ordenados por `timestamp` ascendente.

**Validates: Requirements 6.1, 6.3**

---

### Property 9: Round-trip de consulta por ID

*Para todo* evento insertado en la base de datos, si su ID asignado es `k`, entonces `GET /eventos/{k}` debe retornar un objeto JSON con código HTTP 200 cuyos campos (`tipo_maquina`, `clase_id`, `confianza`, `numero_fotograma`, `timestamp`) coinciden exactamente con los datos insertados.

**Validates: Requirements 6.4**

---

## Error Handling

### Principio general

El sistema sigue el principio de **falla rápida** para errores de inicialización y **resiliencia** para errores durante el procesamiento:

- **Errores de inicialización** (video inaccesible, modelo corrupto, variable de entorno faltante): el sistema detiene la ejecución inmediatamente con un mensaje descriptivo. No tiene sentido continuar sin los recursos necesarios.
- **Errores durante el procesamiento de un fotograma**: se loguea el error con formato `[ERROR]` y se continúa con el fotograma siguiente. Un fotograma fallido no debe interrumpir el procesamiento del video completo.
- **Errores de inserción en base de datos**: se loguea el error con los datos del evento fallido y se continúa. La pérdida de un evento aislado es preferible a detener el pipeline.

### Tabla de errores y respuestas

| Escenario | Comportamiento | Log/Respuesta |
|---|---|---|
| `VIDEO_PATH` no definido | Detener ejecución | `[CONFIG ERROR] Variable VIDEO_PATH no definida` |
| Video no encontrado | Detener ejecución | `[VIDEO ERROR] Archivo no encontrado: {ruta}` |
| Video con formato inválido | Detener ejecución | `[VIDEO ERROR] OpenCV no puede decodificar: {ruta}` |
| `MODEL_PATH` no definido | Detener ejecución | `[CONFIG ERROR] Variable MODEL_PATH no definida` |
| Modelo no encontrado o corrupto | Detener ejecución | `[MODEL ERROR] No se pudo cargar el modelo: {ruta}` |
| Error en inferencia de fotograma | Continuar | `[ERROR] fotograma={n} \| detalle={e}` |
| Error de inserción en DB | Continuar | `[DB ERROR] No se pudo insertar evento: {datos} \| {e}` |
| DB no disponible en API | Retornar 503 | `{"detalle": "Base de datos no disponible"}` |
| ID no existe en `GET /eventos/{id}` | Retornar 404 | `{"detalle": "Evento no encontrado"}` |

### Gestión de transacciones

Cada inserción en la tabla `eventos` ocurre dentro de una transacción SQLAlchemy independiente (`session.add()` + `session.commit()`). Si `commit()` falla, se ejecuta `session.rollback()` para mantener la integridad de la base de datos.

```python
def insertar(self, evento: dict) -> None:
    try:
        db_evento = Evento(**evento)
        self.session.add(db_evento)
        self.session.commit()
    except Exception as e:
        self.session.rollback()
        print(f"[DB ERROR] No se pudo insertar evento: {evento} | {e}")
```

---

## Testing Strategy

### Herramientas

- **pytest** como framework principal
- **hypothesis** para property-based testing (mínimo 100 iteraciones por propiedad)
- **httpx** + **TestClient de FastAPI** para tests de la API
- **unittest.mock** para mockear el modelo YOLO y la base de datos en tests unitarios
- Base de datos SQLite **en memoria** (`sqlite:///:memory:`) para tests de persistencia

### Estructura de tests

```
tests/
├── conftest.py              # Fixtures compartidos: sesión DB en memoria, video mock, detector mock
├── test_detector.py         # Tests unitarios y de propiedad del Detector
├── test_repository.py       # Tests unitarios y de propiedad del RepositorioEventos
├── test_api.py              # Tests de integración de la API FastAPI
├── test_config.py           # Smoke tests de configuración
└── test_pipeline.py         # Test de integración del pipeline completo
```

### Tests de propiedad (hypothesis)

Cada propiedad del diseño se implementa como un único test de propiedad con `@given`. Las etiquetas permiten trazabilidad hacia este documento.

```python
# Feature: mining-dispatch-mvp, Property 1: Completitud del procesamiento de fotogramas
@given(n_frames=st.integers(min_value=1, max_value=500))
@settings(max_examples=100)
def test_property_1_completitud_fotogramas(n_frames):
    ...

# Feature: mining-dispatch-mvp, Property 2: Filtrado de clases
@given(detecciones=st.lists(st.builds(DeteccionYOLO, clase=st.text(), confianza=st.floats(0.5, 1.0))))
@settings(max_examples=100)
def test_property_2_filtrado_clases(detecciones):
    ...
```

### Tests de ejemplo y smoke

Cubren casos concretos, errores y configuración:

- **Smoke**: carga de modelo con `model_path` configurable, `confidence_threshold` default 0.5, creación de DB, health check
- **Example**: video no encontrado detiene ejecución, formato inválido detiene ejecución, log de carga exitosa, 404 con ID inexistente, 503 con DB no disponible, error de fotograma no detiene pipeline, inserción fallida no detiene pipeline

### Tests de integración

- `test_pipeline.py`: procesa un video real de prueba mínimo (pocos fotogramas) con el modelo YOLO26n y verifica que los eventos se persisten y son consultables via `obtener_todos()`
- `test_api.py`: usa `TestClient` con DB en memoria; verifica todos los endpoints con datos reales

### Balance

Los tests de propiedad (hypothesis) cubren las invariantes universales y el espacio de inputs amplio. Los tests de ejemplo cubren los casos de error y flujos concretos documentados en los requerimientos. Se evita duplicar cobertura: los smoke tests verifican la inicialización, los tests de propiedad no necesitan volver a testear si el modelo se carga.
