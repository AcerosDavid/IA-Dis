# Implementation Plan: Módulo 1 – MVP de Detección de Maquinaria

## Overview

Plan de implementación secuencial del pipeline completo: scaffolding del proyecto → configuración → base de datos → detector YOLO → entry point CLI → API REST → tests → documentación.

El lenguaje de implementación es **Python 3.11+**. Cada tarea construye sobre las anteriores y termina con código integrado y funcional; no hay código huérfano.

---

## Tasks

- [x] 1. Scaffolding del proyecto
  - [x] 1.1 Crear estructura de directorios y archivos base
    - Crear los directorios: `data/raw/`, `data/processed/`, `models/`, `reports/`, `src/ai/`, `src/core/`, `src/database/`, `src/api/routers/`, `tests/`
    - Crear archivos `__init__.py` vacíos en: `src/`, `src/ai/`, `src/core/`, `src/database/`, `src/api/`, `src/api/routers/`, `tests/`
    - Crear `.gitkeep` en `data/raw/`, `models/`, `reports/` para que git los rastree (vacíos por ahora)
    - `data/processed/` NO necesita `.gitkeep` — el usuario colocará aquí el dataset descargado de Roboflow en formato YOLO26; agregar a `.gitignore` para excluir las imágenes del repositorio
    - _Requirements: 8.1_

  - [x] 1.2 Crear `requirements.txt` con dependencias fijadas
    - Incluir con versiones pinned: `ultralytics>=8.3` (incluye YOLO26), `opencv-python-headless`, `sqlalchemy`, `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `python-dotenv`, `pytest`, `hypothesis`, `httpx`
    - _Requirements: 8.9_

  - [x] 1.3 Crear `.env.example` con todas las variables de configuración
    - Incluir variables obligatorias: `VIDEO_PATH`, `MODEL_PATH`, `DB_PATH`
    - Incluir variables opcionales con defaults: `CONFIDENCE_THRESHOLD=0.5`, `API_HOST=127.0.0.1`, `API_PORT=8000`
    - _Requirements: 7.1_

  - [x] 1.4 Crear `README.md` mínimo con instrucciones de instalación y ejecución
    - Sección de instalación: `python -m venv .venv`, activación, `pip install -r requirements.txt`
    - Sección de configuración: copiar `.env.example` a `.env` y completar valores
    - Sección de ejecución del pipeline: `python main.py`
    - Sección de ejecución de la API: `uvicorn src.api.main:app --reload`
    - _Requirements: 8.10_

- [x] 2. Configuración con pydantic-settings
  - [x] 2.1 Implementar `src/config.py` con la clase `Settings`
    - Definir `Settings(BaseSettings)` con los campos: `video_path: str`, `model_path: str`, `db_path: str`, `confidence_threshold: float = 0.5`, `api_host: str = "127.0.0.1"`, `api_port: int = 8000`
    - Configurar `model_config = SettingsConfigDict(env_file=".env")` para leer desde `.env`
    - Si una variable obligatoria no está definida, pydantic-settings lanza `ValidationError`; capturarla en `load_config()` e imprimir `[CONFIG ERROR] Variable {nombre} no definida` antes de relanzar
    - Exponer función `load_config() -> Settings` como punto de acceso único
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.2_

  - [ ]* 2.2 Escribir smoke tests para `src/config.py` en `tests/test_config.py`
    - Test: `Settings` carga correctamente con todas las variables definidas
    - Test: `CONFIDENCE_THRESHOLD` toma el valor `0.5` cuando no está definido
    - Test: `ValidationError` se lanza cuando falta `VIDEO_PATH`, `MODEL_PATH` o `DB_PATH`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 3. Capa de base de datos
  - [x] 3.1 Implementar `src/database/models.py` con el ORM SQLAlchemy
    - Definir `Base = DeclarativeBase()`
    - Definir clase `Evento(Base)` con columnas: `id INTEGER PK AUTOINCREMENT`, `timestamp TEXT NOT NULL`, `tipo_maquina TEXT NOT NULL`, `clase_id TEXT NOT NULL`, `confianza REAL NOT NULL`, `numero_fotograma INTEGER NOT NULL`
    - _Requirements: 4.2, 8.4_

  - [x] 3.2 Implementar `src/database/database.py` con engine, `init_db` y `RepositorioEventos`
    - Implementar `get_engine(db_path: str)` con `create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})`
    - Implementar `init_db(db_path: str) -> None` que llama `Base.metadata.create_all(engine)` (idempotente)
    - Implementar `get_session_factory(db_path: str)` que retorna un `sessionmaker`
    - Implementar clase `RepositorioEventos` con métodos: `insertar(evento: dict)` (transacción atómica con rollback en error y log `[DB ERROR]`), `obtener_todos() -> list[Evento]` (ordenado por `timestamp` ASC), `obtener_por_id(id: int) -> Evento | None`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 8.5_

  - [ ]* 3.3 Escribir property test para round-trip de persistencia (Property 5)
    - Archivo: `tests/test_repository.py`
    - **Property 5: Round-trip de persistencia de eventos**
    - **Validates: Requirements 4.3, 4.5, 4.6**
    - Usar `@given` con `hypothesis` generando eventos válidos (`tipo_maquina` ∈ `{"camion", "pala"}`, `confianza` ∈ `[0.5, 1.0]`, `numero_fotograma` ≥ 0, `timestamp` string no vacío)
    - Verificar que tras `insertar()` + `obtener_todos()` el evento está presente con todos sus campos intactos
    - Usar DB SQLite en memoria (`sqlite:///:memory:`)

  - [ ]* 3.4 Escribir property test para ordenamiento de `obtener_todos()` (Property 6)
    - Archivo: `tests/test_repository.py`
    - **Property 6: Ordenamiento de `obtener_todos()` por timestamp**
    - **Validates: Requirements 4.5**
    - Generar N eventos con timestamps arbitrarios insertados en orden aleatorio
    - Verificar que `obtener_todos()` retorna exactamente N eventos ordenados por `timestamp` ASC

  - [ ]* 3.5 Escribir unit tests para `RepositorioEventos` en `tests/test_repository.py`
    - Test: `init_db` crea la tabla `eventos` si no existe (idempotente — llamar dos veces no falla)
    - Test: `insertar` fallido (ej. campo nulo forzado) logea `[DB ERROR]` y no relanza excepción
    - Test: `obtener_por_id` retorna `None` para un ID que no existe
    - _Requirements: 4.1, 4.3, 4.4, 4.6_

- [x] 4. Checkpoint — base de datos
  - Ejecutar `pytest tests/test_repository.py tests/test_config.py -v` y verificar que todos los tests pasan. Preguntar al usuario si hay dudas antes de continuar.

- [ ] 5. Detector de maquinaria con YOLO
  - [-] 5.1 Implementar `src/ai/detector.py` con la clase `Detector`
    - Definir `CLASE_MAP = {"truck": "camion", "excavator": "pala"}`
    - Implementar `__init__(self, model_path: str, confidence_threshold: float = 0.5)`
    - Implementar `load_model(self) -> None`: carga `YOLO(self.model_path)` de Ultralytics (usar `yolo26n.pt` como modelo placeholder durante desarrollo); si la ruta no existe o el archivo está corrupto, imprimir `[MODEL ERROR] No se pudo cargar el modelo: {ruta}` y relanzar
    - Implementar `detect_frame(self, frame: np.ndarray, frame_number: int, timestamp: str) -> list[dict]`: ejecuta inferencia, filtra por `CLASE_MAP` y `confidence_threshold`, retorna lista de dicts con claves `{tipo_maquina, clase_id, confianza, numero_fotograma, timestamp}`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.2, 3.3, 3.4, 8.3_

  - [ ]* 5.2 Escribir property test para filtrado de clases (Property 2)
    - Archivo: `tests/test_detector.py`
    - **Property 2: Filtrado de clases — solo camion y pala**
    - **Validates: Requirements 3.2**
    - Mockear el modelo YOLO para retornar detecciones con clases arbitrarias (`st.text()`)
    - Verificar que `detect_frame` solo retorna dicts cuyo `tipo_maquina` ∈ `{"camion", "pala"}`

  - [ ]* 5.3 Escribir property test para filtrado por umbral de confianza (Property 3)
    - Archivo: `tests/test_detector.py`
    - **Property 3: Filtrado por umbral de confianza**
    - **Validates: Requirements 3.3**
    - Generar detecciones con valores de `confianza` ∈ `[0.0, 1.0]` y umbrales arbitrarios
    - Verificar que ningún evento retornado tiene `confianza < confidence_threshold`

  - [ ]* 5.4 Escribir property test para estructura e invariantes del evento (Property 4)
    - Archivo: `tests/test_detector.py`
    - **Property 4: Estructura e invariantes del evento generado**
    - **Validates: Requirements 3.4**
    - Para detecciones válidas (clase ∈ `{"truck", "excavator"}`, confianza ≥ umbral), verificar que cada dict retornado contiene exactamente las claves `{tipo_maquina, clase_id, confianza, numero_fotograma, timestamp}` y que `tipo_maquina` ∈ `{"camion", "pala"}`

  - [ ]* 5.5 Escribir smoke tests para `Detector` en `tests/test_detector.py`
    - Test: `load_model` falla con `[MODEL ERROR]` cuando la ruta no existe
    - Test: `confidence_threshold` toma el valor `0.5` por defecto
    - _Requirements: 2.1, 2.2, 2.4_

- [ ] 6. Entry point CLI — pipeline completo
  - [~] 6.1 Implementar función `open_video(video_path: str) -> cv2.VideoCapture` en `main.py`
    - Verificar que el archivo existe con `os.path.exists`; si no, imprimir `[VIDEO ERROR] Archivo no encontrado: {ruta}` y lanzar `SystemExit`
    - Abrir con `cv2.VideoCapture`; si `cap.isOpened()` es `False`, imprimir `[VIDEO ERROR] OpenCV no puede decodificar: {ruta}` y lanzar `SystemExit`
    - Imprimir en consola la resolución, frame rate y número total de fotogramas
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [~] 6.2 Implementar función `calcular_timestamp(cap: cv2.VideoCapture, frame_number: int) -> str` en `main.py`
    - Calcular segundos como `frame_number / fps` (manejar `fps == 0`)
    - Retornar string con formato `"HH:MM:SS.mmm"`
    - _Requirements: 3.4_

  - [~] 6.3 Implementar función `print_evento(evento: dict) -> None` en `main.py`
    - Imprimir con formato exacto: `[EVENTO] fotograma={numero_fotograma} | timestamp={timestamp} | maquina={tipo_maquina} | clase={clase_id} | confianza={confianza:.2f}`
    - _Requirements: 5.1_

  - [~] 6.4 Implementar función `main()` en `main.py` integrando todos los componentes
    - Llamar `load_config()` → `init_db()` → `Detector.load_model()` → `open_video()` → loop de fotogramas
    - En cada iteración: `calcular_timestamp()` → `detect_frame()` → `repo.insertar()` → `print_evento()`
    - Capturar excepciones por fotograma: imprimir `[ERROR] fotograma={n} | detalle={e}` y continuar
    - Al finalizar: imprimir `[RESUMEN] fotogramas_procesados={n} | eventos_registrados={m} | duracion_segundos={s:.2f}`
    - Llamar `cap.release()` en bloque `finally`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.4, 3.5, 4.3, 5.1, 5.2, 5.3_

  - [ ]* 6.5 Escribir property test para completitud del procesamiento de fotogramas (Property 1)
    - Archivo: `tests/test_pipeline.py`
    - **Property 1: Completitud del procesamiento de fotogramas**
    - **Validates: Requirements 3.1**
    - Mockear `cv2.VideoCapture` para simular un video de N fotogramas (`st.integers(min_value=1, max_value=500)`)
    - Mockear `Detector.detect_frame` para registrar cuántas veces se invoca
    - Verificar que `detect_frame` se llama exactamente N veces

  - [ ]* 6.6 Escribir property test para formato de línea de log (Property 7)
    - Archivo: `tests/test_pipeline.py`
    - **Property 7: Formato de línea de log de evento**
    - **Validates: Requirements 5.1**
    - Generar eventos válidos arbitrarios con `hypothesis`
    - Capturar stdout con `capsys` de pytest
    - Verificar que la línea impresa sigue el patrón exacto definido en el diseño

  - [ ]* 6.7 Escribir smoke tests de integración del pipeline en `tests/test_pipeline.py`
    - Test: video no encontrado detiene la ejecución (lanza `SystemExit`)
    - Test: formato de video inválido detiene la ejecución (lanza `SystemExit`)
    - Test: error en `detect_frame` de un fotograma no detiene el pipeline (loguea `[ERROR]` y continúa)
    - _Requirements: 1.2, 1.3, 5.2_

- [~] 7. Checkpoint — pipeline CLI
  - Ejecutar `pytest tests/test_detector.py tests/test_pipeline.py tests/test_config.py tests/test_repository.py -v` y verificar que todos los tests pasan. Preguntar al usuario si hay dudas antes de continuar.

- [ ] 8. API REST con FastAPI
  - [~] 8.1 Implementar `src/api/schemas.py` con `EventoSchema`
    - Definir `EventoSchema(BaseModel)` con campos: `id: int`, `timestamp: str`, `tipo_maquina: str`, `clase_id: str`, `confianza: float`, `numero_fotograma: int`
    - Configurar `model_config = ConfigDict(from_attributes=True)` para compatibilidad con ORM
    - _Requirements: 6.1, 6.3, 6.4, 8.8_

  - [~] 8.2 Implementar `src/api/routers/eventos.py` con los tres endpoints
    - Endpoint `GET /health` → retornar `{"status": "ok"}` con HTTP 200
    - Endpoint `GET /eventos` → usar `RepositorioEventos.obtener_todos()`, serializar con `EventoSchema`, retornar lista con HTTP 200; si DB no disponible, retornar HTTP 503 con `{"detalle": "Base de datos no disponible"}`
    - Endpoint `GET /eventos/{id}` → usar `RepositorioEventos.obtener_por_id(id)`, retornar `EventoSchema` con HTTP 200; si no existe, retornar HTTP 404 con `{"detalle": "Evento no encontrado"}`; si DB no disponible, HTTP 503
    - Inyectar sesión de DB mediante dependencia FastAPI (`Depends`)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 8.6, 8.7_

  - [~] 8.3 Implementar `src/api/main.py` con la aplicación FastAPI
    - Crear `app = FastAPI(title="Mining Dispatch API", version="0.1.0")`
    - Registrar el router de eventos con `app.include_router(eventos.router)`
    - _Requirements: 8.6_

  - [ ]* 8.4 Escribir property test para `GET /eventos` (Property 8)
    - Archivo: `tests/test_api.py`
    - **Property 8: `GET /eventos` retorna todos los eventos ordenados**
    - **Validates: Requirements 6.1, 6.3**
    - Usar `TestClient` de FastAPI con DB SQLite en memoria
    - Insertar N eventos con timestamps arbitrarios usando `hypothesis`
    - Verificar que la respuesta tiene HTTP 200, longitud N y orden `timestamp` ASC

  - [ ]* 8.5 Escribir property test para round-trip de consulta por ID (Property 9)
    - Archivo: `tests/test_api.py`
    - **Property 9: Round-trip de consulta por ID**
    - **Validates: Requirements 6.4**
    - Insertar un evento en DB en memoria, obtener su `id` asignado
    - Verificar que `GET /eventos/{id}` retorna HTTP 200 con campos que coinciden exactamente con los datos insertados

  - [ ]* 8.6 Escribir tests de ejemplo para la API en `tests/test_api.py`
    - Test: `GET /health` retorna `{"status": "ok"}` con HTTP 200
    - Test: `GET /eventos` con DB vacía retorna `[]` y HTTP 200
    - Test: `GET /eventos/{id}` con ID inexistente retorna HTTP 404 con mensaje correcto
    - Test: `GET /eventos` o `GET /eventos/{id}` con DB no disponible retorna HTTP 503
    - _Requirements: 6.2, 6.5, 6.6, 6.7_

- [ ] 9. Fixtures compartidos de tests
  - [~] 9.1 Implementar `tests/conftest.py` con fixtures reutilizables
    - Fixture `db_session`: crea engine SQLite en memoria, llama `init_db`, retorna sesión y hace teardown al finalizar
    - Fixture `repo`: instancia `RepositorioEventos` usando `db_session`
    - Fixture `test_client`: crea `TestClient` de FastAPI con override de la dependencia de DB hacia `db_session`
    - Fixture `mock_detector`: retorna un `Detector` con el modelo YOLO mockeado (sin necesidad de `best.pt`)
    - _Requirements: (soporte transversal para todos los tests)_

- [~] 10. Checkpoint final — suite completa
  - Ejecutar `pytest tests/ -v --tb=short` y verificar que todos los tests pasan. Preguntar al usuario si hay dudas antes de continuar.

- [ ] 11. Documentación final
  - [~] 11.1 Actualizar `README.md` con instrucciones completas
    - Sección de prerequisitos: Python 3.11+, pip
    - Sección de instalación detallada con comandos paso a paso
    - Sección de configuración: descripción de cada variable de `.env`
    - Sección de uso: ejecución del pipeline CLI y de la API por separado
    - Sección de tests: comando `pytest tests/ -v` con descripción de qué cubre cada módulo de tests
    - _Requirements: 8.10_

---

## Notes

- Las tareas marcadas con `*` son opcionales y pueden omitirse para una iteración MVP más rápida; sin embargo, cubren las 9 propiedades de corrección del diseño.
- El orden de las tareas es deliberado: la DB se implementa antes que el Detector porque `main.py` los integra a ambos.
- `conftest.py` (tarea 9.1) se implementa después de la API para tener visibilidad de todas las dependencias que los fixtures deben cubrir.
- El modelo YOLO en los tests siempre se mockea; nunca se require `best.pt` ni conexión a internet en la suite de tests.
- Checkpoints en tareas 4 y 7 permiten validar capas de forma incremental antes de construir sobre ellas.
- El dataset de Roboflow (imágenes + labels) debe colocarse en `data/processed/<nombre-dataset>/` antes de entrenar el modelo. El MVP no requiere este dataset para funcionar — solo se necesita `best.pt` entrenado (o `yolo26n.pt` como placeholder).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1"] },
    { "id": 3, "tasks": ["3.2"] },
    { "id": 4, "tasks": ["3.3", "3.4", "3.5"] },
    { "id": 5, "tasks": ["5.1"] },
    { "id": 6, "tasks": ["5.2", "5.3", "5.4", "5.5"] },
    { "id": 7, "tasks": ["6.1", "6.2", "6.3"] },
    { "id": 8, "tasks": ["6.4"] },
    { "id": 9, "tasks": ["6.5", "6.6", "6.7", "8.1"] },
    { "id": 10, "tasks": ["8.2"] },
    { "id": 11, "tasks": ["8.3"] },
    { "id": 12, "tasks": ["9.1"] },
    { "id": 13, "tasks": ["8.4", "8.5", "8.6"] },
    { "id": 14, "tasks": ["11.1"] }
  ]
}
```
