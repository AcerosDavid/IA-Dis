# Mining Dispatch MVP — Detección de Maquinaria Minera

Sistema de detección y tracking de maquinaria minera (camiones y palas) usando YOLO26 + ByteTrack. Registra eventos de aparición y desaparición de cada máquina en una base de datos y expone los datos vía API REST.

Funciona con **video grabado** (`main.py`) y **cámara en tiempo real** (`camera.py`).

---

## Prerequisitos

- Python 3.11+
- pip

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración

```bash
# Copiar el archivo de ejemplo
cp .env.example .env
```

Editar `.env` con los valores del entorno:

| Variable | Descripción | Requerida |
|---|---|---|
| `VIDEO_PATH` | Ruta al video (.mp4, .avi, .mov) | ✅ |
| `MODEL_PATH` | Ruta al modelo YOLO26 (.pt) | ✅ |
| `DB_URL` | URL de conexión a la base de datos | ✅ |
| `CAMERA_SOURCE` | Fuente de cámara para `camera.py` | ❌ |
| `CONFIDENCE_THRESHOLD` | Umbral de confianza (default: `0.5`) | ❌ |
| `API_HOST` | Host de la API (default: `127.0.0.1`) | ❌ |
| `API_PORT` | Puerto de la API (default: `8000`) | ❌ |

### Formatos de DB_URL

```env
# SQLite (local, sin servidor)
DB_URL=sqlite:///data/eventos.db

# MySQL
DB_URL=mysql+pymysql://usuario:clave@localhost:3306/mining_dispatch

# PostgreSQL
DB_URL=postgresql+psycopg2://usuario:clave@localhost:5432/mining_dispatch
```

### Fuentes de cámara (CAMERA_SOURCE)

```env
CAMERA_SOURCE=0                              # webcam local
CAMERA_SOURCE=http://192.168.1.X:4747/video  # DroidCam (Android)
CAMERA_SOURCE=rtsp://usuario:clave@ip/stream # cámara RTSP industrial
```

**Modelo base:** Con `MODEL_PATH=models/yolo26n.pt`, Ultralytics descarga el modelo automáticamente la primera vez.  
**Modelo entrenado:** Usar `MODEL_PATH=models/best.pt` tras ejecutar `python train.py`.

---

## Migraciones

Las tablas se crean automáticamente al arrancar `main.py` o `camera.py`. Para entornos de producción o al cambiar modelos, usar Alembic:

```bash
# Aplicar migraciones pendientes
python -m alembic upgrade head

# Crear una nueva migración tras cambiar un modelo ORM
python -m alembic revision --autogenerate -m "descripcion_del_cambio"
python -m alembic upgrade head
```

---

## Uso

### 1. Procesar video grabado

```bash
python main.py
```

Controles durante la reproducción: `Q` / `ESC` para salir | `ESPACIO` para pausar.

Salida esperada:

```
[INFO] Modelo best.pt cargado en cpu
[INFO] Video cargado: 848x478 @ 30.0fps | 1802 fotogramas
[INFO] Tracking activo (ByteTrack) -- Q=salir | ESPACIO=pausar
[APARICION]    id=1 | maquina=camion | clase=dump_truck | frame=0   | ts=00:00:00.000 | conf=0.91
[APARICION]    id=2 | maquina=pala   | clase=excavator  | frame=0   | ts=00:00:00.000 | conf=0.81
[DESAPARICION] id=1 | maquina=camion | frames_en_escena=290 | aparecio=00:00:00.000
[RESUMEN] fotogramas_procesados=1802 | eventos_registrados=12 | duracion_segundos=62.3
```

### 2. Cámara en tiempo real

```bash
python camera.py
```

Mismos controles que `main.py`. Usa `GestorPresencia` para registrar solo eventos de aparición/desaparición, igual que el pipeline de video.

### 3. Entrenar el modelo

```bash
python train.py
```

- Si existe `models/train/weights/last.pt` de una sesión anterior, continúa automáticamente desde la última época (`resume=True`).
- Al terminar, copia `best.pt` a `models/best.pt` listo para usar.
- Para continuar entrenamiento en otro PC, copiar: `models/yolo26n.pt`, `models/train/weights/last.pt`, `models/train/args.yaml` y el dataset `data/processed/`.

Parámetros editables en `train.py`:

| Parámetro | Default | Descripción |
|---|---|---|
| `EPOCHS` | `50` | Total de épocas |
| `BATCH_SIZE` | `8` | Bajar a `4` si hay error de memoria |
| `IMG_SIZE` | `640` | Tamaño de imagen para entrenamiento |

### 4. API REST

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints disponibles:

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Verifica que la API está activa |
| `GET` | `/eventos` | Lista todos los eventos registrados |
| `GET` | `/eventos/{id}` | Obtiene un evento por ID |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

---

## Dataset (Roboflow)

El dataset se descarga desde [Roboflow Universe](https://universe.roboflow.com/traindata/equipment/dataset/4) y se coloca en:

```
data/processed/
└── equipment.v4-acid-.yolo26/
    ├── data.yaml
    ├── train/images/   (~2500 imágenes)
    ├── train/labels/
    ├── valid/images/
    └── valid/labels/
```

Clases detectadas: `dump_truck`, `dump`, `concrete_mixer_truck`, `excavator` → mapeadas a `camion` y `pala` en el dominio del sistema.

Para agregar más data de Roboflow manteniendo los labels consistentes, combinar los datasets dentro del mismo proyecto en Roboflow antes de exportar.

---

## Tests

```bash
pytest tests/ -v
```

La suite cubre: configuración, detector (unitarios + property-based), repositorio, API y pipeline. Usa SQLite en memoria — no requiere base de datos real.

---

## Ajuste de rendimiento

Los parámetros de `main.py` y `camera.py` se pueden ajustar según el hardware:

| `TRACK_IMGSZ` | `TRACK_SKIP` | FPS aprox. (CPU i5) | Calidad |
|---|---|---|---|
| `640` | `2` | 3–5 | Máxima, detecta objetos pequeños |
| `480` | `2` | 5–8 | Buena (recomendado) |
| `320` | `2` | 8–12 | Media |
| `160` | `2` | 12–15 | Básica, pierde objetos pequeños |

**IDs inestables:** Si los IDs de tracking cambian frecuentemente, aumentar `FRAMES_AUSENCIA` (default `30`) a `60` o reducir `TRACK_SKIP` a `1`. Esto mejora a medida que el modelo acumula más épocas de entrenamiento.

---

## Estructura del proyecto

```
IA-Dis/
├── data/
│   ├── raw/                      # Videos de prueba (no versionados)
│   ├── processed/                # Dataset Roboflow (no versionado)
│   └── eventos.db                # Base de datos SQLite (generada al ejecutar)
├── migrations/                   # Migraciones Alembic
│   └── versions/
├── models/                       # Modelos YOLO (.pt)
├── runs/                         # Checkpoints de entrenamiento (Ultralytics)
├── src/
│   ├── config.py                 # Configuración centralizada (pydantic-settings)
│   ├── ai/
│   │   └── detector.py           # Detector + tracker (YOLO26 + ByteTrack)
│   ├── core/
│   │   └── geometry.py           # BBox, Zona, GestorGeofencing
│   ├── database/
│   │   ├── db.py                 # Engine, sesión, RepositorioEventos
│   │   └── models.py             # ORM: Evento, ZonaDB, Ciclo, Turno
│   └── api/
│       ├── main.py               # App FastAPI + CORS
│       ├── schemas.py            # Schemas Pydantic
│       └── routers/
│           └── eventos.py        # Endpoints REST
├── tests/                        # Suite de tests (pytest + hypothesis)
├── reports/                      # Reportes generados
├── main.py                       # Pipeline: video grabado
├── camera.py                     # Pipeline: cámara en tiempo real
├── train.py                      # Entrenamiento YOLO26
├── alembic.ini                   # Configuración Alembic
└── requirements.txt
```
