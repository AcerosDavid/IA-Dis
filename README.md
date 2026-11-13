<p align="center">
  <img src="logo.svg" alt="Mining Dispatch" width="140">
</p>

# Mining Dispatch — Detección de Maquinaria Minera (Escritorio)

Aplicación de escritorio para analizar video o cámara en tiempo real y detectar
maquinaria minera (camiones, excavadoras y palas) usando YOLO + tracker. Los
eventos de aparición/desaparición y los ciclos se guardan en una base de datos
local SQLite.

La interfaz está hecha con **Tkinter** y no expone API ni endpoints: todo corre
en el equipo del usuario.

---

## Requisitos

- Python 3.11+
- pip

## Instalación

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuración

La aplicación abre sin `.env` gracias a los valores por defecto. Si necesitas
cambiar el modelo, la base de datos o el rendimiento:

```bash
cp .env.example .env
```

| Variable | Descripción | Default |
|---|---|---|
| `MODEL_PATH` | Ruta al modelo YOLO (`.pt`) | `models/best.pt` |
| `DB_URL` | URL de la base de datos local | `sqlite:///data/eventos.db` |
| `VIDEO_PATH` | Video sugerido al abrir | vacío |
| `CAMERA_SOURCE` | Índice de webcam o URL RTSP | `0` |
| `CONFIDENCE_THRESHOLD` | Umbral de confianza | `0.5` |
| `TRACK_SKIP` | Procesar 1 de cada N frames | `3` |
| `TRACK_IMGSZ` | Resolución de inferencia | `320` |
| `FRAMES_AUSENCIA` | Frames sin ver un track | `90` |
| `OCR_HABILITADO` | Leer números de flota | `false` |
| `CLASIFICADOR_HABILITADO` | Clasificar estado operativo | `false` |

---

## Uso

```bash
python app.py
```

1. Pulsa **Abrir video...** o **Usar cámara**.
2. Ajusta las opciones (guardar en BD, OCR, clasificador, frames a saltar).
3. Pulsa **Iniciar**.
4. Observa el video anotado, las estadísticas y la tabla de eventos.
5. **Pausar** / **Detener** controlan el análisis.
6. **Ver eventos guardados en BD** carga el historial local.

Los eventos se almacenan en `data/eventos.db` (SQLite).

---

## Scripts de entrenamiento y herramientas

Los scripts se organizan por categoría y se ejecutan como módulos desde la raíz:

```bash
# Entrenamiento
python -m scripts.training.train
python -m scripts.training.entrenar_clasificador
python -m scripts.training.entrenar_lstm
python -m scripts.training.finetune

# Preparación de datasets
python -m scripts.datasets.extraer_frames
python -m scripts.datasets.etiquetar_video
python -m scripts.datasets.organizar_recortes

# Evaluación / benchmarks
python -m scripts.evaluation.evaluar_modelos
python -m scripts.evaluation.benchmark
```

Varios scripts aceptan `--help`; revisa el docstring de cada uno. Los que no
usan `argparse` (p. ej. `scripts.training.train`) arrancan directamente con la
configuración definida al inicio del archivo.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

La suite usa SQLite en memoria y mockea el modelo YOLO; no requiere GPU ni
descargar pesos.

---

## Estructura del proyecto

```
IA-Dis/
├── app.py                     # Entrada de la aplicación de escritorio
├── src/
│   ├── config.py              # Configuración (pydantic-settings)
│   ├── ai/                    # Detector, OCR y clasificadores
│   ├── core/
│   │   ├── drawing.py         # Dibujo de bboxes, HUD y paneles
│   │   └── geometry.py        # Zonas, bboxes y geofencing
│   ├── database/              # Modelos ORM y repositorios
│   ├── pipeline/
│   │   ├── presencia.py       # Apariciones/desapariciones
│   │   └── runner.py          # Motor de análisis en hilo de fondo
│   └── ui/
│       └── app.py             # Interfaz Tkinter
├── scripts/
│   ├── training/              # Entrenamiento de modelos
│   ├── datasets/              # Extracción y etiquetado
│   └── evaluation/            # Evaluación y benchmarks
├── tests/                     # Suite pytest
├── docs/                      # Guías y documentación
├── legacy/                    # Código archivado (API, auth, infra, CLI)
├── cfg/                       # Configuración del tracker
├── data/                      # Datos, videos y base SQLite
├── models/                    # Modelos YOLO (.pt)
└── weights/                   # Modelos LSTM (.pt)
```

`legacy/` conserva el backend FastAPI, autenticación JWT, migraciones Alembic,
Docker y el pipeline CLI anterior. No forma parte de la aplicación actual.
