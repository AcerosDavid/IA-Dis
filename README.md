# Mining Dispatch MVP — Módulo 1: Detección de Maquinaria

Prueba de concepto para detección de maquinaria minera (camiones y palas) usando YOLO26 + OpenCV, con persistencia en SQLite y API REST básica con FastAPI.

Funciona **exclusivamente en modo consola/terminal** — sin interfaz gráfica.

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

Editar `.env` y completar los valores:

| Variable | Descripción | Requerida |
|---|---|---|
| `VIDEO_PATH` | Ruta al video de prueba (.mp4, .avi, .mov) | ✅ |
| `MODEL_PATH` | Ruta al modelo YOLO26 (.pt) | ✅ |
| `DB_PATH` | Ruta al archivo SQLite de eventos | ✅ |
| `CONFIDENCE_THRESHOLD` | Umbral de confianza (default: 0.5) | ❌ |
| `API_HOST` | Host de la API (default: 127.0.0.1) | ❌ |
| `API_PORT` | Puerto de la API (default: 8000) | ❌ |

**Modelo placeholder:** La primera vez que ejecutes con `MODEL_PATH=models/yolo26n.pt`, Ultralytics descarga el modelo automáticamente.

---

## Uso

### 1. Procesar un video (pipeline principal)

```bash
python main.py
```

Salida esperada en consola:
```
[INFO] Video cargado: 1920x1080 @ 30.0fps | 450 fotogramas
[INFO] Modelo YOLO26n cargado en cpu
[EVENTO] fotograma=12 | timestamp=00:00:00.400 | maquina=camion | clase=truck | confianza=0.87
[EVENTO] fotograma=34 | timestamp=00:00:01.133 | maquina=pala | clase=excavator | confianza=0.72
...
[RESUMEN] fotogramas_procesados=450 | eventos_registrados=23 | duracion_segundos=18.45
```

### 2. Consultar eventos via API (proceso separado)

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints disponibles:
- `GET http://127.0.0.1:8000/health` — verificar que la API está activa
- `GET http://127.0.0.1:8000/eventos` — listar todos los eventos detectados
- `GET http://127.0.0.1:8000/eventos/{id}` — obtener un evento por ID
- `GET http://127.0.0.1:8000/docs` — documentación interactiva (Swagger UI)

---

## Dataset (Roboflow)

El dataset de entrenamiento se descarga manualmente desde [Roboflow](https://roboflow.com) y se coloca en `data/processed/`:

```
data/processed/
└── <nombre-dataset>/
    ├── data.yaml
    ├── train/images/
    ├── train/labels/
    ├── valid/images/
    └── valid/labels/
```

El MVP no requiere el dataset para funcionar — solo se usa para entrenar `best.pt`.

---

## Tests

```bash
pytest tests/ -v
```

---

## Estructura del proyecto

```
dispatch_videy/
├── data/raw/          # Videos de prueba
├── data/processed/    # Dataset Roboflow (no versionado)
├── models/            # Modelos YOLO (.pt)
├── src/
│   ├── config.py      # Configuración centralizada
│   ├── ai/detector.py # Clase Detector (YOLO26)
│   ├── database/      # ORM + SQLite
│   └── api/           # FastAPI
├── reports/           # Reportes generados
├── tests/             # Suite de tests
├── main.py            # Entry point CLI
└── requirements.txt
```
