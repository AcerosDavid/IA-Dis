# Mining Dispatch — Guía de Google Colab

Esta guía tiene dos partes:
1. **Inferencia en GPU** — Ejecutar detección en videos (para PCs sin GPU)
2. **Entrenamiento/Fine-tuning** — Mejorar el modelo con más datos

> **Modelo:** YOLO26 (YOLO11n en ultralytics) — arquitectura de 181 capas, 2.5M parámetros

## Parte 1: Inferencia en GPU (para PCs sin gráfica)

### ¿Cuándo usar esta parte?
- Tu PC no tiene GPU y la detección es muy lenta
- Quieres probar el modelo en un dispositivo potente (T4 GPU de Colab)
- Necesitas procesar videos largos rápidamente

### Archivos necesarios en Google Drive:
```
MyDrive/
└── mining_dispatch/
    ├── best.pt              ← Modelo entrenado
    └── video.mp4            ← Video a analizar
```

### Pasos para inferencia:

1. **Sube los archivos a Drive:**
   - `best.pt` → `mining_dispatch/`
   - Tu video → `mining_dispatch/video.mp4`

2. **Abre el notebook en Colab:**
   - Ve a [colab.research.google.com](https://colab.research.google.com)
   - Archivo → Subir notebook → `colab_inference.ipynb`
   - Entorno de ejecución → Cambiar tipo → T4 GPU

3. **Ejecuta las celdas en orden:**
   - Celda 1: Verificar GPU
   - Celda 2: Instalar dependencias
   - Celda 3: Montar Drive
   - Celda 4-10: Ejecutar detección y descargar resultados

### Resultados:
- Video anotado con bboxes y track IDs
- Estadísticas de detección (FPS, tracks únicos, clases)
- CSV con métricas comparativas

### Ventajas vs tu PC:
- **Velocidad:** 5-10x más rápido (GPU vs CPU)
- **Precisión:** Puede usar IMGSZ=640 o 800 (más preciso)
- **Estabilidad:** Más memoria para procesar videos largos

---

## Parte 2: Entrenamiento/Fine-tuning

> **Base:** `best.pt` entrenado con 20k imágenes de maquinaria minera
> **Objetivo:** Fine-tuning con 22k imágenes (20k originales + 2k casos difíciles)

### Archivos que deben estar en Google Drive (raíz de MyDrive):
| Archivo | Descripción |
|---|---|
| `best.pt` | Modelo actual entrenado |
| `NOMBRE_DATASET.zip` | Dataset descargado de Roboflow (22k imágenes) |

### Carpeta que debe existir en Drive:
```
MyDrive/
└── mining_checkpoints/   ← ya creada, aquí van los backups automáticos
```

---

## Abrir Colab

1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. **Archivo → Subir notebook** → selecciona `colab_finetune.ipynb`
3. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**
4. Ejecuta las celdas **en orden**

---

## Celda 1 — Verificar GPU

```python
!nvidia-smi
import torch
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
BATCH = 16 if vram >= 14 else (8 if vram >= 8 else 4)
print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram:.1f}GB  BATCH: {BATCH}')
```

**Resultado esperado:**
```
Tesla T4 | VRAM: 14.x GB | BATCH: 16
```

---

## Celda 2 — Instalar dependencias

```python
# Instalar ultralytics — incluye YOLO26 (YOLO11) desde la versión 8.3+
!pip install "ultralytics>=8.3.0" -q

import ultralytics, torch
print('ultralytics:', ultralytics.__version__)
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())

# Verificar que YOLO26 (YOLO11n) está disponible
from ultralytics import YOLO
m = YOLO('yolo26n.pt')   # descarga el modelo base si no existe
print('YOLO26 OK — clases base COCO disponibles')
del m   # liberar memoria, solo era para verificar
```

---

## Celda 3 — Montar Drive y configurar rutas

> ⚠️ **IMPORTANTE:** Cambia `NOMBRE_DEL_ZIP.zip` por el nombre real de tu archivo

```python
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os

# ── CAMBIA ESTE NOMBRE ──────────────────────────────────────────
DRIVE_ZIP        = '/content/drive/MyDrive/NOMBRE_DEL_ZIP.zip'
# ────────────────────────────────────────────────────────────────

DRIVE_MODEL      = '/content/drive/MyDrive/best.pt'
DRIVE_BACKUP_DIR = '/content/drive/MyDrive/mining_checkpoints'

# Si no sabes el nombre exacto del zip, este código lo busca automáticamente
if not Path(DRIVE_ZIP).exists():
    print('Buscando zips en Drive...')
    for root, dirs, files in os.walk('/content/drive/MyDrive'):
        for f in files:
            if f.endswith('.zip'):
                print(f'  Encontrado: {os.path.join(root, f)}')

assert Path(DRIVE_MODEL).exists(), f'No se encontró best.pt en {DRIVE_MODEL}'
Path(DRIVE_BACKUP_DIR).mkdir(parents=True, exist_ok=True)

print('Drive OK')
print(f'ZIP     : {DRIVE_ZIP}')
print(f'Modelo  : {DRIVE_MODEL}')
print(f'Backups : {DRIVE_BACKUP_DIR}')
```

---

## Celda 4 — Descomprimir dataset

```python
import zipfile, shutil
from pathlib import Path

DATASET_DIR = Path('/content/dataset')
MODEL_LOCAL = Path('/content/best.pt')

if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
    print('Descomprimiendo dataset...')
    DATASET_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(DRIVE_ZIP, 'r') as z:
        z.extractall(str(DATASET_DIR))
    print('Listo.')
else:
    print('Dataset ya descomprimido.')

shutil.copy(DRIVE_MODEL, MODEL_LOCAL)
print(f'Modelo copiado: {MODEL_LOCAL}')

# Verificar estructura
print('\nEstructura del dataset:')
for split in ['train', 'valid', 'test']:
    for sub in DATASET_DIR.rglob(split):
        imgs = list((sub / 'images').glob('*')) if (sub / 'images').exists() else []
        if imgs:
            print(f'  {sub.relative_to(DATASET_DIR)}/images: {len(imgs)} imágenes')
```

---

## Celda 5 — Crear data.yaml con rutas absolutas

> ⚠️ Ultralytics tiene un bug con rutas relativas que duplica el path. Siempre usar absolutas.

```python
import yaml
from pathlib import Path

# Buscar data.yaml automáticamente dentro del zip descomprimido
yaml_files = list(DATASET_DIR.rglob('data.yaml'))
print(f'data.yaml encontrados: {yaml_files}')

ORIGINAL_YAML = yaml_files[0]
with open(ORIGINAL_YAML, encoding='utf-8') as f:
    orig = yaml.safe_load(f)

print(f'Clases : {orig["names"]}')
print(f'nc     : {orig["nc"]}')

# Crear data.yaml con rutas absolutas
base_dir  = ORIGINAL_YAML.parent
DATA_YAML = '/content/data.yaml'

with open(DATA_YAML, 'w') as f:
    f.write(f"train: {(base_dir / 'train' / 'images').resolve()}\n")
    f.write(f"val:   {(base_dir / 'valid' / 'images').resolve()}\n")
    f.write(f"test:  {(base_dir / 'test'  / 'images').resolve()}\n\n")
    f.write(f"nc: {orig['nc']}\n")
    f.write(f"names: {orig['names']}\n")

print('\ndata.yaml creado:')
with open(DATA_YAML) as f:
    print(f.read())
```

---

## Celda 6 — Configurar backup automático

Guarda el progreso en Drive cada 5 épocas. Si Colab se desconecta, el entrenamiento puede reanudarse desde el último checkpoint.

```python
import shutil
from pathlib import Path

BACKUP_EVERY = 5
WEIGHTS_DIR  = Path('/content/runs/finetune_v2/weights')
BACKUP_DIR   = Path(DRIVE_BACKUP_DIR)

def backup_a_drive(trainer):
    epoca = trainer.epoch + 1
    if epoca % BACKUP_EVERY != 0:
        return
    try:
        for nombre in ['last.pt', 'best.pt']:
            origen = WEIGHTS_DIR / nombre
            if origen.exists():
                # Copia con número de época (historial)
                shutil.copy2(origen, BACKUP_DIR / f'epoch{epoca:03d}_{nombre}')
                # Copia con nombre fijo (para reanudar rápido)
                shutil.copy2(origen, BACKUP_DIR / nombre)
        print(f'  [BACKUP] Época {epoca} → Drive ✓')
    except Exception as e:
        print(f'  [BACKUP] Error: {e}')

print(f'Backup configurado: cada {BACKUP_EVERY} épocas → {BACKUP_DIR}')
```

---

## Celda 7 — Detectar checkpoint para reanudar

Si el entrenamiento se cortó, esta celda detecta el último checkpoint automáticamente.

```python
import shutil
from pathlib import Path

CHECKPOINT_DRIVE = Path(DRIVE_BACKUP_DIR) / 'last.pt'
CHECKPOINT_LOCAL = Path('/content/runs/finetune_v2/weights/last.pt')
MODO_RESUME      = False
MODELO_INICIAL   = str(MODEL_LOCAL)

if CHECKPOINT_LOCAL.exists():
    print(f'Checkpoint local encontrado: {CHECKPOINT_LOCAL}')
    MODO_RESUME = True
elif CHECKPOINT_DRIVE.exists():
    print(f'Checkpoint en Drive, copiando...')
    CHECKPOINT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(CHECKPOINT_DRIVE, CHECKPOINT_LOCAL)
    MODO_RESUME = True
else:
    print('Sin checkpoint — entrenamiento nuevo desde best.pt')

if MODO_RESUME:
    MODELO_INICIAL = str(CHECKPOINT_LOCAL)
    print(f'→ RESUME desde: {MODELO_INICIAL}')
else:
    print(f'→ NUEVO desde: {MODELO_INICIAL}')
```

---

## Celda 8 — ENTRENAMIENTO

```python
from ultralytics import YOLO
import torch

# Recalcular BATCH si la sesión se reinició
if 'BATCH' not in dir():
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    BATCH = 16 if vram >= 14 else (8 if vram >= 8 else 4)
    print(f'BATCH recalculado: {BATCH}')

model = YOLO(MODELO_INICIAL)
model.add_callback('on_train_epoch_end', backup_a_drive)

print('=' * 60)
print('  Fine-tuning v2 — 22k imágenes (20k + 2k casos difíciles)')
print('=' * 60)
print(f'  Modelo   : {MODELO_INICIAL}')
print(f'  Resume   : {MODO_RESUME}')
print(f'  Dataset  : {DATA_YAML}')
print(f'  Batch    : {BATCH}')
print(f'  Backup   : cada {BACKUP_EVERY} épocas → {DRIVE_BACKUP_DIR}')
print('=' * 60)

results = model.train(
    data          = DATA_YAML,
    epochs        = 30,
    imgsz         = 640,
    batch         = BATCH,
    project       = '/content/runs',
    name          = 'finetune_v2',
    exist_ok      = True,
    resume        = MODO_RESUME,

    # Optimizador — SGD respeta lr0 (auto lo ignora)
    optimizer     = 'SGD',
    lr0           = 0.00003,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3,

    # Todas las capas entrenables — el modelo ya está adaptado
    freeze        = 0,
    patience      = 15,

    # Augmentación para oclusiones y ángulos difíciles
    degrees       = 20.0,   # rotación ±20°
    perspective   = 0.002,  # perspectiva
    shear         = 5.0,    # cizallamiento
    fliplr        = 0.5,    # flip horizontal
    flipud        = 0.15,   # flip vertical
    scale         = 0.8,    # zoom out — objetos pequeños
    mosaic        = 1.0,    # mosaico completo
    copy_paste    = 0.5,    # oclusiones agresivas
    erasing       = 0.4,    # borrado aleatorio
    mixup         = 0.1,    # generalización
    hsv_h         = 0.015,
    hsv_s         = 0.5,
    hsv_v         = 0.4,
    amp           = True,   # Mixed Precision — reduce VRAM
    save          = True,
    save_period   = 5,      # checkpoint local cada 5 épocas
    workers       = 2,
    verbose       = True,
)

print('Entrenamiento completado.')
```

**Tiempo esperado con T4:** ~3-4 horas para 30 épocas con 22k imágenes.

---

## Celda 9 — Métricas finales

```python
if results is not None:
    try:
        m = results.results_dict
        print('\n[MÉTRICAS FINALES]')
        print(f"  mAP50    : {m.get('metrics/mAP50(B)',    'N/A'):.4f}")
        print(f"  mAP50-95 : {m.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision: {m.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall   : {m.get('metrics/recall(B)',    'N/A'):.4f}")
    except Exception as e:
        print(f'No se pudieron obtener métricas: {e}')
```

---

## Celda 10 — Guardar modelo final en Drive

```python
from datetime import datetime
import shutil
from pathlib import Path

timestamp   = datetime.now().strftime('%Y%m%d_%H%M')
WEIGHTS_DIR = Path('/content/runs/finetune_v2/weights')
BACKUP_DIR  = Path(DRIVE_BACKUP_DIR)

for nombre in ['best.pt', 'last.pt']:
    origen = WEIGHTS_DIR / nombre
    if origen.exists():
        shutil.copy2(origen, BACKUP_DIR / f'{timestamp}_{nombre}')
        shutil.copy2(origen, BACKUP_DIR / nombre)
        print(f'Guardado: {timestamp}_{nombre}')

best = WEIGHTS_DIR / 'best.pt'
if best.exists():
    destino = BACKUP_DIR / 'best_v2.pt'
    shutil.copy2(best, destino)
    print(f'\nModelo final guardado: {destino}')

print('\n[SIGUIENTE PASO]')
print('1. Descarga best_v2.pt desde Drive → mining_checkpoints/')
print('2. Copia a models/best_advanced.pt en tu PC')
print('3. Cambia en .env: MODEL_PATH=models/best_advanced.pt')
print('4. Corre main.py y compara con el modelo anterior')
print('5. Si mejora → renombra a models/best.pt')
```

---

## Si el entrenamiento se corta (Colab se desconecta)

1. Reconecta la sesión en Colab
2. Ejecuta las celdas **1 a 7** en orden
3. La **celda 7** detecta automáticamente el checkpoint en Drive
4. Ejecuta la **celda 8** — el entrenamiento continúa desde donde se cortó

El backup en `mining_checkpoints/` tiene:
```
mining_checkpoints/
├── best.pt              ← mejor modelo hasta ahora (siempre actualizado)
├── last.pt              ← último checkpoint (para reanudar)
├── epoch005_best.pt     ← historial por época
├── epoch005_last.pt
├── epoch010_best.pt
└── ...
```

---

## Después del entrenamiento — instalar en tu PC

```powershell
# 1. Descargar best_v2.pt de Drive a tu PC
# 2. Copiar al proyecto
Copy-Item "$env:USERPROFILE\Downloads\best_v2.pt" "models\best_advanced.pt"

# 3. Activar el nuevo modelo
# Editar .env:
# MODEL_PATH=models/best_advanced.pt

# 4. Probar
C:\laragon\bin\python\python-3.13\python.exe main.py
```

---

## Comparar modelos antes de reemplazar

```powershell
C:\laragon\bin\python\python-3.13\python.exe -c "
from ultralytics import YOLO

print('=== MODELO ORIGINAL ===')
m1 = YOLO('models/best.pt')
r1 = m1.val(data='data/processed/equipment.yolo26/data.yaml', verbose=False)
print(f'mAP50: {r1.results_dict[\"metrics/mAP50(B)\"]:.4f}')

print()
print('=== MODELO NUEVO ===')
m2 = YOLO('models/best_advanced.pt')
r2 = m2.val(data='data/processed/equipment.yolo26/data.yaml', verbose=False)
print(f'mAP50: {r2.results_dict[\"metrics/mAP50(B)\"]:.4f}')
"
```

Si el nuevo modelo tiene mejor mAP50 → reemplaza:
```powershell
Copy-Item "models\best_advanced.pt" "models\best.pt"
```
