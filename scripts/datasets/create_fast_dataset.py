"""
Crea un subset reducido del dataset para entrenamiento rapido en CPU.
Mantiene distribucion balanceada de clases.
Ejecutar: python create_fast_dataset.py
"""
from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path


# ── Configuracion ─────────────────────────────────────────────────────────────

SOURCE_DIR     = Path("data/processed/equipment.yolo26")
TARGET_DIR     = Path("data/processed/equipment-fast")
TRAIN_SAMPLES  = 2000   # de 18,672
VAL_SAMPLES    = 400    # de 1,371
SEED           = 42

random.seed(SEED)


def reduce_split(split_name: str, target_count: int) -> int:
    """Copia target_count imagenes manteniendo distribucion de clases."""
    src_images = SOURCE_DIR / split_name / "images"
    src_labels = SOURCE_DIR / split_name / "labels"
    dst_images = TARGET_DIR / split_name / "images"
    dst_labels = TARGET_DIR / split_name / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    # Agrupar por clase dominante en cada imagen
    by_class: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    sin_label = 0

    for img in sorted(src_images.glob("*.jpg")) + sorted(src_images.glob("*.png")):
        lbl = src_labels / (img.stem + ".txt")
        if not lbl.exists():
            sin_label += 1
            continue
        contenido = lbl.read_text(encoding="utf-8").strip()
        if not contenido:
            continue
        clase_id = int(contenido.split()[0])
        by_class[clase_id].append((img, lbl))

    if sin_label:
        print(f"  [WARN] {sin_label} imagenes sin label omitidas")

    # Muestrear equitativamente por clase
    samples_per_class = target_count // len(by_class)
    seleccionados: list[tuple[Path, Path]] = []

    for clase_id, archivos in sorted(by_class.items()):
        random.shuffle(archivos)
        muestra = archivos[:samples_per_class]
        seleccionados.extend(muestra)
        print(f"  clase {clase_id}: {len(muestra)} de {len(archivos)} imagenes")

    # Completar si faltan por redondeo
    restantes = target_count - len(seleccionados)
    if restantes > 0:
        todos = [f for archivos in by_class.values() for f in archivos]
        random.shuffle(todos)
        seleccionados.extend(todos[:restantes])

    # Copiar archivos
    for img_path, lbl_path in seleccionados:
        shutil.copy2(img_path, dst_images / img_path.name)
        shutil.copy2(lbl_path, dst_labels / lbl_path.name)

    return len(seleccionados)


def main() -> None:
    print(f"[INFO] Fuente  : {SOURCE_DIR}")
    print(f"[INFO] Destino : {TARGET_DIR}")
    print()

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    print(f"[INFO] Creando train ({TRAIN_SAMPLES} imagenes)...")
    n_train = reduce_split("train", TRAIN_SAMPLES)

    print(f"[INFO] Creando valid ({VAL_SAMPLES} imagenes)...")
    n_val = reduce_split("valid", VAL_SAMPLES)

    # data.yaml con rutas relativas (funciona en cualquier PC)
    yaml_content = """\
train: train/images
val:   valid/images
test:  test/images

nc: 3
names: ['dump_truck', 'excavator', 'wheel_loader']

roboflow:
  workspace: david-aceros
  project: equipment-rh1n3
  version: dataset
  license: CC BY 4.0
  url: https://universe.roboflow.com/david-aceros/equipment-rh1n3/dataset/dataset
"""
    (TARGET_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"\n[OK] Dataset reducido creado:")
    print(f"     train : {n_train} imagenes")
    print(f"     valid : {n_val}  imagenes")
    print(f"     Ubicacion: {TARGET_DIR}")
    print(f"\n[INFO] Actualiza train.py con:")
    print(f"       DATA_YAML = 'data/processed/equipment-fast/data.yaml'")


if __name__ == "__main__":
    main()
