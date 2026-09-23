"""
Script para dividir el dataset en train/val/test (80/15/5).
Ejecutar: python split_dataset.py

Genera data/processed/equipment-split/ con la estructura correcta para YOLO.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path


# ── Configuracion ─────────────────────────────────────────────────────────────

SOURCE_DIR   = Path("data/processed/equipment.v4-acid-.yolo26/train")
OUTPUT_DIR   = Path("data/processed/equipment-split")
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}
SEED         = 42

random.seed(SEED)


def main() -> None:
    print(f"[INFO] Fuente  : {SOURCE_DIR}")
    print(f"[INFO] Destino : {OUTPUT_DIR}")
    print(f"[INFO] Splits  : {SPLIT_RATIOS}")

    images_dir = SOURCE_DIR / "images"
    labels_dir = SOURCE_DIR / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ERROR] No existen carpetas images/labels en: {SOURCE_DIR}")
        raise SystemExit(1)

    # Emparejar imágenes con sus labels
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    pairs: list[tuple[Path, Path]] = []
    sin_label: list[str] = []

    for img in image_files:
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            sin_label.append(img.name)

    print(f"[INFO] Pares imagen-label: {len(pairs)}")
    if sin_label:
        print(f"[WARN] Imágenes sin label: {len(sin_label)} (se omiten)")

    if not pairs:
        print("[ERROR] No se encontraron pares válidos.")
        raise SystemExit(1)

    # Mezclar y dividir
    random.shuffle(pairs)
    total      = len(pairs)
    n_train    = int(total * SPLIT_RATIOS["train"])
    n_val      = int(total * SPLIT_RATIOS["val"])

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    for name, group in splits.items():
        print(f"  {name:5s}: {len(group):4d} imágenes ({len(group)/total*100:.1f}%)")

    # Crear directorios y copiar archivos
    for split_name, group in splits.items():
        img_dst = OUTPUT_DIR / split_name / "images"
        lbl_dst = OUTPUT_DIR / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Copiando {split_name}...")
        for img_path, lbl_path in group:
            shutil.copy2(img_path, img_dst / img_path.name)
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    # Crear data.yaml con rutas relativas al archivo (funciona en cualquier PC)
    yaml_content = """\
train: train/images
val:   val/images
test:  test/images

nc: 4
names: ['concrete_mixer_truck', 'dump', 'dump_truck', 'excavator']

roboflow:
  workspace: traindata
  project: equipment
  version: 4
  license: CC BY 4.0
  url: https://universe.roboflow.com/traindata/equipment/dataset/4
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\n[OK] Dataset dividido en: {OUTPUT_DIR}")
    print(f"[OK] data.yaml creado en: {yaml_path}")
    print(f"\n[INFO] Actualiza train.py con:")
    print(f"       DATA_YAML = 'data/processed/equipment-split/data.yaml'")


if __name__ == "__main__":
    main()
