"""
extraer_recortes_dataset.py — Extrae recortes de maquinaria desde el dataset
de las 20k imágenes usando los labels YOLO existentes.

A diferencia de extraer_recortes.py (que usa el video en tiempo real),
este script usa las anotaciones ya existentes — son 100% precisas.

Genera recortes en data/recortes_dataset/{camion,excavadora,pala}/
para etiquetar manualmente el estado y moverlos a data/estados/

Uso:
    python extraer_recortes_dataset.py                  # usa las 20k
    python extraer_recortes_dataset.py --max 500        # máximo 500 por clase
    python extraer_recortes_dataset.py --split train    # solo train
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

# ── Configuración ─────────────────────────────────────────────────────────────
DATASET_DIR  = Path("data/processed/equipment.yolo26")
OUTPUT_DIR   = Path("data/recortes_dataset")
MARGEN_PCT   = 0.08    # margen extra del 8% del tamaño del bbox
TAMANO_MIN   = 40      # tamaño mínimo del recorte en píxeles

# Mapeo índice de clase → nombre de carpeta
CLASE_MAP = {
    0: "camion",       # dump_truck
    1: "excavadora",   # excavator
    2: "pala",         # wheel_loader
}


def extraer_recortes_dataset(
    split:    str,
    max_por_clase: int,
    output:   Path,
) -> None:
    imgs_dir   = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    if not imgs_dir.exists():
        print(f"[ERROR] No se encontró: {imgs_dir}")
        return

    # Crear carpetas de salida
    carpetas = {}
    for nombre in CLASE_MAP.values():
        p = output / nombre
        p.mkdir(parents=True, exist_ok=True)
        carpetas[nombre] = p

    # Contar imágenes disponibles por clase
    conteo_disponible = {n: 0 for n in CLASE_MAP.values()}
    archivos = list(imgs_dir.glob("*"))
    random.shuffle(archivos)   # mezclar para variedad

    print(f"\n[INFO] Split: {split} | {len(archivos)} imágenes")
    print(f"[INFO] Máximo por clase: {max_por_clase}")
    print(f"[INFO] Salida: {output.resolve()}\n")

    conteo = {n: 0 for n in CLASE_MAP.values()}

    for img_path in archivos:
        # Verificar si todas las clases ya tienen suficientes recortes
        if all(v >= max_por_clase for v in conteo.values()):
            break

        # Cargar imagen
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Cargar label correspondiente
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        lineas = label_path.read_text(encoding="utf-8").splitlines()

        for linea in lineas:
            partes = linea.strip().split()
            if len(partes) < 5:
                continue

            try:
                cls_id = int(partes[0])
                cx, cy, bw, bh = map(float, partes[1:5])
            except ValueError:
                continue

            nombre = CLASE_MAP.get(cls_id)
            if nombre is None:
                continue

            # Saltar si ya tenemos suficientes de esta clase
            if conteo[nombre] >= max_por_clase:
                continue

            # Convertir de formato YOLO normalizado a píxeles
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Agregar margen
            margen_x = int((x2 - x1) * MARGEN_PCT)
            margen_y = int((y2 - y1) * MARGEN_PCT)
            x1 = max(0, x1 - margen_x)
            y1 = max(0, y1 - margen_y)
            x2 = min(w, x2 + margen_x)
            y2 = min(h, y2 + margen_y)

            # Verificar tamaño mínimo
            if (x2 - x1) < TAMANO_MIN or (y2 - y1) < TAMANO_MIN:
                continue

            recorte = img[y1:y2, x1:x2]
            if recorte.size == 0:
                continue

            # Guardar
            nombre_archivo = f"{img_path.stem}_cls{cls_id}_{conteo[nombre]:04d}.jpg"
            ruta_salida    = carpetas[nombre] / nombre_archivo
            cv2.imwrite(str(ruta_salida), recorte)
            conteo[nombre] += 1

    # Resumen
    print(f"{'='*55}")
    print(f"  EXTRACCIÓN DESDE DATASET COMPLETADA")
    print(f"{'='*55}")
    for nombre, n in conteo.items():
        print(f"  {nombre:<15} {n:5d} recortes → {carpetas[nombre]}")
    print(f"  TOTAL          {sum(conteo.values()):5d} recortes")
    print(f"{'='*55}")
    print(f"""
SIGUIENTE PASO — Etiquetar manualmente:

  Abre: {output.resolve()}

  Para cada imagen en camion/:
    tolva VACÍA  → mover a data/estados/camion/empty/
    tolva LLENA  → mover a data/estados/camion/loaded/

  Para cada imagen en excavadora/:
    excavando    → data/estados/excavadora/digging/
    cargando     → data/estados/excavadora/loading/
    quieta       → data/estados/excavadora/idle/

  Para cada imagen en pala/:
    recogiendo   → data/estados/pala/scooping/
    descargando  → data/estados/pala/dumping/
    quieta       → data/estados/pala/idle/
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae recortes del dataset de 20k usando labels YOLO"
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "valid", "test"],
        help="Split del dataset a usar (default: train)"
    )
    parser.add_argument(
        "--max", type=int, default=300,
        help="Máximo de recortes por clase (default: 300)"
    )
    parser.add_argument(
        "--output", default="data/recortes_dataset",
        help="Carpeta de salida (default: data/recortes_dataset)"
    )
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset no encontrado: {DATASET_DIR}")
        print("[ERROR] Verifica que data/processed/equipment.yolo26/ existe")
        raise SystemExit(1)

    extraer_recortes_dataset(
        split         = args.split,
        max_por_clase = args.max,
        output        = Path(args.output),
    )


if __name__ == "__main__":
    main()
