"""
finetune.py — Fine-tuning del modelo YOLO sobre nuevo dataset complementario.

Directrices implementadas:
  1. Parte SIEMPRE de models/best.pt (nunca desde cero ni last.pt)
  2. Valida que las clases del nuevo dataset coincidan EXACTAMENTE con el modelo
  3. Combina el nuevo dataset con muestras del original (anti-olvido catastrófico)
  4. Parámetros conservadores: epochs=25, lr0=0.0001, freeze=10
  5. Guarda el resultado en models/best_finetuned.pt

Uso:
    python finetune.py --data ruta/al/nuevo/data.yaml
    python finetune.py --data ruta/al/nuevo/data.yaml --imgsz 320 --epochs 25

Si no se pasa --data usa el dataset existente en data/processed/equipment.yolo26/
para prueba del pipeline completo.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import yaml
from pathlib import Path


# ── Constantes ────────────────────────────────────────────────────────────────

# Clases del modelo original — NO cambiar el orden
CLASES_ORIGINALES = {
    0: "dump_truck",
    1: "excavator",
    2: "wheel_loader",
}

# Pesos base para fine-tuning (NUNCA last.pt ni desde cero)
MODEL_BASE     = Path("models/best.pt")
MODEL_OUTPUT   = Path("models/best_finetuned.pt")

# Dataset original (para anti-olvido catastrófico)
DATASET_BASE   = Path("data/processed/equipment.yolo26/data.yaml")

# Directorio donde se creará el dataset combinado
DATASET_MERGED = Path("data/processed/finetune_merged")

# Parámetros de fine-tuning (conservadores por diseño)
DEFAULT_EPOCHS  = 25
DEFAULT_LR0     = 0.0001    # tasa baja: no corrompe pesos previos
DEFAULT_FREEZE  = 10        # congelar primeras 10 capas (backbone)
DEFAULT_IMGSZ   = 640
DEFAULT_BATCH   = 8         # conservador para CPU/GPU modesta


# ── Validación de clases ──────────────────────────────────────────────────────

def validar_clases(data_yaml_path: str) -> dict:
    """
    Verifica que el data.yaml del nuevo dataset tenga exactamente las mismas
    clases y el mismo orden que el modelo original.

    Raises:
        SystemExit si las clases no coinciden.

    Returns:
        El dict parseado del data.yaml.
    """
    with open(data_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    nombres_nuevos: list[str] = data.get("names", [])
    nc_nuevo: int             = data.get("nc", len(nombres_nuevos))

    print(f"\n[VALIDACION] Clases del nuevo dataset ({nc_nuevo}):")
    for i, nombre in enumerate(nombres_nuevos):
        original = CLASES_ORIGINALES.get(i, "???")
        match    = "✓" if nombre == original else "✗ DIFERENTE"
        print(f"  {i}: {nombre:20s}  (original: {original}) {match}")

    # Verificar número de clases
    if nc_nuevo != len(CLASES_ORIGINALES):
        print(f"\n[ERROR] El nuevo dataset tiene {nc_nuevo} clases, "
              f"pero el modelo espera {len(CLASES_ORIGINALES)}.")
        print("[ERROR] Fine-tuning abortado — ajusta el data.yaml antes de continuar.")
        raise SystemExit(1)

    # Verificar nombres y orden
    errores = []
    for idx, nombre_original in CLASES_ORIGINALES.items():
        if idx >= len(nombres_nuevos):
            errores.append(f"  Índice {idx}: falta '{nombre_original}'")
        elif nombres_nuevos[idx] != nombre_original:
            errores.append(
                f"  Índice {idx}: se esperaba '{nombre_original}', "
                f"se encontró '{nombres_nuevos[idx]}'"
            )

    if errores:
        print("\n[ERROR] Las clases NO coinciden con el modelo original:")
        for e in errores:
            print(e)
        print("\n[AYUDA] Edita el data.yaml del nuevo dataset para que tenga:")
        print(f"  nc: {len(CLASES_ORIGINALES)}")
        print(f"  names: {list(CLASES_ORIGINALES.values())}")
        raise SystemExit(1)

    print("[VALIDACION] Clases OK — coinciden exactamente con el modelo original\n")
    return data


# ── Combinación de datasets (anti-olvido catastrófico) ────────────────────────

def crear_dataset_combinado(
    data_yaml_nuevo: str,
    porcentaje_base: float = 0.30,
) -> str:
    """
    Combina el nuevo dataset con una muestra del dataset original.

    El dataset original aporta un 30% de sus imágenes (configurable) para
    que el modelo no olvide lo que ya aprendió.

    Args:
        data_yaml_nuevo:  ruta al data.yaml del nuevo dataset
        porcentaje_base:  fracción del dataset original a incluir (0.0-1.0)

    Returns:
        Ruta al nuevo data.yaml combinado.
    """
    import random
    import shutil as sh

    nuevo_path = Path(data_yaml_nuevo).parent
    base_path  = DATASET_BASE.parent

    print(f"[MERGE] Combinando datasets para evitar olvido catastrófico...")
    print(f"  Nuevo dataset : {nuevo_path}")
    print(f"  Dataset base  : {base_path}")
    print(f"  % base incluido: {porcentaje_base*100:.0f}%")

    # Crear estructura de directorios del dataset combinado
    for split in ("train", "valid", "test"):
        (DATASET_MERGED / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_MERGED / split / "labels").mkdir(parents=True, exist_ok=True)

    def _copiar_split(origen: Path, destino: Path, prefijo: str, max_ratio: float = 1.0):
        """Copia imágenes y labels de un split con un ratio máximo."""
        imgs_dir   = origen / "images"
        labels_dir = origen / "labels"
        if not imgs_dir.exists():
            return 0

        imagenes = sorted(imgs_dir.glob("*"))
        if max_ratio < 1.0:
            k = max(1, int(len(imagenes) * max_ratio))
            imagenes = random.sample(imagenes, k)

        copiadas = 0
        for img in imagenes:
            label = labels_dir / img.with_suffix(".txt").name
            dst_img   = destino / "images" / f"{prefijo}_{img.name}"
            dst_label = destino / "labels" / f"{prefijo}_{img.with_suffix('.txt').name}"

            sh.copy2(img, dst_img)
            if label.exists():
                sh.copy2(label, dst_label)
            else:
                # Crear label vacío si no existe (imagen sin anotaciones)
                dst_label.write_text("")
            copiadas += 1
        return copiadas

    totales = {}

    # 1. Copiar TODO el nuevo dataset
    for split in ("train", "valid", "test"):
        n = _copiar_split(
            nuevo_path / split,
            DATASET_MERGED / split,
            prefijo="nuevo",
            max_ratio=1.0,
        )
        totales[split] = totales.get(split, 0) + n
        if n:
            print(f"  [NUEVO]  {split}: {n} imágenes")

    # 2. Añadir muestra del dataset base (anti-olvido)
    if DATASET_BASE.exists():
        for split in ("train", "valid"):
            ratio = porcentaje_base if split == "train" else porcentaje_base * 0.5
            n = _copiar_split(
                base_path / split,
                DATASET_MERGED / split,
                prefijo="base",
                max_ratio=ratio,
            )
            totales[split] = totales.get(split, 0) + n
            if n:
                print(f"  [BASE]   {split}: {n} imágenes ({ratio*100:.0f}% del original)")
    else:
        print(f"  [WARN] Dataset base no encontrado en {base_path} — sin anti-olvido")

    print(f"\n  Total train: {totales.get('train', 0)} imágenes")
    print(f"  Total valid: {totales.get('valid', 0)} imágenes")

    # Generar data.yaml combinado
    merged_yaml = DATASET_MERGED / "data.yaml"
    train_abs = (DATASET_MERGED / "train" / "images").resolve().as_posix()
    valid_abs = (DATASET_MERGED / "valid" / "images").resolve().as_posix()
    test_abs  = (DATASET_MERGED / "test"  / "images").resolve().as_posix()
    merged_yaml.write_text(
        f"train: {train_abs}\n"
        f"val:   {valid_abs}\n"
        f"test:  {test_abs}\n\n"
        f"nc: {len(CLASES_ORIGINALES)}\n"
        f"names: {list(CLASES_ORIGINALES.values())}\n",
        encoding="utf-8",
    )
    print(f"  data.yaml combinado: {merged_yaml}\n")
    return str(merged_yaml)


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(
    data_yaml:  str,
    epochs:     int   = DEFAULT_EPOCHS,
    lr0:        float = DEFAULT_LR0,
    freeze:     int   = DEFAULT_FREEZE,
    imgsz:      int   = DEFAULT_IMGSZ,
    batch:      int   = DEFAULT_BATCH,
    sin_merge:  bool  = False,
) -> None:
    """
    Ejecuta el fine-tuning con los parámetros conservadores recomendados.

    Args:
        data_yaml:  ruta al data.yaml (nuevo dataset o combinado)
        epochs:     épocas de entrenamiento (default: 25)
        lr0:        tasa de aprendizaje inicial (default: 0.0001)
        freeze:     capas a congelar desde el inicio (default: 10)
        imgsz:      tamaño de imagen de entrada (default: 640)
        batch:      tamaño del batch (default: 8)
        sin_merge:  si True, no combina con dataset base (solo nuevo)
    """
    from ultralytics import YOLO

    # ── 1. Verificar modelo base ──────────────────────────────────────────────
    if not MODEL_BASE.exists():
        print(f"[ERROR] No se encontró el modelo base: {MODEL_BASE}")
        print("[ERROR] Asegúrate de que models/best.pt existe antes de hacer fine-tuning.")
        raise SystemExit(1)

    # ── 2. Validar clases ─────────────────────────────────────────────────────
    validar_clases(data_yaml)

    # ── 3. Combinar datasets (anti-olvido catastrófico) ───────────────────────
    if sin_merge:
        print("[WARN] --sin-merge activo: NO se combinará con dataset base.")
        print("[WARN] Riesgo de olvido catastrófico si el nuevo dataset es pequeño.\n")
        yaml_final = data_yaml
    else:
        yaml_final = crear_dataset_combinado(data_yaml)

    # ── 4. Cargar modelo desde best.pt ────────────────────────────────────────
    print(f"[INFO] Cargando modelo base: {MODEL_BASE}")
    model = YOLO(str(MODEL_BASE))

    print(f"\n[FINE-TUNING] Parámetros:")
    print(f"  Modelo base  : {MODEL_BASE}")
    print(f"  Dataset      : {yaml_final}")
    print(f"  Épocas       : {epochs}  (corto para evitar sobreentrenamiento)")
    print(f"  lr0          : {lr0}     (bajo para no corromper pesos previos)")
    print(f"  freeze       : {freeze} capas  (protege backbone)")
    print(f"  imgsz        : {imgsz}px")
    print(f"  batch        : {batch}")
    print()

    # ── 5. Entrenar ───────────────────────────────────────────────────────────
    results = model.train(
        data        = yaml_final,
        epochs      = epochs,
        imgsz       = imgsz,
        batch       = batch,
        lr0         = lr0,           # tasa de aprendizaje baja
        lrf         = 0.01,          # lr final = lr0 * lrf
        freeze      = freeze,        # congelar primeras N capas (backbone)
        warmup_epochs = 2,           # calentamiento suave
        patience    = 10,            # early stopping conservador
        save        = True,
        cache       = False,
        verbose     = True,
        name        = "finetune",
        exist_ok    = True,
        # Augmentación orientada a mejorar ángulos difíciles y oclusiones
        degrees     = 15.0,          # rotación ±15° (ángulos difíciles)
        perspective = 0.001,         # perspectiva leve
        flipud      = 0.1,           # flip vertical ocasional
        mosaic      = 0.5,           # mosaico: mejora objetos pequeños
        copy_paste  = 0.3,           # pegar objetos: simula oclusiones
        scale       = 0.75,          # escala variable: objetos pequeños
        mixup       = 0.1,           # mixup suave: generalización
        erasing     = 0.3,           # borrado aleatorio: oclusiones parciales
    )

    # ── 6. Métricas ───────────────────────────────────────────────────────────
    if results is not None:
        try:
            m         = results.results_dict
            map50     = m.get("metrics/mAP50(B)",    "N/A")
            map5095   = m.get("metrics/mAP50-95(B)", "N/A")
            precision = m.get("metrics/precision(B)", "N/A")
            recall    = m.get("metrics/recall(B)",    "N/A")
            if isinstance(map50, float):
                print(f"\n[METRICAS FINALES]")
                print(f"  mAP50     : {map50:.4f}")
                print(f"  mAP50-95  : {map5095:.4f}")
                print(f"  Precision : {precision:.4f}")
                print(f"  Recall    : {recall:.4f}")
        except Exception:
            pass

    # ── 7. Copiar resultado ───────────────────────────────────────────────────
    best_run = Path("runs/detect/finetune/weights/best.pt")
    if best_run.exists():
        shutil.copy2(best_run, MODEL_OUTPUT)
        print(f"\n[OK] Modelo fine-tuneado guardado en: {MODEL_OUTPUT}")
        print(f"[OK] Para usar en producción, renómbralo o actualiza MODEL_PATH en .env:")
        print(f"     MODEL_PATH=models/best_finetuned.pt")
        print(f"\n[COMPARACION] Evalúa antes de reemplazar best.pt:")
        print(f"  python -c \"from ultralytics import YOLO; YOLO('{MODEL_OUTPUT}').val(data='{yaml_final}')\"")
    else:
        print(f"\n[WARN] No se encontró best.pt en {best_run}")
        for pt in Path("runs").rglob("best.pt"):
            print(f"  Encontrado en: {pt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tuning de YOLO para detección de maquinaria minera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Probar con el dataset existente (sin nuevo dataset)
  python finetune.py

  # Fine-tuning con nuevo dataset de Roboflow
  python finetune.py --data /ruta/al/nuevo/dataset/data.yaml

  # Personalizar parámetros
  python finetune.py --data data.yaml --epochs 30 --imgsz 320 --batch 16

  # Sin combinar con dataset base (no recomendado para datasets pequeños)
  python finetune.py --data data.yaml --sin-merge
        """
    )
    parser.add_argument(
        "--data",
        default=str(DATASET_BASE),
        help=f"Ruta al data.yaml del nuevo dataset (default: {DATASET_BASE})",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Épocas de entrenamiento (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr0", type=float, default=DEFAULT_LR0,
        help=f"Tasa de aprendizaje inicial (default: {DEFAULT_LR0})",
    )
    parser.add_argument(
        "--freeze", type=int, default=DEFAULT_FREEZE,
        help=f"Capas a congelar (default: {DEFAULT_FREEZE})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help=f"Tamaño de imagen (default: {DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH,
        help=f"Batch size (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--sin-merge", action="store_true",
        help="No combinar con dataset base (riesgo de olvido catastrófico)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] data.yaml no encontrado: {args.data}")
        print("[ERROR] Descarga el nuevo dataset de Roboflow y pasa la ruta con --data")
        raise SystemExit(1)

    print("=" * 65)
    print("  Mining Dispatch — Fine-Tuning YOLO")
    print("=" * 65)
    print(f"  Modelo base   : {MODEL_BASE}")
    print(f"  Nuevo dataset : {args.data}")
    print(f"  Épocas        : {args.epochs}")
    print(f"  lr0           : {args.lr0}")
    print(f"  freeze        : {args.freeze} capas")
    print(f"  imgsz         : {args.imgsz}px")
    print(f"  Anti-olvido   : {'NO (--sin-merge)' if args.sin_merge else 'SÍ (merge con base)'}")
    print("=" * 65)

    finetune(
        data_yaml = args.data,
        epochs    = args.epochs,
        lr0       = args.lr0,
        freeze    = args.freeze,
        imgsz     = args.imgsz,
        batch     = args.batch,
        sin_merge = args.sin_merge,
    )


if __name__ == "__main__":
    main()
