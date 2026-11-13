"""
finetune_advanced.py — Fine-tuning avanzado para corregir 3 fallos críticos.

Fallos que corrige:
  1. Confusión excavadora ↔ pala  (hard negative mining)
  2. Camiones pequeños/lejanos    (imgsz=640, multi-scale)
  3. Objetos truncados/ocluidos   (mosaic=1.0, copy_paste, erasing)

Uso:
  # Con dataset existente (sin casos difíciles aún)
  python finetune_advanced.py

  # Con carpeta de casos difíciles corregidos
  python finetune_advanced.py --hard-negatives ruta/casos_dificiles/

  # Ajustar parámetros
  python finetune_advanced.py --epochs 30 --imgsz 640 --batch 4

Resultado:
  models/best_advanced.pt  — listo para reemplazar best.pt tras validación
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import yaml
from pathlib import Path


# ── Configuración fija (no cambiar sin análisis) ──────────────────────────────

MODEL_BASE      = Path("models/best.pt")
MODEL_OUTPUT    = Path("models/best_advanced.pt")
DATASET_BASE    = Path("data/processed/equipment.yolo26/data.yaml")
DATASET_MERGED  = Path("data/processed/advanced_merged")

CLASES = {0: "dump_truck", 1: "excavator", 2: "wheel_loader"}

# Parámetros del entrenamiento (directriz estricta del usuario)
EPOCHS   = 30
LR0      = 0.00005   # muy baja: no destruye conocimiento previo
FREEZE   = 5         # protege primeras 5 capas (geometría base)
IMGSZ    = 640       # crucial para objetos pequeños/lejanos
BATCH    = 8         # ajustar según VRAM: 4 para 2GB, 8 para 4GB+


# ── Análisis de confusiones (hard negative mining) ───────────────────────────

def analizar_confusiones_dataset(dataset_path: Path) -> dict:
    """
    Analiza el dataset de labels para identificar la distribución de clases.
    Útil para detectar desbalance que contribuye a confusiones excavadora/pala.

    Returns: {clase_id: count}
    """
    conteo = {0: 0, 1: 0, 2: 0}
    labels_dirs = [
        dataset_path / "train" / "labels",
        dataset_path / "valid" / "labels",
    ]
    for labels_dir in labels_dirs:
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            try:
                for linea in label_file.read_text(encoding="utf-8").splitlines():
                    if linea.strip():
                        clase = int(linea.split()[0])
                        if clase in conteo:
                            conteo[clase] += 1
            except Exception:
                pass

    total = sum(conteo.values()) or 1
    print("\n[ANALISIS] Distribución de clases en el dataset:")
    for cls_id, nombre in CLASES.items():
        n = conteo[cls_id]
        pct = n / total * 100
        barra = "█" * int(pct / 2)
        print(f"  {cls_id} {nombre:<15} {n:6d} ({pct:5.1f}%)  {barra}")

    # Detectar desbalance crítico
    min_cls = min(conteo, key=conteo.get)
    max_cls = max(conteo, key=conteo.get)
    ratio   = conteo[max_cls] / max(conteo[min_cls], 1)
    if ratio > 5:
        print(f"\n[WARN] Desbalance severo: {CLASES[max_cls]} tiene {ratio:.1f}x "
              f"más ejemplos que {CLASES[min_cls]}")
        print("[WARN] Esto contribuye a la confusión. Considera añadir más ejemplos "
              f"de {CLASES[min_cls]} o usar class_weights.")
    return conteo


# ── Creación del dataset unificado ────────────────────────────────────────────

def crear_dataset_avanzado(hard_negatives_dir: str | None) -> str:
    """
    Construye el dataset unificado para los 3 fallos críticos:

      1. Dataset base completo (18k+ imágenes)
      2. Hard negatives (casos difíciles excavadora/pala) — si se proveen
      3. Generación de crops de objetos pequeños para refuerzo

    Returns: ruta al data.yaml del dataset unificado
    """
    import random

    print("\n[DATASET] Construyendo dataset unificado...")

    base_path = DATASET_BASE.parent

    for split in ("train", "valid"):
        (DATASET_MERGED / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_MERGED / split / "labels").mkdir(parents=True, exist_ok=True)

    def _copiar_split(
        origen:  Path,
        destino: Path,
        prefijo: str,
        ratio:   float = 1.0,
    ) -> int:
        imgs_dir   = origen / "images"
        labels_dir = origen / "labels"
        if not imgs_dir.exists():
            return 0

        imagenes = sorted(imgs_dir.glob("*"))
        if ratio < 1.0:
            k = max(1, int(len(imagenes) * ratio))
            imagenes = random.sample(imagenes, k)

        copiadas = 0
        for img in imagenes:
            label     = labels_dir / img.with_suffix(".txt").name
            dst_img   = destino / "images" / f"{prefijo}_{img.name}"
            dst_label = destino / "labels" / f"{prefijo}_{img.with_suffix('.txt').name}"
            shutil.copy2(img, dst_img)
            if label.exists():
                shutil.copy2(label, dst_label)
            else:
                dst_label.write_text("")
            copiadas += 1
        return copiadas

    totales = {"train": 0, "valid": 0}

    # ── 1. Dataset base completo ──────────────────────────────────────────────
    for split in ("train", "valid"):
        n = _copiar_split(base_path / split, DATASET_MERGED / split, "base")
        totales[split] += n
        print(f"  [BASE]  {split}: {n:5d} imgs")

    # ── 2. Hard negatives (casos difíciles corregidos) ────────────────────────
    if hard_negatives_dir and os.path.isdir(hard_negatives_dir):
        hn_path = Path(hard_negatives_dir)

        # Asumir estructura: hard_negatives/{images,labels} o hard_negatives/train/
        if (hn_path / "images").exists():
            # Estructura plana — meter todo al train
            n = _copiar_split(hn_path, DATASET_MERGED / "train", "hn")
            totales["train"] += n
            print(f"  [HN]    train: {n:5d} imgs hard negatives")
        elif (hn_path / "train").exists():
            # Estructura con splits
            for split in ("train", "valid"):
                n = _copiar_split(hn_path / split, DATASET_MERGED / split, "hn")
                totales[split] += n
                print(f"  [HN]    {split}: {n:5d} imgs hard negatives")
        else:
            print(f"  [WARN] Hard negatives: estructura no reconocida en {hn_path}")
            print(f"         Espera: {hn_path}/images/ o {hn_path}/train/images/")

        # Oversample de hard negatives: duplicar para dar más peso
        # (compensa su baja cantidad vs el dataset base)
        print("  [HN]    Duplicando hard negatives (oversample x2)...")
        hn_imgs = list((DATASET_MERGED / "train" / "images").glob("hn_*"))
        for img in hn_imgs:
            label = (DATASET_MERGED / "train" / "labels" / img.with_suffix(".txt").name)
            dst_img   = img.parent   / f"hn2_{img.name}"
            dst_label = label.parent / f"hn2_{label.name}"
            shutil.copy2(img, dst_img)
            if label.exists():
                shutil.copy2(label, dst_label)
            totales["train"] += 1
    else:
        print("  [INFO] Sin hard negatives — la augmentación compensará la confusión")

    # ── 3. Resumen ────────────────────────────────────────────────────────────
    print(f"\n  Total train : {totales['train']:6d} imágenes")
    print(f"  Total valid : {totales['valid']:6d} imágenes")

    # Generar data.yaml con rutas ABSOLUTAS.
    # Ultralytics resuelve rutas relativas desde el directorio del yaml,
    # lo que duplica el path. Rutas absolutas evitan este bug.
    merged_yaml = DATASET_MERGED / "data.yaml"
    train_abs = (DATASET_MERGED / "train" / "images").resolve().as_posix()
    valid_abs = (DATASET_MERGED / "valid" / "images").resolve().as_posix()
    merged_yaml.write_text(
        f"train: {train_abs}\n"
        f"val:   {valid_abs}\n\n"
        f"nc: {len(CLASES)}\n"
        f"names: {list(CLASES.values())}\n",
        encoding="utf-8",
    )
    print(f"  data.yaml  : {merged_yaml}")
    return str(merged_yaml)


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def entrenar(
    data_yaml: str,
    epochs:    int   = EPOCHS,
    lr0:       float = LR0,
    freeze:    int   = FREEZE,
    imgsz:     int   = IMGSZ,
    batch:     int   = BATCH,
) -> None:
    from ultralytics import YOLO

    if not MODEL_BASE.exists():
        print(f"[ERROR] No se encontró {MODEL_BASE}")
        raise SystemExit(1)

    print(f"\n{'='*65}")
    print(f"  FINE-TUNING AVANZADO — 3 fallos críticos")
    print(f"{'='*65}")
    print(f"  Modelo base  : {MODEL_BASE}")
    print(f"  Dataset      : {data_yaml}")
    print(f"  Épocas       : {epochs}")
    print(f"  lr0          : {lr0}  (muy baja — sin destruir conocimiento)")
    print(f"  freeze       : {freeze} capas  (backbone protegido)")
    print(f"  imgsz        : {imgsz}px  (objetos pequeños/lejanos)")
    print(f"  batch        : {batch}")
    print(f"{'='*65}\n")

    model = YOLO(str(MODEL_BASE))

    # ── Auto-ajuste para 920MX (2GB VRAM) ────────────────────────────────────
    # imgsz=640 + batch=8 = ~3GB VRAM → CUDA timeout.
    # Detectar VRAM disponible y ajustar automáticamente.
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb <= 2.5 and imgsz == 640 and batch >= 8:
                batch_ajustado = 4
                print(f"\n[AUTO-AJUSTE] GPU con {vram_gb:.1f}GB VRAM detectada.")
                print(f"  imgsz=640 + batch=8 = ~3GB → CUDA timeout en 920MX.")
                print(f"  Ajustando batch: {batch} → {batch_ajustado}")
                print(f"  Para usar imgsz=640 con esta GPU batch debe ser ≤4.\n")
                batch = batch_ajustado
            elif vram_gb <= 2.5 and imgsz > 416:
                imgsz_ajustado = 416
                batch_ajustado = 4
                print(f"\n[AUTO-AJUSTE] {vram_gb:.1f}GB VRAM — ajustando imgsz={imgsz}→{imgsz_ajustado}, batch={batch}→{batch_ajustado}")
                imgsz = imgsz_ajustado
                batch = batch_ajustado
    except Exception:
        pass

    results = model.train(
        data     = data_yaml,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        name     = "advanced",
        exist_ok = True,

        # ── Optimizador explícito — CRÍTICO ───────────────────────────────────
        # optimizer=auto sobreescribe lr0. Forzar SGD para respetar lr0=0.00005.
        optimizer     = "SGD",
        lr0           = lr0,        # 0.00005 — respetado solo con optimizer!=auto
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 3,

        # ── Congelamiento de capas ────────────────────────────────────────────
        freeze   = freeze,

        # ── Early stopping ────────────────────────────────────────────────────
        patience = 10,

        # ── Fallo 1: Confusión excavadora ↔ pala ──────────────────────────────
        degrees      = 20.0,
        perspective  = 0.002,
        shear        = 5.0,
        fliplr       = 0.5,
        flipud       = 0.15,

        # ── Fallo 2: Objetos pequeños/lejanos ─────────────────────────────────
        scale        = 0.8,
        mosaic       = 1.0,

        # ── Fallo 3: Objetos truncados/ocluidos ───────────────────────────────
        copy_paste   = 0.4,
        erasing      = 0.4,
        mixup        = 0.1,

        # ── General ───────────────────────────────────────────────────────────
        hsv_h        = 0.015,
        hsv_s        = 0.5,
        hsv_v        = 0.4,
        save         = True,
        cache        = False,
        verbose      = True,
        workers      = 4,
        amp          = True,   # Automatic Mixed Precision — reduce VRAM ~30%
    )

    # ── Métricas ──────────────────────────────────────────────────────────────
    if results is not None:
        try:
            m          = results.results_dict
            map50      = m.get("metrics/mAP50(B)",    "N/A")
            map5095    = m.get("metrics/mAP50-95(B)", "N/A")
            precision  = m.get("metrics/precision(B)", "N/A")
            recall     = m.get("metrics/recall(B)",    "N/A")
            if isinstance(map50, float):
                print(f"\n[MÉTRICAS]")
                print(f"  mAP50      : {map50:.4f}")
                print(f"  mAP50-95   : {map5095:.4f}")
                print(f"  Precision  : {precision:.4f}")
                print(f"  Recall     : {recall:.4f}")
        except Exception:
            pass

    # ── Guardar resultado ─────────────────────────────────────────────────────
    best_run = Path("runs/detect/advanced/weights/best.pt")
    if best_run.exists():
        shutil.copy2(best_run, MODEL_OUTPUT)
        print(f"\n[OK] Modelo guardado: {MODEL_OUTPUT}")
        print(f"\n[SIGUIENTE PASO] Valida antes de reemplazar best.pt:")
        print(f"  python -c \"")
        print(f"    from ultralytics import YOLO")
        print(f"    m_old = YOLO('models/best.pt')")
        print(f"    m_new = YOLO('models/best_advanced.pt')")
        print(f"    print('ORIGINAL:', m_old.val(data='{DATASET_BASE}').results_dict)")
        print(f"    print('NUEVO:   ', m_new.val(data='{DATASET_BASE}').results_dict)")
        print(f"  \"")
        print(f"\n[DEPLOY] Si el nuevo modelo es mejor, actívalo:")
        print(f"  # En .env:")
        print(f"  MODEL_PATH=models/best_advanced.pt")
    else:
        print(f"\n[WARN] No se encontró best.pt en {best_run}")
        for pt in Path("runs").rglob("best.pt"):
            print(f"  Disponible en: {pt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tuning avanzado YOLO — 3 fallos críticos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fallos que corrige:
  1. Confusión excavadora ↔ pala  →  hard negatives + augmentación angular
  2. Camiones pequeños/lejanos    →  imgsz=640 + mosaic + scale agresivo
  3. Objetos truncados/ocluidos   →  copy_paste + erasing + mosaic=1.0

Ejemplos:
  # Solo con dataset existente (sin casos difíciles aún)
  python finetune_advanced.py

  # Con carpeta de casos difíciles corregidos
  python finetune_advanced.py --hard-negatives data/hard_negatives/

  # CPU lento: reducir imgsz y batch
  python finetune_advanced.py --imgsz 320 --batch 4 --epochs 30

  # GPU potente
  python finetune_advanced.py --imgsz 640 --batch 16 --epochs 30
        """
    )
    parser.add_argument(
        "--hard-negatives", default=None, metavar="DIR",
        help="Carpeta con imágenes difíciles corregidas (excavadora/pala confundidos)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Épocas (default: {EPOCHS})"
    )
    parser.add_argument(
        "--lr0", type=float, default=LR0,
        help=f"Tasa de aprendizaje (default: {LR0})"
    )
    parser.add_argument(
        "--freeze", type=int, default=FREEZE,
        help=f"Capas a congelar (default: {FREEZE})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=IMGSZ,
        help=f"Tamaño imagen (default: {IMGSZ})"
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH,
        help=f"Batch size (default: {BATCH})"
    )
    parser.add_argument(
        "--sin-merge", action="store_true",
        help="Usar el dataset base sin combinar (más rápido, menos seguro)"
    )

    args = parser.parse_args()

    # Analizar distribución de clases antes de entrenar
    print("=" * 65)
    print("  Fine-Tuning Avanzado — Mining Dispatch YOLO")
    print("=" * 65)
    analizar_confusiones_dataset(DATASET_BASE.parent)

    # Construir dataset unificado
    if args.sin_merge:
        yaml_final = str(DATASET_BASE)
        print(f"\n[INFO] Usando dataset base directamente: {yaml_final}")
    else:
        yaml_final = crear_dataset_avanzado(args.hard_negatives)

    # Entrenar
    entrenar(
        data_yaml = yaml_final,
        epochs    = args.epochs,
        lr0       = args.lr0,
        freeze    = args.freeze,
        imgsz     = args.imgsz,
        batch     = args.batch,
    )


if __name__ == "__main__":
    main()
