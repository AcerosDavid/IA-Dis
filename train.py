"""
Script de entrenamiento del modelo YOLO26 con el dataset de Roboflow.
Ejecutar: python train.py

Genera models/best.pt listo para usar en main.py.
Si se interrumpe, vuelve a ejecutar y continua desde la ultima epoca.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from ultralytics import YOLO


# ── Configuracion del entrenamiento ──────────────────────────────────────────

DATA_YAML  = "data/processed/equipment-split/data.yaml"
BASE_MODEL = "models/best.pt"   # continuar desde el mejor modelo entrenado
EPOCHS     = 50                  # 50 epocas adicionales
IMG_SIZE   = 640                 # subir resolucion para mejor deteccion
BATCH_SIZE = 8
WORKERS    = 2

# Ultralytics siempre guarda en runs/detect/{name}/weights/
# independientemente del parametro project que se pase.
# Usamos su estructura por defecto para evitar inconsistencias.
TRAIN_DIR = Path("runs/detect/train/weights")
LAST_PT   = TRAIN_DIR / "last.pt"
BEST_PT   = TRAIN_DIR / "best.pt"

# Destino final del modelo entrenado
MODEL_DST = Path("models/best.pt")


def main() -> None:
    if not os.path.exists(DATA_YAML):
        print(f"[ERROR] No se encontro el dataset: {DATA_YAML}")
        raise SystemExit(1)

    # Si existe last.pt de una sesion anterior, hacer resume
    if LAST_PT.exists():
        print(f"[INFO] Sesion anterior detectada en: {LAST_PT}")
        print("[INFO] Continuando desde la ultima epoca guardada...")
        model   = YOLO(str(LAST_PT))
        results = model.train(resume=True)
    else:
        print(f"[INFO] Dataset   : {DATA_YAML}")
        print(f"[INFO] Modelo    : {BASE_MODEL}")
        print(f"[INFO] Epocas    : {EPOCHS}")
        print(f"[INFO] Imagen    : {IMG_SIZE}px  |  Batch: {BATCH_SIZE}")
        print("[INFO] Iniciando entrenamiento desde cero...")
        model   = YOLO(BASE_MODEL)
        results = model.train(
            data     = DATA_YAML,
            epochs   = EPOCHS,
            imgsz    = IMG_SIZE,
            batch    = BATCH_SIZE,
            workers  = WORKERS,
            name     = "train",   # → guarda en runs/detect/train/weights/
            exist_ok = True,
            patience = 15,
            save     = True,      # forzar guardado de checkpoints
            cache    = False,
            verbose  = True,
        )

    # Mostrar metricas finales
    if results is not None:
        try:
            box = results.results_dict
            print(f"\n[METRICAS] mAP50={box.get('metrics/mAP50(B)', 'N/A'):.4f} "
                  f"| mAP50-95={box.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        except Exception:
            pass  # metricas no disponibles en modo resume

    # Copiar best.pt al destino final
    if BEST_PT.exists():
        shutil.copy2(BEST_PT, MODEL_DST)
        print(f"\n[OK] Modelo guardado en: {MODEL_DST}")
        print("[OK] Cambia en .env:  MODEL_PATH=models/best.pt")
    else:
        print(f"[WARN] No se encontro {BEST_PT}")
        print(f"[WARN] Busca manualmente en: {TRAIN_DIR}")


if __name__ == "__main__":
    main()
