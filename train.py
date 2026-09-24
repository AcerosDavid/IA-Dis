"""
Script de entrenamiento del modelo YOLO26 con el dataset de Roboflow.
Ejecutar: python train.py  (siempre desde la raiz del proyecto)

- Si existe last.pt de un run anterior: hace resume automaticamente.
- Si existe models/best.pt: continua fine-tuning desde ahi.
- Si no existe ninguno: entrena desde el modelo base yolo26n.pt.

Al terminar copia best.pt a models/best.pt listo para usar en main.py.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from ultralytics import YOLO


# ── Configuracion ─────────────────────────────────────────────────────────────

DATA_YAML  = "data/processed/equipment.yolo26/data.yaml"
EPOCHS     = 50
IMG_SIZE   = 640
BATCH_SIZE = 8    # bajar a 4 si hay error de memoria
WORKERS    = 2

# Modelos en orden de prioridad
MODEL_BEST    = Path("models/best.pt")     # modelo ya entrenado (fine-tuning)
MODEL_BASE    = Path("models/yolo26n.pt")  # modelo base COCO (desde cero)

# Ultralytics guarda en runs/detect/train/weights/ cuando no se pasa project=
TRAIN_WEIGHTS = Path("runs/detect/train/weights")
LAST_PT       = TRAIN_WEIGHTS / "last.pt"
BEST_PT       = TRAIN_WEIGHTS / "best.pt"


def elegir_modelo_base() -> str:
    """
    Elige el modelo de partida en orden de prioridad:
    1. models/best.pt  (continuar fine-tuning)
    2. models/yolo26n.pt (entrenar desde modelo base COCO)
    """
    if MODEL_BEST.exists():
        print(f"[INFO] Partiendo desde modelo entrenado: {MODEL_BEST}")
        return str(MODEL_BEST)
    print(f"[INFO] Partiendo desde modelo base: {MODEL_BASE}")
    print(f"[INFO] (Si no existe, Ultralytics lo descarga automaticamente)")
    return str(MODEL_BASE)


def main() -> None:
    # Verificar dataset
    if not os.path.exists(DATA_YAML):
        print(f"[ERROR] No se encontro el dataset: {DATA_YAML}")
        print(f"[ERROR] Verifica que la carpeta data/processed/equipment.yolo26/ existe")
        raise SystemExit(1)

    # Verificar que el data.yaml tiene las rutas correctas
    data_path = Path(DATA_YAML).parent
    yaml_text = Path(DATA_YAML).read_text(encoding="utf-8")
    if "../train" in yaml_text or "../valid" in yaml_text:
        print(f"[WARN] El data.yaml usa rutas relativas (../train/images)")
        print(f"[WARN] Asegurate de ejecutar train.py desde la raiz del proyecto")

    # ── Resume si hay un run interrumpido ────────────────────────────────────
    if LAST_PT.exists():
        print(f"[INFO] Run anterior detectado: {LAST_PT}")
        print(f"[INFO] Haciendo resume desde la ultima epoca guardada...")
        model   = YOLO(str(LAST_PT))
        results = model.train(resume=True)

    # ── Entrenamiento nuevo ───────────────────────────────────────────────────
    else:
        base_model = elegir_modelo_base()

        print(f"[INFO] Dataset   : {DATA_YAML}")
        print(f"[INFO] Epocas    : {EPOCHS}")
        print(f"[INFO] Imagen    : {IMG_SIZE}px  |  Batch: {BATCH_SIZE}")
        print(f"[INFO] Guardando en: {TRAIN_WEIGHTS}")
        print("[INFO] Iniciando entrenamiento...")

        model   = YOLO(base_model)
        results = model.train(
            data       = DATA_YAML,
            epochs     = EPOCHS,
            imgsz      = IMG_SIZE,
            batch      = BATCH_SIZE,
            workers    = WORKERS,
            name       = "train",     # → runs/detect/train/weights/
            exist_ok   = True,        # no falla si ya existe la carpeta
            patience   = 20,
            save       = True,        # guardar checkpoints cada epoca
            cache      = False,
            verbose    = True,
            copy_paste = 0.3,         # mejora deteccion de equipos parciales
            scale      = 0.9,         # mejora deteccion a distintas escalas
        )

    # ── Metricas finales ──────────────────────────────────────────────────────
    if results is not None:
        try:
            m = results.results_dict
            map50    = m.get("metrics/mAP50(B)",    "N/A")
            map5095  = m.get("metrics/mAP50-95(B)", "N/A")
            precision = m.get("metrics/precision(B)", "N/A")
            recall    = m.get("metrics/recall(B)",    "N/A")
            if isinstance(map50, float):
                print(f"\n[METRICAS] mAP50={map50:.4f} | mAP50-95={map5095:.4f} "
                      f"| precision={precision:.4f} | recall={recall:.4f}")
        except Exception:
            pass

    # ── Copiar best.pt al destino final ───────────────────────────────────────
    if BEST_PT.exists():
        shutil.copy2(BEST_PT, MODEL_BEST)
        print(f"\n[OK] Modelo guardado en: {MODEL_BEST}")
        print(f"[OK] Listo para usar con: MODEL_PATH=models/best.pt en .env")
    else:
        print(f"\n[WARN] No se encontro best.pt en: {TRAIN_WEIGHTS}")
        print(f"[WARN] Busca manualmente los .pt en: runs/detect/")
        # Buscar en rutas alternativas por si Ultralytics cambio la estructura
        for pt in Path("runs").rglob("best.pt"):
            print(f"[WARN] Encontrado en: {pt}")


if __name__ == "__main__":
    main()
