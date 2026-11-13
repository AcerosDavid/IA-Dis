"""
entrenar_lstm.py — Entrenamiento del clasificador LSTM de estados operacionales.

Flujo:
  1. Carga secuencias pre-extraídas desde data/secuencias_{tipo}.npz
     (generadas por el etiquetador: etiquetar_video.py)
  2. Divide en train/val (80/20).
  3. Entrena LSTMEstados con Early Stopping.
  4. Guarda el mejor modelo en weights/lstm_{tipo}.pt

Uso:
    # Entrenar con .npz ya preparado
    python entrenar_lstm.py --tipo excavadora

    # Entrenar todos los tipos disponibles
    python entrenar_lstm.py --tipo todos

    # Modo demo (datos sintéticos — verifica que la arquitectura converge)
    python entrenar_lstm.py --tipo excavadora --demo

Estructura esperada del .npz (generado por etiquetar_video.py):
    X:      float32 (N, 16, 576)   — secuencias de vectores
    y:      int64   (N,)           — índice de clase
    clases: array de strings       — nombres de los estados
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Hiperparámetros ───────────────────────────────────────────────────────────

WINDOW_SIZE = 16
FEATURE_DIM = 576
VAL_SPLIT   = 0.2
EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 12


def entrenar_tipo(tipo: str, demo: bool = False, device: str = "cpu") -> None:
    from src.ai.lstm_estados import (
        LSTMEstados, DatasetSecuencias, EntrenadorLSTM, ESTADOS_POR_TIPO
    )

    print(f"\n{'='*60}")
    print(f"  Entrenando LSTM: {tipo.upper()}")
    print(f"{'='*60}\n")

    # ── Cargar datos ──────────────────────────────────────────────────────────
    ruta_npz = Path("data") / f"secuencias_{tipo}.npz"

    if not demo and ruta_npz.exists():
        print(f"[INFO] Cargando: {ruta_npz}")
        ds_completo = DatasetSecuencias.desde_npz(ruta_npz)
        X      = ds_completo.x.numpy()
        y      = ds_completo.y.numpy()
        clases = ds_completo.clases

    else:
        # ── Modo DEMO: datos sintéticos para verificar arquitectura ────────────
        if not demo:
            print(f"[WARN] No se encontró {ruta_npz}")
            print(f"[WARN] Ejecuta primero: python etiquetar_video.py")
            print(f"[WARN] Usando datos sintéticos para demo...\n")

        clases      = ESTADOS_POR_TIPO[tipo]
        n_por_clase = 80
        N           = n_por_clase * len(clases)
        X = np.random.randn(N, WINDOW_SIZE, FEATURE_DIM).astype(np.float32)
        y = np.repeat(np.arange(len(clases)), n_por_clase).astype(np.int64)
        # Hacer clases separables para verificar convergencia
        for i in range(len(clases)):
            X[y == i] += i * 0.6

    # ── Split train / val ─────────────────────────────────────────────────────
    idx      = np.random.permutation(len(X))
    n_val    = max(1, int(len(X) * VAL_SPLIT))
    ds_train = DatasetSecuencias(X[idx[n_val:]], y[idx[n_val:]], clases, augment=True)
    ds_val   = DatasetSecuencias(X[idx[:n_val]], y[idx[:n_val]], clases, augment=False)

    print(f"  Train: {len(ds_train)} | Val: {len(ds_val)} | Clases: {clases}")

    # ── Modelo ────────────────────────────────────────────────────────────────
    modelo = LSTMEstados.crear(tipo, feature_dim=FEATURE_DIM)
    print(f"  Arquitectura: {modelo}\n")

    ruta_pt = Path("weights") / f"lstm_{tipo}.pt"
    ruta_pt.parent.mkdir(exist_ok=True)

    trainer = EntrenadorLSTM(
        modelo, ds_train, ds_val,
        lr=LR, batch_size=BATCH_SIZE, device=device,
    )
    hist = trainer.entrenar(epochs=EPOCHS, patience=PATIENCE, ruta_guardado=ruta_pt)

    # Guardar aunque no haya val (modo demo sin val_acc)
    if not ruta_pt.exists():
        modelo.guardar(ruta_pt, acc_val=0.0)

    mejor_acc = max(hist.get("val_acc", [0.0]), default=0.0)
    print(f"\n  ✓ Listo | mejor acc_val={mejor_acc:.3f} | guardado en: {ruta_pt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento LSTM — estados operacionales maquinaria minera"
    )
    parser.add_argument(
        "--tipo", default="todos",
        choices=["excavadora", "pala", "camion", "todos"],
        help="Tipo de máquina a entrenar (default: todos)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Dispositivo (default: cpu)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Usar datos sintéticos para verificar la arquitectura",
    )
    args = parser.parse_args()

    tipos = ["excavadora", "pala", "camion"] if args.tipo == "todos" else [args.tipo]

    for tipo in tipos:
        entrenar_tipo(tipo, demo=args.demo, device=args.device)

    print("\n✓ Entrenamiento completado.")


if __name__ == "__main__":
    main()
