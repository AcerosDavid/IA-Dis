"""
entrenar_clasificador.py — Entrena un clasificador de estado para maquinaria minera.

Entrena 3 modelos independientes (uno por tipo de máquina):
  - camion:     empty / loaded
  - excavadora: digging / loading / idle
  - pala:       scooping / dumping / idle

Requiere que data/estados/ tenga la siguiente estructura (creada manualmente):
  data/estados/
  ├── camion/
  │   ├── empty/        (imágenes de tolva vacía)
  │   └── loaded/       (imágenes de tolva llena)
  ├── excavadora/
  │   ├── digging/
  │   ├── loading/
  │   └── idle/
  └── pala/
      ├── scooping/
      ├── dumping/
      └── idle/

Uso:
  python entrenar_clasificador.py              # entrena los 3 modelos
  python entrenar_clasificador.py --tipo camion  # solo camion
  python entrenar_clasificador.py --epochs 30    # más épocas

Resultado:
  models/estado_camion.pt
  models/estado_excavadora.pt
  models/estado_pala.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ── Configuración ─────────────────────────────────────────────────────────────
ESTADOS_DIR  = Path("data/estados")
MODELS_DIR   = Path("models")
EPOCHS       = 25
BATCH_SIZE   = 32
LR           = 0.001
IMG_SIZE     = 224
VAL_SPLIT    = 0.2       # 20% para validación
MIN_IMAGENES = 20        # mínimo de imágenes por clase para entrenar

TIPOS = {
    "camion":     ["empty", "loaded", "dumping"],
    "excavadora": ["digging", "loading_truck"],
    "pala":       ["scooping", "dumping_truck"],
}


# ── Augmentación ──────────────────────────────────────────────────────────────

def get_transforms(entrenamiento: bool):
    if entrenamiento:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


# ── Modelo ────────────────────────────────────────────────────────────────────

def crear_modelo(num_clases: int) -> nn.Module:
    """
    MobileNetV3-Small preentrenado en ImageNet.
    Ligero (~10MB), rápido (~2ms en CPU), bueno para clasificación de estado.
    """
    modelo = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Reemplazar la última capa de clasificación
    in_features = modelo.classifier[-1].in_features
    modelo.classifier[-1] = nn.Linear(in_features, num_clases)
    return modelo


# ── Validar dataset ───────────────────────────────────────────────────────────

def validar_dataset(tipo: str) -> bool:
    """Verifica que el dataset del tipo tiene suficientes imágenes."""
    base     = ESTADOS_DIR / tipo
    clases   = TIPOS[tipo]
    valido   = True
    total    = 0

    print(f"\n[VALIDACION] Dataset {tipo}:")
    for clase in clases:
        ruta = base / clase
        if not ruta.exists():
            print(f"  ✗ {clase:<15} — carpeta no encontrada: {ruta}")
            valido = False
            continue

        imgs = [f for f in ruta.glob("*")
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        n    = len(imgs)
        total += n

        if n < MIN_IMAGENES:
            print(f"  ⚠ {clase:<15} {n:4d} imgs  (mínimo recomendado: {MIN_IMAGENES})")
            if n == 0:
                valido = False
        else:
            print(f"  ✓ {clase:<15} {n:4d} imgs")

    print(f"  Total: {total} imágenes")
    return valido


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def entrenar_tipo(
    tipo:    str,
    epochs:  int   = EPOCHS,
    batch:   int   = BATCH_SIZE,
    lr:      float = LR,
) -> None:
    """Entrena el clasificador para un tipo de máquina."""
    base_dir = ESTADOS_DIR / tipo
    clases   = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])

    if not clases:
        print(f"[ERROR] No se encontraron clases en {base_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Entrenando clasificador: {tipo.upper()}")
    print(f"{'='*60}")
    print(f"  Clases   : {clases}")
    print(f"  Device   : {device}")
    print(f"  Épocas   : {epochs}")
    print(f"  Batch    : {batch}")
    print(f"  lr       : {lr}")

    # Dataset completo con augmentación de entrenamiento
    dataset_full = datasets.ImageFolder(
        str(base_dir),
        transform=get_transforms(entrenamiento=True),
    )

    # Verificar que las clases coinciden
    print(f"  Clases detectadas: {dataset_full.classes}")

    n_total = len(dataset_full)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    if n_train < 1:
        print(f"[ERROR] No hay suficientes imágenes para entrenar ({n_total} total)")
        return

    train_ds, val_ds = random_split(
        dataset_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Dataset de validación sin augmentación
    val_ds.dataset = datasets.ImageFolder(
        str(base_dir),
        transform=get_transforms(entrenamiento=False),
    )

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=2, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"))

    print(f"  Train    : {n_train} imgs | Val: {n_val} imgs\n")

    # Modelo
    modelo    = crear_modelo(len(clases)).to(device)
    criterio  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    mejor_acc  = 0.0
    mejor_path = MODELS_DIR / f"estado_{tipo}.pt"
    historial  = []

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        modelo.train()
        loss_train = 0.0
        correctos  = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            salida = modelo(imgs)
            loss   = criterio(salida, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item() * imgs.size(0)
            correctos  += (salida.argmax(1) == labels).sum().item()

        acc_train  = correctos / n_train
        loss_train = loss_train / n_train

        # ── Validación ─────────────────────────────────────────────────────
        modelo.eval()
        loss_val  = 0.0
        correctos = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                salida    = modelo(imgs)
                loss_val += criterio(salida, labels).item() * imgs.size(0)
                correctos += (salida.argmax(1) == labels).sum().item()

        acc_val  = correctos / n_val
        loss_val = loss_val / n_val

        scheduler.step(acc_val)

        # Guardar mejor modelo
        marca = ""
        if acc_val > mejor_acc:
            mejor_acc = acc_val
            torch.save({
                "modelo_state": modelo.state_dict(),
                "clases":       dataset_full.classes,
                "tipo":         tipo,
                "img_size":     IMG_SIZE,
                "acc_val":      acc_val,
                "epoch":        epoch,
            }, str(mejor_path))
            marca = " ← mejor"

        historial.append({
            "epoch": epoch, "loss_train": loss_train,
            "acc_train": acc_train, "loss_val": loss_val, "acc_val": acc_val,
        })

        print(
            f"  Época {epoch:3d}/{epochs}  "
            f"loss_train={loss_train:.4f}  acc_train={acc_train:.3f}  "
            f"loss_val={loss_val:.4f}  acc_val={acc_val:.3f}{marca}"
        )

    print(f"\n  Mejor acc_val: {mejor_acc:.4f} ({mejor_acc*100:.1f}%)")
    print(f"  Modelo guardado: {mejor_path}")

    # Guardar historial
    hist_path = MODELS_DIR / f"estado_{tipo}_historial.json"
    with open(hist_path, "w") as f:
        json.dump(historial, f, indent=2)

    # Reporte por clase
    _reporte_por_clase(modelo, val_loader, dataset_full.classes, device)


def _reporte_por_clase(modelo, loader, clases, device) -> None:
    """Calcula accuracy por clase en el set de validación."""
    modelo.eval()
    correctos_clase = {c: 0 for c in clases}
    totales_clase   = {c: 0 for c in clases}

    with torch.no_grad():
        for imgs, labels in loader:
            imgs    = imgs.to(device)
            salidas = modelo(imgs).cpu()
            preds   = salidas.argmax(1)
            for pred, label in zip(preds, labels):
                nombre_clase = clases[label]
                totales_clase[nombre_clase]   += 1
                if pred == label:
                    correctos_clase[nombre_clase] += 1

    print(f"\n  Accuracy por clase:")
    for clase in clases:
        total = totales_clase[clase]
        if total == 0:
            continue
        acc = correctos_clase[clase] / total
        barra = "█" * int(acc * 20)
        print(f"    {clase:<15} {acc*100:5.1f}%  {barra}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena clasificadores de estado para maquinaria minera"
    )
    parser.add_argument(
        "--tipo", choices=list(TIPOS.keys()) + ["todos"], default="todos",
        help="Tipo de máquina a entrenar (default: todos)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Épocas de entrenamiento (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--lr", type=float, default=LR,
        help=f"Learning rate (default: {LR})"
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tipos_a_entrenar = list(TIPOS.keys()) if args.tipo == "todos" else [args.tipo]

    print("=" * 60)
    print("  Clasificador de Estado — Maquinaria Minera")
    print("=" * 60)
    print(f"  Tipos a entrenar: {tipos_a_entrenar}")
    print(f"  Dataset base    : {ESTADOS_DIR.resolve()}")
    print(f"  Modelos output  : {MODELS_DIR.resolve()}")

    # Validar datasets antes de entrenar
    tipos_validos = []
    for tipo in tipos_a_entrenar:
        if validar_dataset(tipo):
            tipos_validos.append(tipo)
        else:
            print(f"\n[SKIP] {tipo} — dataset incompleto, saltando")

    if not tipos_validos:
        print("\n[ERROR] Ningún tipo tiene dataset válido.")
        print("  Crea las carpetas y agrega imágenes en:")
        print(f"  {ESTADOS_DIR.resolve()}/")
        for tipo in tipos_a_entrenar:
            for clase in TIPOS[tipo]:
                print(f"    {tipo}/{clase}/  ← mínimo {MIN_IMAGENES} imágenes")
        raise SystemExit(1)

    # Entrenar
    for tipo in tipos_validos:
        entrenar_tipo(tipo, epochs=args.epochs, batch=args.batch, lr=args.lr)

    # Resumen final
    print(f"\n{'='*60}")
    print("  MODELOS ENTRENADOS")
    print(f"{'='*60}")
    for tipo in tipos_validos:
        ruta = MODELS_DIR / f"estado_{tipo}.pt"
        if ruta.exists():
            ckpt = torch.load(str(ruta), map_location="cpu", weights_only=False)
            print(
                f"  {tipo:<15} → {ruta.name}"
                f"  (acc={ckpt['acc_val']:.3f}, clases={ckpt['clases']})"
            )

    print(f"\n  Siguiente paso:")
    print(f"    python main.py")
    print(f"  El sistema usará automáticamente los clasificadores.")


if __name__ == "__main__":
    main()
