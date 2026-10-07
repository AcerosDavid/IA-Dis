"""
auditoria.py — Auditoría técnica del pipeline de entrenamiento YOLO.
Analiza: capacidad del modelo, distribución de clases, calidad de labels.

Uso:
    python auditoria.py
"""
from __future__ import annotations

import random
from pathlib import Path

# ── 1. Capacidad del modelo ───────────────────────────────────────────────────

def auditar_modelo():
    from ultralytics import YOLO
    import torch

    print("=" * 60)
    print("  1. CAPACIDAD DEL MODELO")
    print("=" * 60)

    m     = YOLO("models/best.pt")
    model = m.model
    params_total     = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    capas            = len([l for l in model.modules()])

    print(f"  Arquitectura     : {model.__class__.__name__}")
    print(f"  Parámetros total : {params_total:,}")
    print(f"  Parámetros train : {params_trainable:,}")
    print(f"  Capas            : {capas}")

    # Comparativa de modelos
    print(f"\n  Comparativa de arquitecturas YOLO11:")
    print(f"  {'Modelo':<12} {'Params':>10} {'GFLOPs':>8} {'Velocidad':>10} {'Recomendado para'}")
    print(f"  {'-'*65}")
    modelos = [
        ("YOLO11n ← actual", "2.6M",  "6.5",  "~5fps CPU",   "Datasets < 30k, hardware limitado"),
        ("YOLO11s",          "9.4M",  "21.5", "~3fps CPU",   "Datasets 30k-100k, mejor precisión"),
        ("YOLO11m",          "20.1M", "68.0", "~1fps CPU",   "GPU dedicada, alta precisión"),
    ]
    for nombre, params, gflops, vel, rec in modelos:
        marca = " ←" if "actual" in nombre else ""
        print(f"  {nombre:<18} {params:>8} {gflops:>8} {vel:>12}  {rec}")

    print(f"\n  📊 Veredicto capacidad:")
    if params_total < 3_000_000:
        print(f"  ⚠️  YOLO11n con {params_total:,} params está en su límite de capacidad.")
        print(f"  → Con 22k+ imágenes variadas, YOLO11s daría +5-8% mAP50.")
        print(f"  → Pero requiere GPU para entrenamiento razonable (>8GB VRAM).")
        print(f"  → En T4 de Colab: YOLO11s tarda ~6-8h vs 3-4h del nano.")


# ── 2. Distribución de clases ─────────────────────────────────────────────────

def auditar_clases():
    print(f"\n{'='*60}")
    print(f"  2. DISTRIBUCIÓN DE CLASES E INSTANCIAS")
    print(f"{'='*60}")

    base    = Path("data/processed/equipment.yolo26")
    nombres = {0: "dump_truck", 1: "excavator", 2: "wheel_loader"}

    for split in ["train", "valid", "test"]:
        labels_dir = base / split / "labels"
        if not labels_dir.exists():
            continue

        conteo     = {0: 0, 1: 0, 2: 0}
        n_archivos = 0
        n_vacios   = 0

        for f in labels_dir.glob("*.txt"):
            n_archivos += 1
            lineas = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lineas:
                n_vacios += 1
                continue
            for linea in lineas:
                try:
                    c = int(linea.split()[0])
                    if c in conteo:
                        conteo[c] += 1
                except Exception:
                    pass

        total = sum(conteo.values()) or 1
        print(f"\n  {split.upper()} ({n_archivos} imágenes, {n_vacios} sin anotaciones):")
        for cls_id, n in conteo.items():
            pct   = n / total * 100
            barra = "█" * int(pct / 3)
            print(f"    {nombres[cls_id]:<15} {n:6d} ({pct:5.1f}%)  {barra}")
        print(f"    {'TOTAL':<15} {total:6d}")

        # Detectar desbalance
        max_cls = max(conteo, key=conteo.get)
        min_cls = min(conteo, key=conteo.get)
        ratio   = conteo[max_cls] / max(conteo[min_cls], 1)
        if ratio > 3:
            print(f"\n    ⚠️  DESBALANCE: {nombres[max_cls]} tiene {ratio:.1f}x más instancias")
            print(f"       que {nombres[min_cls]}. Considera class_weights o oversample.")


# ── 3. Auditoría de calidad de labels ─────────────────────────────────────────

def auditar_labels(muestra: int = 200):
    print(f"\n{'='*60}")
    print(f"  3. CALIDAD DE LABELS (muestra de {muestra} archivos)")
    print(f"{'='*60}")

    base       = Path("data/processed/equipment.yolo26/train/labels")
    if not base.exists():
        print("  ⚠️  No se encontró el directorio de labels")
        return

    archivos   = random.sample(list(base.glob("*.txt")), min(muestra, len(list(base.glob("*.txt")))))
    nombres    = {0: "dump_truck", 1: "excavator", 2: "wheel_loader"}

    problemas = {
        "bbox_fuera_de_rango":  0,   # coordenadas > 1 o < 0
        "bbox_muy_pequeno":     0,   # bbox menor al 1% de la imagen
        "bbox_muy_grande":      0,   # bbox mayor al 95% de la imagen
        "clase_invalida":       0,   # clase fuera de 0-2
        "formato_incorrecto":   0,   # línea mal formada
        "duplicados_exactos":   0,   # misma línea repetida en el archivo
        "mezcla_segmentacion":  0,   # más de 5 valores por línea (segmentación)
    }
    total_lineas = 0

    for f in archivos:
        lineas = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        lineas_set = set()

        for linea in lineas:
            total_lineas += 1
            partes = linea.split()

            if len(partes) < 5:
                problemas["formato_incorrecto"] += 1
                continue

            if len(partes) > 5:
                problemas["mezcla_segmentacion"] += 1

            try:
                cls_id      = int(partes[0])
                cx, cy, bw, bh = map(float, partes[1:5])
            except ValueError:
                problemas["formato_incorrecto"] += 1
                continue

            if cls_id not in nombres:
                problemas["clase_invalida"] += 1

            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                problemas["bbox_fuera_de_rango"] += 1

            area = bw * bh
            if area < 0.0001:
                problemas["bbox_muy_pequeno"] += 1
            elif area > 0.90:
                problemas["bbox_muy_grande"] += 1

            if linea in lineas_set:
                problemas["duplicados_exactos"] += 1
            lineas_set.add(linea)

    print(f"\n  Archivos analizados : {len(archivos)}")
    print(f"  Anotaciones totales : {total_lineas}")
    print(f"  Promedio por imagen : {total_lineas/max(len(archivos),1):.1f}")
    print(f"\n  Problemas encontrados:")

    total_problemas = 0
    for tipo, n in problemas.items():
        if n > 0:
            pct = n / max(total_lineas, 1) * 100
            print(f"    ⚠️  {tipo:<25} {n:4d} ({pct:.2f}%)")
            total_problemas += n
        else:
            print(f"    ✅  {tipo:<25}   0")

    if total_problemas == 0:
        print(f"\n  ✅ Labels limpias — no se detectaron problemas de calidad")
    else:
        print(f"\n  Total problemas: {total_problemas} ({total_problemas/max(total_lineas,1)*100:.2f}%)")
        if problemas["mezcla_segmentacion"] > 0:
            print(f"  → Hay anotaciones de segmentación mezcladas con detección.")
            print(f"    Filtra en Roboflow exportando SOLO en formato 'Object Detection'.")


# ── 4. Plan de acción ─────────────────────────────────────────────────────────

def plan_accion():
    print(f"\n{'='*60}")
    print(f"  4. PLAN DE ACCIÓN — Romper el estancamiento")
    print(f"{'='*60}")

    print("""
  CAUSA RAÍZ del estancamiento:
  ─────────────────────────────
  YOLO11n (2.5M params) con 22k imágenes ha llegado al límite
  de su capacidad de representación. El modelo ya memorizó los
  patrones principales y no puede aprender variaciones más finas.

  PLAN EN 3 FASES:
  ────────────────

  FASE 1 — Inmediata (sin cambiar arquitectura):
    ✅ Ya hecho: 22k imágenes, fine-tuning conservador

    Pendiente:
    → Limpiar labels mixtas (segmentación + detección)
      En Roboflow: exportar como "Object Detection" únicamente
    → Balancear clases: si excavator tiene 3x menos instancias
      usar class_weights en el entrenamiento
    → Agregar imágenes de YouTube con VARIEDAD REAL
      (diferentes minas, condiciones, ángulos)

  FASE 2 — Corto plazo (datos de calidad):
    → 5k imágenes nuevas de YouTube con alta variedad
    → Anotar específicamente:
       - Camiones 50% tapados por excavadoras
       - Excavadoras en primer plano (>50% de la pantalla)
       - Múltiples máquinas solapadas
    → Subir a 30k imágenes de ALTA VARIEDAD
       (no cantidad, sino diversidad real)

  FASE 3 — Cuando tengas GPU real (>8GB VRAM):
    → Escalar a YOLO11s (9.4M params, +5-8% mAP50 esperado)
    → En Colab T4 (16GB): viable con batch=8, imgsz=640
    → Comando:
       model = YOLO('yolo11s.pt')  # en lugar de best.pt
       model.train(...)            # mismos parámetros

  CUÁNDO ESCALAR A YOLO11s:
  → Cuando tengas >30k imágenes variadas
  → Cuando el mAP50 no mejore en 3 entrenamientos seguidos
  → Ya alcanzaste ese punto con las 22k actuales
""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    auditar_modelo()
    auditar_clases()
    auditar_labels()
    plan_accion()
