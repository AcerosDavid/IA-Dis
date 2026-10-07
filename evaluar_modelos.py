"""
evaluar_modelos.py — Compara dos modelos YOLO en datasets separados.

Responde 3 preguntas:
  1. ¿Aprendió los casos nuevos?        → test en imágenes nuevas
  2. ¿Conservó lo que sabía?            → test en imágenes originales
  3. ¿Mejoró globalmente?               → test en ambos

Uso:
    # Comparar best.pt vs best_advanced.pt en el dataset original
    python evaluar_modelos.py

    # Con datasets separados (original vs nuevos ángulos)
    python evaluar_modelos.py --test-orig data/test_original --test-nuevo data/test_nuevos

Resultado:
    ┌─────────────────────────────────────────────────────────────────┐
    │              best.pt    best_advanced.pt    Diferencia          │
    ├─────────────────────────────────────────────────────────────────┤
    │ Originales  mAP50=0.877    mAP50=0.862     -0.015 ⚠️            │
    │ Nuevos      mAP50=0.621    mAP50=0.950     +0.329 ✅✅          │
    │ Global      mAP50=0.821    mAP50=0.924     +0.103 ✅            │
    └─────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import yaml

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_ORIGINAL  = Path("models/best_v1.pt")   # modelo anterior (renombrar si es diferente)
MODEL_NUEVO     = Path("models/best_advanced.pt")  # modelo nuevo

# Dataset completo (para evaluación global)
DATASET_BASE    = Path("data/processed/equipment.yolo26/data.yaml")

# Dataset de casos difíciles (las 2k nuevas anotadas)
# Si no tienes separados, se usa el test split del dataset principal
DATASET_NUEVOS  = Path("data/hard_negatives")


# ── Crear data.yaml temporal con rutas absolutas ─────────────────────────────

def crear_yaml_temporal(img_dir: Path, lbl_dir: Path, nombres_clases: list[str]) -> str:
    """Crea un data.yaml temporal apuntando a un directorio específico."""
    tmp = tempfile.mkdtemp()
    yaml_path = os.path.join(tmp, "eval.yaml")

    with open(yaml_path, "w") as f:
        f.write(f"train: {img_dir.resolve()}\n")
        f.write(f"val:   {img_dir.resolve()}\n\n")
        f.write(f"nc: {len(nombres_clases)}\n")
        f.write(f"names: {nombres_clases}\n")

    return yaml_path


def crear_yaml_desde_base(data_yaml: Path) -> str:
    """Crea yaml con rutas absolutas desde un data.yaml existente."""
    with open(data_yaml, encoding="utf-8") as f:
        orig = yaml.safe_load(f)

    base_dir  = data_yaml.parent
    tmp       = tempfile.mkdtemp()
    yaml_path = os.path.join(tmp, "eval.yaml")

    # Resolver rutas al directorio test o valid
    test_dir = base_dir / "test" / "images"
    val_dir  = base_dir / "valid" / "images"
    eval_dir = test_dir if test_dir.exists() else val_dir

    with open(yaml_path, "w") as f:
        f.write(f"train: {eval_dir.resolve()}\n")
        f.write(f"val:   {eval_dir.resolve()}\n\n")
        f.write(f"nc: {orig['nc']}\n")
        f.write(f"names: {orig['names']}\n")

    return yaml_path, orig["names"]


# ── Evaluación ────────────────────────────────────────────────────────────────

def evaluar_modelo(modelo_path: Path, data_yaml: str, nombre_modelo: str) -> dict:
    """Corre validación y retorna métricas."""
    from ultralytics import YOLO

    if not modelo_path.exists():
        print(f"  [SKIP] {modelo_path} no encontrado")
        return {}

    print(f"  Evaluando {nombre_modelo}...")
    model = YOLO(str(modelo_path))

    try:
        results = model.val(
            data    = data_yaml,
            verbose = False,
            plots   = False,
            save    = False,
        )
        m = results.results_dict
        clases = results.names

        metricas = {
            "map50":      m.get("metrics/mAP50(B)",    0),
            "map5095":    m.get("metrics/mAP50-95(B)", 0),
            "precision":  m.get("metrics/precision(B)", 0),
            "recall":     m.get("metrics/recall(B)",    0),
            "por_clase":  {},
        }

        # Métricas por clase
        if hasattr(results, "ap_class_index") and results.ap_class_index is not None:
            for i, cls_idx in enumerate(results.ap_class_index):
                nombre_cls = clases.get(int(cls_idx), str(cls_idx))
                if hasattr(results, "box") and hasattr(results.box, "ap50"):
                    metricas["por_clase"][nombre_cls] = {
                        "map50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0,
                    }

        return metricas
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {}


# ── Reporte ───────────────────────────────────────────────────────────────────

def imprimir_reporte(
    nombre_test:  str,
    orig_metrics: dict,
    nuevo_metrics: dict,
    model_orig:   str,
    model_nuevo:  str,
) -> None:
    if not orig_metrics or not nuevo_metrics:
        return

    print(f"\n  {'─'*65}")
    print(f"  {nombre_test}")
    print(f"  {'─'*65}")
    print(f"  {'Métrica':<20} {model_orig:>18} {model_nuevo:>18} {'Δ':>8}")
    print(f"  {'─'*65}")

    metricas = [
        ("mAP50",     "map50"),
        ("mAP50-95",  "map5095"),
        ("Precision",  "precision"),
        ("Recall",     "recall"),
    ]

    for nombre, key in metricas:
        v_orig  = orig_metrics.get(key, 0)
        v_nuevo = nuevo_metrics.get(key, 0)
        delta   = v_nuevo - v_orig
        emoji   = "✅" if delta > 0.005 else ("⚠️ " if delta < -0.005 else "➖")
        signo   = "+" if delta >= 0 else ""
        print(
            f"  {nombre:<20} {v_orig:>18.4f} {v_nuevo:>18.4f} "
            f"{signo}{delta:>6.4f} {emoji}"
        )

    # Por clase
    if orig_metrics.get("por_clase") and nuevo_metrics.get("por_clase"):
        print(f"\n  Por clase (mAP50):")
        todas_clases = set(orig_metrics["por_clase"]) | set(nuevo_metrics["por_clase"])
        for cls in sorted(todas_clases):
            v_o = orig_metrics["por_clase"].get(cls, {}).get("map50", 0)
            v_n = nuevo_metrics["por_clase"].get(cls, {}).get("map50", 0)
            delta = v_n - v_o
            emoji = "✅" if delta > 0.01 else ("⚠️ " if delta < -0.01 else "➖")
            signo = "+" if delta >= 0 else ""
            print(
                f"    {cls:<18} {v_o:>18.4f} {v_n:>18.4f} "
                f"{signo}{delta:>6.4f} {emoji}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara dos modelos YOLO en datasets separados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Comparación básica con dataset existente
  python evaluar_modelos.py

  # Con rutas personalizadas
  python evaluar_modelos.py --orig models/best.pt --nuevo models/best_advanced.pt

  # Con datasets separados
  python evaluar_modelos.py --test-nuevo data/hard_negatives/
        """
    )
    parser.add_argument("--orig",       default=str(MODEL_ORIGINAL),
                        help=f"Modelo original (default: {MODEL_ORIGINAL})")
    parser.add_argument("--nuevo",      default=str(MODEL_NUEVO),
                        help=f"Modelo nuevo (default: {MODEL_NUEVO})")
    parser.add_argument("--dataset",    default=str(DATASET_BASE),
                        help="data.yaml del dataset principal")
    parser.add_argument("--test-nuevo", default=None,
                        help="Carpeta con imágenes de los casos nuevos")
    args = parser.parse_args()

    model_orig  = Path(args.orig)
    model_nuevo = Path(args.nuevo)

    print("=" * 65)
    print("  Evaluación comparativa de modelos YOLO")
    print("=" * 65)
    print(f"  Modelo original : {model_orig}")
    print(f"  Modelo nuevo    : {model_nuevo}")

    # Verificar que existen
    if not model_orig.exists():
        print(f"\n[WARN] {model_orig} no existe.")
        print(f"  Renombra tu modelo anterior como: {model_orig}")
        print(f"  Ejemplo: Copy-Item models\\best_v1.pt models\\best_original.pt")

    if not model_nuevo.exists():
        print(f"\n[ERROR] {model_nuevo} no existe.")
        raise SystemExit(1)

    # Crear yaml con rutas absolutas
    print(f"\n[INFO] Preparando datasets...")
    yaml_global, clases = crear_yaml_desde_base(Path(args.dataset))
    print(f"  Dataset global: {Path(args.dataset).parent}")
    print(f"  Clases: {clases}")

    nombre_orig  = model_orig.stem
    nombre_nuevo = model_nuevo.stem

    # ── Evaluación global ──────────────────────────────────────────────────────
    print(f"\n[1/2] Evaluación en dataset completo...")
    m_orig_global  = evaluar_modelo(model_orig,  yaml_global, nombre_orig)
    m_nuevo_global = evaluar_modelo(model_nuevo, yaml_global, nombre_nuevo)
    imprimir_reporte("DATASET GLOBAL", m_orig_global, m_nuevo_global,
                     nombre_orig, nombre_nuevo)

    # ── Evaluación en casos nuevos (si se provee) ──────────────────────────────
    if args.test_nuevo:
        nuevo_dir = Path(args.test_nuevo)
        img_dir   = nuevo_dir / "images" if (nuevo_dir / "images").exists() else nuevo_dir

        if img_dir.exists() and any(img_dir.glob("*.jpg")):
            lbl_dir = nuevo_dir / "labels" if (nuevo_dir / "labels").exists() else None

            if lbl_dir and lbl_dir.exists():
                print(f"\n[2/2] Evaluación en casos nuevos ({img_dir})...")
                yaml_nuevo = crear_yaml_temporal(img_dir, lbl_dir, clases)
                m_orig_nuevo  = evaluar_modelo(model_orig,  yaml_nuevo, nombre_orig)
                m_nuevo_nuevo = evaluar_modelo(model_nuevo, yaml_nuevo, nombre_nuevo)
                imprimir_reporte("CASOS NUEVOS (ángulos difíciles)",
                                 m_orig_nuevo, m_nuevo_nuevo,
                                 nombre_orig, nombre_nuevo)
            else:
                print(f"\n[SKIP] No se encontró carpeta labels/ en {nuevo_dir}")
                print(f"  Las imágenes necesitan anotaciones para evaluar")
        else:
            print(f"\n[SKIP] No se encontraron imágenes en {img_dir}")

    # ── Veredicto final ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  VEREDICTO")
    print(f"{'='*65}")

    if m_orig_global and m_nuevo_global:
        delta = m_nuevo_global["map50"] - m_orig_global.get("map50", 0)
        if delta > 0.005:
            print(f"  ✅ El modelo nuevo es MEJOR en {delta:+.4f} mAP50")
            print(f"  → Usa {model_nuevo} como modelo principal")
            print(f"  → Actualiza .env: MODEL_PATH={model_nuevo}")
        elif delta < -0.005:
            print(f"  ⚠️  El modelo nuevo es PEOR en {delta:+.4f} mAP50")
            print(f"  → Quédate con {model_orig}")
            print(f"  → Analiza qué clases empeoraron y añade más datos de esas clases")
        else:
            print(f"  ➖ Sin diferencia significativa ({delta:+.4f} mAP50)")
            print(f"  → Ambos modelos son equivalentes")
            print(f"  → Si el nuevo detecta mejor en el video real, úsalo")

    print()


if __name__ == "__main__":
    main()
