"""
Monitor de entrenamiento YOLO en tiempo real.
Ejecutar en una terminal separada mientras train.py corre:
    python monitor_train.py

Muestra cada 60 segundos:
  - Si last.pt se esta actualizando (epocas guardandose)
  - Epoca actual segun results.csv
  - Mejores metricas hasta ahora
  - Tiempo transcurrido y estimacion de fin
"""
from __future__ import annotations

import time
import csv
from pathlib import Path
from datetime import datetime, timedelta

# ── Rutas ─────────────────────────────────────────────────────────────────────
TRAIN_DIR   = Path("runs/detect/train")
LAST_PT     = TRAIN_DIR / "weights/last.pt"
BEST_PT     = TRAIN_DIR / "weights/best.pt"
RESULTS_CSV = TRAIN_DIR / "results.csv"

INTERVALO   = 60   # segundos entre cada chequeo
TOTAL_EPOCAS = 150  # debe coincidir con EPOCHS en train.py


def leer_ultima_epoca() -> dict | None:
    """Lee la ultima fila del results.csv y retorna las metricas."""
    if not RESULTS_CSV.exists():
        return None
    try:
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        if not filas:
            return None
        ultima = filas[-1]
        # Limpiar espacios en los nombres de columna
        return {k.strip(): v.strip() for k, v in ultima.items()}
    except Exception:
        return None


def formato_tiempo(segundos: float) -> str:
    """Convierte segundos a formato hh:mm:ss."""
    return str(timedelta(seconds=int(segundos)))


def main() -> None:
    print("=" * 60)
    print("  MONITOR DE ENTRENAMIENTO YOLO")
    print("  Ctrl+C para salir")
    print("=" * 60)

    ultimo_mtime   = None
    epocas_sin_cambio = 0
    inicio_monitor = time.time()

    while True:
        ahora     = datetime.now().strftime("%H:%M:%S")
        transcurrido = time.time() - inicio_monitor

        print(f"\n[{ahora}] ── Chequeo ──────────────────────────────")

        # ── Verificar last.pt ─────────────────────────────────────────────
        if LAST_PT.exists():
            mtime   = LAST_PT.stat().st_mtime
            tam_mb  = LAST_PT.stat().st_size / 1024 / 1024
            hace    = time.time() - mtime

            if ultimo_mtime is None or mtime != ultimo_mtime:
                estado = "✅ Actualizado"
                epocas_sin_cambio = 0
                ultimo_mtime = mtime
            else:
                epocas_sin_cambio += 1
                if epocas_sin_cambio >= 3:
                    estado = f"⚠️  Sin cambios hace {formato_tiempo(hace)} — puede estar en medio de una epoca"
                else:
                    estado = f"🔄 Sin cambios en este chequeo (hace {formato_tiempo(hace)})"

            print(f"  last.pt : {estado}")
            print(f"            Tamaño: {tam_mb:.1f} MB | "
                  f"Ultima modificacion: {datetime.fromtimestamp(mtime).strftime('%H:%M:%S')}")
        else:
            print(f"  last.pt : ❌ No existe aun — esperando que termine la primera epoca...")

        # ── Verificar best.pt ─────────────────────────────────────────────
        if BEST_PT.exists():
            mtime_best = BEST_PT.stat().st_mtime
            print(f"  best.pt : ✅ Existe | "
                  f"Ultima mejora: {datetime.fromtimestamp(mtime_best).strftime('%H:%M:%S')}")
        else:
            print(f"  best.pt : ⏳ Aun no generado")

        # ── Leer metricas del results.csv ─────────────────────────────────
        metricas = leer_ultima_epoca()
        if metricas:
            try:
                epoca_actual = int(float(metricas.get("epoch", 0)))
                map50        = float(metricas.get("metrics/mAP50(B)", 0))
                map5095      = float(metricas.get("metrics/mAP50-95(B)", 0))
                precision    = float(metricas.get("metrics/precision(B)", 0))
                recall       = float(metricas.get("metrics/recall(B)", 0))
                box_loss     = float(metricas.get("train/box_loss", 0))
                cls_loss     = float(metricas.get("train/cls_loss", 0))

                progreso = (epoca_actual / TOTAL_EPOCAS) * 100

                print(f"\n  Epoca   : {epoca_actual}/{TOTAL_EPOCAS} ({progreso:.1f}%)")
                print(f"  mAP50   : {map50:.4f}  |  mAP50-95: {map5095:.4f}")
                print(f"  Precision: {precision:.4f}  |  Recall: {recall:.4f}")
                print(f"  box_loss: {box_loss:.4f}  |  cls_loss: {cls_loss:.4f}")

                # Estimacion de tiempo restante
                if epoca_actual > 0:
                    seg_por_epoca = transcurrido / epoca_actual
                    epocas_restantes = TOTAL_EPOCAS - epoca_actual
                    seg_restantes = seg_por_epoca * epocas_restantes
                    fin_estimado = datetime.now() + timedelta(seconds=seg_restantes)
                    print(f"\n  Tiempo transcurrido : {formato_tiempo(transcurrido)}")
                    print(f"  Tiempo por epoca    : {formato_tiempo(seg_por_epoca)}")
                    print(f"  Tiempo restante     : {formato_tiempo(seg_restantes)}")
                    print(f"  Fin estimado        : {fin_estimado.strftime('%d/%m %H:%M')}")

                # Alerta si los losses no bajan
                if cls_loss > 2.0 and epoca_actual > 10:
                    print(f"\n  ⚠️  cls_loss alto ({cls_loss:.3f}) despues de {epoca_actual} epocas")
                    print(f"      Puede indicar problema con las clases del dataset")
                elif map50 > 0.85:
                    print(f"\n  🎯 Excelente mAP50: {map50:.4f} — modelo funcionando bien")
                elif map50 > 0.75:
                    print(f"\n  ✅ Buen mAP50: {map50:.4f} — sigue mejorando")

            except (ValueError, KeyError):
                print(f"  Metricas: no se pudieron parsear ({metricas})")
        else:
            print(f"\n  Metricas: ⏳ results.csv aun no existe (esperando primera epoca)")

        print(f"\n  Proximo chequeo en {INTERVALO} segundos... (Ctrl+C para salir)")
        print("─" * 60)

        try:
            time.sleep(INTERVALO)
        except KeyboardInterrupt:
            print("\n[INFO] Monitor detenido.")
            break


if __name__ == "__main__":
    main()
