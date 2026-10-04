"""
capturar_casos_dificiles.py — Captura frames donde la IA falla o tiene
baja confianza para usar como hard negatives en el reentrenamiento.

Casos que captura:
  1. Frames donde una clase no se detecta aunque debería estar
  2. Detecciones con confianza baja (posible oclusión o objeto grande)
  3. Frames donde hay más de 2 máquinas (oclusiones entre máquinas)

Uso:
    python capturar_casos_dificiles.py
    python capturar_casos_dificiles.py --conf 0.55  # umbral más estricto
    python capturar_casos_dificiles.py --video data/raw/video2.mp4

Genera imágenes en:
    data/hard_negatives/revisar/   ← revisar y anotar manualmente
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


CONF_BAJA     = 0.50   # detectar como confianza baja por debajo de este umbral
CADA_N_FRAMES = 3      # analizar 1 de cada 3 frames (velocidad)
MAX_CAPTURAS  = 2000   # máximo de frames a guardar


def analizar_y_capturar(video_path: str, output_dir: str, conf_umbral: float) -> None:
    from ultralytics import YOLO
    from src.config import load_config

    config = load_config()
    model  = YOLO(config.model_path)

    out_dir = Path(output_dir) / "revisar"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {w}x{h} @ {fps:.1f}fps | {total} frames")
    print(f"[INFO] Umbral confianza baja: {conf_umbral}")
    print(f"[INFO] Analizando 1 de cada {CADA_N_FRAMES} frames...")
    print(f"[INFO] Guardando en: {out_dir.resolve()}\n")

    # Estadísticas
    stats = {
        "sin_deteccion":   0,
        "confianza_baja":  0,
        "multiples":       0,
        "total_guardados": 0,
    }

    frame_idx     = 0
    ultimo_clases: set[str] = set()   # clases vistas recientemente

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Analizar solo cada N frames
            if frame_idx % CADA_N_FRAMES != 0:
                frame_idx += 1
                continue

            if stats["total_guardados"] >= MAX_CAPTURAS:
                print(f"[INFO] Límite de {MAX_CAPTURAS} capturas alcanzado.")
                break

            # Inferencia con umbral bajo para detectar objetos difíciles
            res  = model(frame, verbose=False, conf=0.25, imgsz=640)
            dets = res[0].boxes
            razon = None

            if dets is None or len(dets) == 0:
                # Caso 1: sin detecciones — potencial objeto perdido
                razon = "sin_det"
                stats["sin_deteccion"] += 1

            else:
                confs  = [float(b.conf) for b in dets]
                clases = [res[0].names[int(b.cls)] for b in dets]
                n_dets = len(dets)

                # Caso 2: confianza baja en alguna detección
                if any(c < conf_umbral for c in confs):
                    razon = f"low_conf_{min(confs):.2f}"
                    stats["confianza_baja"] += 1

                # Caso 3: múltiples máquinas (oclusiones probables)
                elif n_dets >= 2:
                    razon = f"multi_{n_dets}"
                    stats["multiples"] += 1

                # Caso 4: clase que no se ha visto recientemente reaparece
                clases_actuales = set(clases)
                if ultimo_clases and not clases_actuales.issubset(ultimo_clases):
                    if razon is None:
                        razon = "nueva_clase"
                ultimo_clases = clases_actuales

            # Guardar frame si es problemático
            if razon:
                ts      = f"{int(frame_idx/fps//60):02d}m{int(frame_idx/fps%60):02d}s"
                nombre  = f"f{frame_idx:06d}_{ts}_{razon}.jpg"
                ruta    = out_dir / nombre
                cv2.imwrite(str(ruta), frame)
                stats["total_guardados"] += 1

            frame_idx += 1

            # Progreso
            if frame_idx % 300 == 0:
                pct = frame_idx / total * 100
                print(
                    f"  {pct:5.1f}% frame {frame_idx}/{total} | "
                    f"guardados={stats['total_guardados']} "
                    f"(sin_det={stats['sin_deteccion']} "
                    f"low_conf={stats['confianza_baja']} "
                    f"multi={stats['multiples']})"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido.")
    finally:
        cap.release()

    # Resumen
    print(f"\n{'='*60}")
    print(f"  ANÁLISIS COMPLETADO")
    print(f"{'='*60}")
    print(f"  Frames analizados   : {frame_idx // CADA_N_FRAMES}")
    print(f"  Sin detección       : {stats['sin_deteccion']}")
    print(f"  Confianza baja      : {stats['confianza_baja']}")
    print(f"  Múltiples máquinas  : {stats['multiples']}")
    print(f"  Total guardados     : {stats['total_guardados']}")
    print(f"  Carpeta             : {out_dir.resolve()}")
    print(f"{'='*60}")
    print(f"""
SIGUIENTE PASO:
  1. Abre: {out_dir.resolve()}
  2. Revisa cada imagen
  3. Las que tengan máquinas sin bbox → anótalas en Roboflow
  4. Descarga el dataset con las anotaciones nuevas
  5. Ejecuta: python finetune_advanced.py --hard-negatives data/hard_negatives/
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Captura frames difíciles del video para reentrenamiento"
    )
    parser.add_argument(
        "--video", default=None,
        help="Ruta al video (default: VIDEO_PATH del .env)"
    )
    parser.add_argument(
        "--output", default="data/hard_negatives",
        help="Carpeta de salida (default: data/hard_negatives)"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_BAJA,
        help=f"Umbral de confianza baja (default: {CONF_BAJA})"
    )
    args = parser.parse_args()

    if args.video is None:
        from src.config import load_config
        args.video = load_config().video_path

    if not os.path.exists(args.video):
        print(f"[ERROR] Video no encontrado: {args.video}")
        raise SystemExit(1)

    print(f"[INFO] Video: {args.video}")
    analizar_y_capturar(args.video, args.output, args.conf)


if __name__ == "__main__":
    main()
