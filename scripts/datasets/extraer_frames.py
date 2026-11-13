"""
extraer_frames.py — Extrae frames completos del video SIN usar YOLO.

No hace ningún análisis — solo guarda el frame tal cual.
Roboflow se encarga de auto-anotar con el modelo existente.
Tú solo corriges los errores.

Uso:
    python extraer_frames.py --video data/raw/video.mp4 --cada 300
    python extraer_frames.py --video data/raw/video.mp4 --cada 150 --output data/frames_nuevos

Para un video de 1h con --cada 300:
    ~360 frames en ~2 minutos (sin YOLO, solo lectura)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


def extraer(video_path: str, output_dir: str, cada_n: int) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {video_path}")
        return

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duracion_min = total / fps / 60

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames_esperados = total // cada_n
    print(f"[INFO] Video: {w}x{h} @ {fps:.0f}fps | {total} frames | {duracion_min:.1f} min")
    print(f"[INFO] Extrayendo 1 frame cada {cada_n} frames (~{cada_n/fps:.1f}s)")
    print(f"[INFO] Frames esperados: ~{frames_esperados}")
    print(f"[INFO] Guardando en: {out.resolve()}\n")

    frame_idx  = 0
    guardados  = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % cada_n == 0:
                # Nombre con timestamp para identificar dónde en el video
                seg   = frame_idx / fps
                mm    = int(seg // 60)
                ss    = int(seg % 60)
                nombre = f"frame_{frame_idx:07d}_{mm:02d}m{ss:02d}s.jpg"
                cv2.imwrite(str(out / nombre), frame)
                guardados += 1

            frame_idx += 1

            # Progreso cada 3000 frames
            if frame_idx % 3000 == 0:
                pct = frame_idx / total * 100
                print(f"  {pct:5.1f}% | frame {frame_idx}/{total} | guardados={guardados}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido.")
    finally:
        cap.release()

    print(f"\n{'='*50}")
    print(f"  EXTRACCIÓN COMPLETADA")
    print(f"{'='*50}")
    print(f"  Frames guardados : {guardados}")
    print(f"  Carpeta          : {out.resolve()}")
    print(f"\nSIGUIENTE PASO:")
    print(f"  1. Sube la carpeta a Roboflow")
    print(f"  2. Usa Auto-Label con tu modelo entrenado")
    print(f"  3. Corrige solo los errores (objetos tapados, etc.)")
    print(f"  4. Exporta en YOLOv8 y fine-tuning en Colab")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae frames del video sin YOLO — para anotar en Roboflow"
    )
    parser.add_argument("--video",  required=True,
                        help="Ruta al video")
    parser.add_argument("--cada",   type=int, default=300,
                        help="Extraer 1 frame cada N frames (default: 300 = ~10s a 30fps)")
    parser.add_argument("--output", default="data/frames_nuevos",
                        help="Carpeta de salida (default: data/frames_nuevos)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video no encontrado: {args.video}")
        raise SystemExit(1)

    extraer(args.video, args.output, args.cada)


if __name__ == "__main__":
    main()
