"""
extraer_muestras.py — Paso 1: Extrae recortes de maquinaria desde un video.

No etiqueta nada. Solo recorre el video cada N segundos, detecta las
máquinas con YOLO y guarda los recortes en:

    data/muestras_sin_clasificar/{tipo}/frame_{N}_id{ID}_{HHmmss}.jpg

Después usas organizar_recortes.py para clasificarlos con teclas.

INTERVALO AUTOMÁTICO:
  El script calcula el intervalo de muestreo según la duración del video:
    < 5 min   →  cada 1 segundo   (~300 recortes potenciales por clase)
    5-30 min  →  cada 5 segundos
    30-60 min →  cada 15 segundos
    > 60 min  →  cada 30 segundos
  Puedes forzarlo con --intervalo.

USO:
    python extraer_muestras.py                        # usa VIDEO_PATH del .env
    python extraer_muestras.py --video data/raw/v.mp4
    python extraer_muestras.py --video data/raw/v.mp4 --intervalo 10
    python extraer_muestras.py --video data/raw/v.mp4 --yolo models/best_2.pt
    python extraer_muestras.py --video data/raw/v.mp4 --max 500
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

# ── Configuración ─────────────────────────────────────────────────────────────

IMGSZ      = 320    # resolución de inferencia YOLO
CONF_YOLO  = 0.40   # umbral mínimo de detección
MARGEN_ROI = 12     # píxeles de margen alrededor del bbox

# Alias YOLO → tipo canónico
_YOLO_A_TIPO: dict[str, str] = {
    "dump_truck":   "camion",
    "excavator":    "excavadora",
    "wheel_loader": "pala",
    "dump":         "camion",
    "truck":        "camion",
}

# Tipos válidos para crear subcarpetas
TIPOS_VALIDOS = ["camion", "excavadora", "pala"]


def _leer_env() -> dict[str, str]:
    env: dict[str, str] = {}
    ruta = Path(".env")
    if not ruta.exists():
        return env
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        k, _, v = linea.partition("=")
        env[k.strip()] = v.strip()
    return env


def _intervalo_auto(duracion_seg: float) -> int:
    """Calcula el intervalo de muestreo según la duración del video."""
    minutos = duracion_seg / 60
    if minutos < 5:
        return 1
    elif minutos < 30:
        return 5
    elif minutos < 60:
        return 15
    else:
        return 30


def extraer(
    video_path:   str,
    yolo_weights: str,
    output_dir:   Path,
    intervalo:    int | None,
    max_por_tipo: int,
    device:       str,
) -> None:
    from ultralytics import YOLO

    # ── Abrir video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir: {video_path}")
        sys.exit(1)

    fps_video  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion   = total_fr / fps_video
    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Intervalo automático si no se forzó
    if intervalo is None:
        intervalo = _intervalo_auto(duracion)

    salto_frames = max(1, int(fps_video * intervalo))

    dur_fmt = f"{int(duracion//60):02d}:{int(duracion%60):02d}"
    print(f"\n{'='*60}")
    print(f"  VIDEO    : {Path(video_path).name}")
    print(f"  Resoluc. : {w}×{h} @ {fps_video:.1f}fps")
    print(f"  Duración : {dur_fmt}  ({total_fr} frames)")
    print(f"  Intervalo: cada {intervalo}s ({salto_frames} frames)")
    n_muestras_est = total_fr // salto_frames
    print(f"  Muestras : ~{n_muestras_est} momentos de muestreo")
    print(f"  Modelo   : {yolo_weights}")
    print(f"  Salida   : {output_dir.resolve()}")
    print(f"{'='*60}\n")

    # ── Cargar YOLO ───────────────────────────────────────────────────────────
    print("[INFO] Cargando YOLO...")
    modelo = YOLO(yolo_weights)
    print("[INFO] Listo. Extrayendo recortes...\n")

    # ── Crear carpetas de salida ──────────────────────────────────────────────
    carpetas: dict[str, Path] = {}
    for tipo in TIPOS_VALIDOS:
        p = output_dir / tipo
        p.mkdir(parents=True, exist_ok=True)
        carpetas[tipo] = p

    # ── Contadores ────────────────────────────────────────────────────────────
    conteo:   dict[str, int] = {t: 0 for t in TIPOS_VALIDOS}
    frame_idx = 0
    t_inicio  = time.perf_counter()

    try:
        while True:
            # Saltar al siguiente frame de muestreo
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Verificar límite por tipo
            if all(conteo[t] >= max_por_tipo for t in TIPOS_VALIDOS):
                print(f"\n[INFO] Límite de {max_por_tipo} recortes por tipo alcanzado.")
                break

            # ── Inferencia YOLO (sin tracker — solo detección) ────────────────
            resultados = modelo(
                frame,
                verbose      = False,
                imgsz        = IMGSZ,
                conf         = CONF_YOLO,
                device       = device,
                max_det      = 20,
                agnostic_nms = True,
            )

            n_guardados_frame = 0
            ts = time.strftime("%H%M%S")

            for res in resultados:
                if res.boxes is None:
                    continue
                for i, box in enumerate(res.boxes):
                    clase = res.names[int(box.cls)]
                    tipo  = _YOLO_A_TIPO.get(clase)
                    if tipo is None:
                        continue
                    if conteo[tipo] >= max_por_tipo:
                        continue

                    # Recortar ROI con margen
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, x1 - MARGEN_ROI)
                    y1 = max(0, y1 - MARGEN_ROI)
                    x2 = min(w, x2 + MARGEN_ROI)
                    y2 = min(h, y2 + MARGEN_ROI)
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    # Nombre de archivo con metadatos legibles
                    conf  = float(box.conf)
                    nombre = (
                        f"frame_{frame_idx:08d}"
                        f"_det{i:02d}"
                        f"_{ts}"
                        f"_conf{conf:.2f}"
                        f".jpg"
                    )
                    ruta_salida = carpetas[tipo] / nombre
                    cv2.imwrite(str(ruta_salida), roi, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    conteo[tipo] += 1
                    n_guardados_frame += 1

            # Progreso en consola
            pct     = frame_idx / max(total_fr, 1) * 100
            elapsed = time.perf_counter() - t_inicio
            print(
                f"  {pct:5.1f}%  frame={frame_idx:7d}  "
                f"camion={conteo['camion']:4d}  "
                f"excavadora={conteo['excavadora']:4d}  "
                f"pala={conteo['pala']:4d}  "
                f"({elapsed:.0f}s)",
                end="\r",
            )

            frame_idx += salto_frames

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrumpido por el usuario.")
    finally:
        cap.release()

    # ── Resumen ───────────────────────────────────────────────────────────────
    total = sum(conteo.values())
    elapsed = time.perf_counter() - t_inicio
    print(f"\n\n{'='*60}")
    print(f"  EXTRACCIÓN COMPLETADA  ({elapsed:.0f}s)")
    print(f"{'='*60}")
    for tipo, n in conteo.items():
        barra = "█" * (n // max(total // 40, 1))
        print(f"  {tipo:<15} {n:5d} recortes  {barra}")
    print(f"  {'TOTAL':<15} {total:5d} recortes")
    print(f"\n  Guardados en : {output_dir.resolve()}")
    print(f"\n  SIGUIENTE PASO:")
    print(f"    python organizar_recortes.py")
    print(f"{'='*60}\n")


def main() -> None:
    env            = _leer_env()
    default_video  = env.get("VIDEO_PATH", "")
    default_yolo   = env.get("MODEL_PATH", "models/best_2.pt")

    p = argparse.ArgumentParser(
        description="Paso 1 — Extrae recortes de maquinaria desde un video (sin etiquetar)"
    )
    p.add_argument(
        "--video",
        default=default_video or None,
        required=not bool(default_video),
        help=f"Ruta al video (default desde .env: {default_video or 'obligatorio'})",
    )
    p.add_argument(
        "--yolo",
        default=default_yolo,
        help=f"Pesos YOLO (default: {default_yolo})",
    )
    p.add_argument(
        "--output",
        default="data/muestras_sin_clasificar",
        help="Carpeta de salida (default: data/muestras_sin_clasificar)",
    )
    p.add_argument(
        "--intervalo",
        type=int,
        default=None,
        help="Segundos entre muestras (default: automático según duración del video)",
    )
    p.add_argument(
        "--max",
        type=int,
        default=1000,
        help="Máximo de recortes por tipo de máquina (default: 1000)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo de inferencia (default: cpu)",
    )
    args = p.parse_args()

    if not args.video or not os.path.exists(args.video):
        print(f"[ERROR] Video no encontrado: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.yolo):
        print(f"[ERROR] Modelo no encontrado: {args.yolo}")
        sys.exit(1)

    extraer(
        video_path   = args.video,
        yolo_weights = args.yolo,
        output_dir   = Path(args.output),
        intervalo    = args.intervalo,
        max_por_tipo = args.max,
        device       = args.device,
    )


if __name__ == "__main__":
    main()
