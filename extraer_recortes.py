"""
extraer_recortes.py — Extrae recortes de máquinas desde video para etiquetar manualmente.

Uso:
    python extraer_recortes.py                         # usa VIDEO_PATH del .env
    python extraer_recortes.py --video data/raw/v.mp4  # video específico
    python extraer_recortes.py --cada 10               # 1 recorte cada 10 frames

Genera esta estructura en data/recortes/:
    data/recortes/
    ├── camion/         ← recortes de camiones (mover a empty/ o loaded/)
    ├── excavadora/     ← recortes de excavadoras (mover a digging/, loading/, idle/)
    └── pala/           ← recortes de palas (mover a scooping/, dumping/, idle/)

PASO MANUAL después de correr este script:
    Para cada recorte en data/recortes/camion/:
        - Si la tolva está VACÍA  → mover a data/estados/camion/empty/
        - Si la tolva está LLENA  → mover a data/estados/camion/loaded/

    Para cada recorte en data/recortes/excavadora/:
        - Si está excavando       → mover a data/estados/excavadora/digging/
        - Si está cargando camión → mover a data/estados/excavadora/loading/
        - Si está quieta          → mover a data/estados/excavadora/idle/

    Para cada recorte en data/recortes/pala/:
        - Si está recogiendo      → mover a data/estados/pala/scooping/
        - Si está descargando     → mover a data/estados/pala/dumping/
        - Si está quieta          → mover a data/estados/pala/idle/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

# ── Configuración ─────────────────────────────────────────────────────────────
MARGEN_PX    = 20     # píxeles extra alrededor del bbox para contexto
CONF_MIN     = 0.45   # confianza mínima para guardar un recorte
TAMANO_MIN   = 40     # tamaño mínimo del bbox en píxeles (ancho o alto)
CADA_N_FRAMES = 15    # extraer 1 recorte cada N frames por track_id

TIPOS_MAQUINA = {
    "dump_truck":   "camion",
    "excavator":    "excavadora",
    "wheel_loader": "pala",
    "camion":       "camion",
    "excavadora":   "excavadora",
    "pala":         "pala",
}


def crear_estructura(base: Path) -> dict[str, Path]:
    """Crea las carpetas de salida y retorna un dict tipo→Path."""
    carpetas = {}
    for nombre in ["camion", "excavadora", "pala"]:
        p = base / nombre
        p.mkdir(parents=True, exist_ok=True)
        carpetas[nombre] = p
    return carpetas


def extraer(video_path: str, output_dir: str, cada_n: int) -> None:
    from ultralytics import YOLO
    from src.config import load_config

    config  = load_config()
    modelo  = YOLO(config.model_path)
    base    = Path(output_dir)
    carpetas = crear_estructura(base)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {video_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {w}x{h} @ {fps:.1f}fps | {total} frames")
    print(f"[INFO] Extrayendo 1 recorte cada {cada_n} frames por máquina")
    print(f"[INFO] Guardando en: {base.resolve()}\n")

    # Último frame extraído por track_id
    ultimo_frame: dict[int, int] = {}
    conteo: dict[str, int] = {"camion": 0, "excavadora": 0, "pala": 0}
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inferencia con tracking
            resultados = modelo.track(
                frame,
                persist=True,
                verbose=False,
                imgsz=320,
                conf=CONF_MIN,
                iou=0.7,
                max_det=20,
            )

            for resultado in resultados:
                if resultado.boxes is None:
                    continue
                boxes = resultado.boxes
                ids   = boxes.id
                if ids is None:
                    continue

                for i, box in enumerate(boxes):
                    clase_nombre = resultado.names[int(box.cls)]
                    tipo         = TIPOS_MAQUINA.get(clase_nombre)
                    if tipo is None:
                        continue

                    track_id = int(ids[i])
                    confianza = float(box.conf)

                    # Throttle: no guardar el mismo track_id demasiado seguido
                    ultimo = ultimo_frame.get(track_id, -999)
                    if frame_idx - ultimo < cada_n:
                        continue
                    ultimo_frame[track_id] = frame_idx

                    # Coordenadas del bbox con margen
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, x1 - MARGEN_PX)
                    y1 = max(0, y1 - MARGEN_PX)
                    x2 = min(w, x2 + MARGEN_PX)
                    y2 = min(h, y2 + MARGEN_PX)

                    ancho = x2 - x1
                    alto  = y2 - y1

                    # Descartar recortes demasiado pequeños
                    if ancho < TAMANO_MIN or alto < TAMANO_MIN:
                        continue

                    recorte = frame[y1:y2, x1:x2]
                    if recorte.size == 0:
                        continue

                    # Nombre del archivo: tipo_trackid_frame_conf.jpg
                    nombre = (
                        f"{tipo}_id{track_id:03d}"
                        f"_f{frame_idx:05d}"
                        f"_c{int(confianza*100):02d}.jpg"
                    )
                    ruta = carpetas[tipo] / nombre
                    cv2.imwrite(str(ruta), recorte)
                    conteo[tipo] += 1

            frame_idx += 1

            # Progreso cada 500 frames
            if frame_idx % 500 == 0:
                total_rec = sum(conteo.values())
                print(
                    f"  Frame {frame_idx}/{total} | "
                    f"camion={conteo['camion']} "
                    f"excavadora={conteo['excavadora']} "
                    f"pala={conteo['pala']} "
                    f"| total={total_rec}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por usuario.")
    finally:
        cap.release()

    # Resumen final
    print(f"\n{'='*55}")
    print(f"  EXTRACCIÓN COMPLETADA")
    print(f"{'='*55}")
    for tipo, n in conteo.items():
        print(f"  {tipo:<15} {n:5d} recortes → {carpetas[tipo]}")
    print(f"  TOTAL          {sum(conteo.values()):5d} recortes")
    print(f"{'='*55}")

    print(f"""
SIGUIENTE PASO — Etiquetar manualmente:

  Abre el Explorador de Windows en:
    {base.resolve()}

  Para cada imagen en camion/:
    tolva VACÍA  → mover a data/estados/camion/empty/
    tolva LLENA  → mover a data/estados/camion/loaded/

  Para cada imagen en excavadora/:
    excavando    → mover a data/estados/excavadora/digging/
    cargando     → mover a data/estados/excavadora/loading/
    quieta       → mover a data/estados/excavadora/idle/

  Para cada imagen en pala/:
    recogiendo   → mover a data/estados/pala/scooping/
    descargando  → mover a data/estados/pala/dumping/
    quieta       → mover a data/estados/pala/idle/

  Cuando termines ejecuta:
    python entrenar_clasificador.py
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae recortes de máquinas del video para etiquetar"
    )
    parser.add_argument(
        "--video", default=None,
        help="Ruta al video (default: VIDEO_PATH del .env)"
    )
    parser.add_argument(
        "--output", default="data/recortes",
        help="Carpeta de salida (default: data/recortes)"
    )
    parser.add_argument(
        "--cada", type=int, default=CADA_N_FRAMES,
        help=f"Extraer 1 recorte cada N frames por track (default: {CADA_N_FRAMES})"
    )
    args = parser.parse_args()

    if args.video is None:
        from src.config import load_config
        args.video = load_config().video_path

    if not os.path.exists(args.video):
        print(f"[ERROR] Video no encontrado: {args.video}")
        raise SystemExit(1)

    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] Extrayendo cada {args.cada} frames...")
    extraer(args.video, args.output, args.cada)


if __name__ == "__main__":
    main()
