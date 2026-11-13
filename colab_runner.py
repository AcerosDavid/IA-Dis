"""
Script simplificado para ejecutar el pipeline de detección en Google Colab.

No usa interfaz Tkinter (no funciona en Colab).
Procesa videos y genera resultados estadísticos + video anotado.

Uso en Colab:
    !python colab_runner.py --video /content/drive/MyDrive/mining_dispatch/video.mp4 \
                           --model /content/drive/MyDrive/mining_dispatch/best.pt \
                           --output /content/drive/MyDrive/mining_dispatch/output.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def crear_config_tracker():
    """Crea el archivo de configuración del tracker BoT-SORT."""
    cfg_content = """# BoT-SORT config para minería
tracker_type: botsort
track_high_thresh: 0.45
track_low_thresh: 0.20
new_track_thresh: 0.55
track_buffer: 150
match_thresh: 0.70
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.30
appearance_thresh: 0.65
with_reid: false
model: auto
"""
    cfg_dir = Path('/content/cfg')
    cfg_dir.mkdir(exist_ok=True)
    cfg_path = cfg_dir / 'botsort_mining.yaml'
    cfg_path.write_text(cfg_content)
    return str(cfg_path)


def procesar_video(video_path: str, model_path: str, output_path: str,
                   confidence: float = 0.5, imgsz: int = 640,
                   max_frames: int = None, device: str = 'auto'):
    """
    Procesa un video con YOLO + BoT-SORT tracking.

    Args:
        video_path: Ruta al video de entrada
        model_path: Ruta al modelo YOLO
        output_path: Ruta para guardar video anotado
        confidence: Umbral de confianza
        imgsz: Tamaño de imagen para inferencia
        max_frames: Procesar solo N frames (None = todo)
        device: 'auto', 'cuda', o 'cpu'
    """
    print('=' * 60)
    print('MINING DISPATCH - PIPELINE EN COLAB')
    print('=' * 60)

    # Verificar dispositivo
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Dispositivo: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    # Cargar modelo
    print(f'\nCargando modelo: {model_path}')
    model = YOLO(model_path)
    print(f'Clases: {model.names}')

    # Crear config del tracker
    tracker_cfg = crear_config_tracker()
    print(f'Tracker config: {tracker_cfg}')

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'No se pudo abrir el video: {video_path}')

    # Propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'\nVideo: {width}x{height} @ {fps:.1f}fps | {total_frames} frames')
    print(f'Output: {output_path}')
    print(f'Confianza: {confidence} | ImgSz: {imgsz}')
    if max_frames:
        print(f'Procesando solo {max_frames} frames (modo prueba)')
    print('=' * 60)

    # Configurar writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Estadísticas
    stats = {
        'frames': 0,
        'detecciones': 0,
        'tracks': set(),
        'clases': {},
        'start_time': time.time()
    }

    # Colores por clase
    colors = {
        'dump_truck': (0, 200, 255),    # naranja
        'excavator': (0, 255, 80),     # verde
        'wheel_loader': (255, 100, 0),  # azul
    }

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            # Inferencia con tracking
            results = model.track(
                frame,
                persist=True,
                verbose=False,
                imgsz=imgsz,
                conf=confidence,
                device=device,
                iou=0.7,
                max_det=20,
                agnostic_nms=True,
                tracker=tracker_cfg
            )

            # Procesar resultados
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    ids = boxes.id

                    if ids is not None:
                        for i, box in enumerate(boxes):
                            class_id = int(box.cls)
                            class_name = result.names[class_id]
                            track_id = int(ids[i])
                            conf = float(box.conf)

                            # Actualizar estadísticas
                            stats['detecciones'] += 1
                            stats['tracks'].add(track_id)
                            stats['clases'][class_name] = stats['clases'].get(class_name, 0) + 1

                            # Dibujar bbox
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            color = colors.get(class_name, (200, 200, 200))

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # Etiqueta
                            label = f'ID:{track_id} {class_name} {conf:.2f}'
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Guardar frame
            writer.write(frame)
            frame_count += 1

            # Progreso cada 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - stats['start_time']
                fps = frame_count / elapsed
                print(f'Frame {frame_count}/{total_frames} | FPS: {fps:.1f} | Tracks: {len(stats["tracks"])}')

    finally:
        cap.release()
        writer.release()

    # Estadísticas finales
    elapsed = time.time() - stats['start_time']
    stats['frames'] = frame_count
    stats['tracks'] = len(stats['tracks'])
    stats['fps'] = frame_count / elapsed if elapsed > 0 else 0

    print('=' * 60)
    print('PROCESAMIENTO COMPLETADO')
    print('=' * 60)
    print(f'Frames procesados: {stats["frames"]}')
    print(f'FPS promedio: {stats["fps"]:.2f}')
    print(f'Total detecciones: {stats["detecciones"]}')
    print(f'Tracks únicos: {stats["tracks"]}')
    print(f'Clases detectadas:')
    for clase, count in stats['clases'].items():
        print(f'  {clase}: {count}')
    print(f'Video guardado: {output_path}')
    print('=' * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Ejecutar detección en Colab')
    parser.add_argument('--video', required=True, help='Ruta al video de entrada')
    parser.add_argument('--model', required=True, help='Ruta al modelo YOLO')
    parser.add_argument('--output', required=True, help='Ruta para guardar video anotado')
    parser.add_argument('--confidence', type=float, default=0.5, help='Umbral de confianza')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamaño de imagen')
    parser.add_argument('--max-frames', type=int, default=None, help='Limitar frames para prueba')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Dispositivo')

    args = parser.parse_args()

    # Verificar archivos
    if not Path(args.video).exists():
        raise FileNotFoundError(f'Video no encontrado: {args.video}')
    if not Path(args.model).exists():
        raise FileNotFoundError(f'Modelo no encontrado: {args.model}')

    # Crear directorio de output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Ejecutar
    procesar_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        confidence=args.confidence,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
        device=args.device
    )


if __name__ == '__main__':
    main()
