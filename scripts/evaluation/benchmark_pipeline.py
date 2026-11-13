"""
benchmark_pipeline.py — Mide el tiempo de CADA parte del pipeline real
para identificar exactamente dónde se pierde tiempo.

Corre desde la raíz:
    python benchmark_pipeline.py

Mide por separado:
  1. Lectura de frame (cap.read)
  2. Resize del frame
  3. Inferencia YOLO (solo modelo)
  4. Gestor de presencia (lógica Python)
  5. Escritura en BD (SQLite INSERT)
  6. Pipeline completo end-to-end
"""
from __future__ import annotations

import os
import sys
import time
import cv2
import numpy as np

VIDEO_PATH  = "data/raw/video.mp4"
MODEL_PATH  = "models/best.pt"
CONF        = 0.5
TEST_FRAMES = 200
IMGSZ       = 320

def medir(nombre: str, tiempos: list[float]) -> None:
    if not tiempos:
        print(f"  {nombre:<35} — sin datos")
        return
    media  = sum(tiempos) / len(tiempos)
    p95    = sorted(tiempos)[int(len(tiempos) * 0.95)]
    fps_eq = 1000 / media if media > 0 else 0
    total  = sum(tiempos) / 1000
    print(f"  {nombre:<35} {media:7.1f}ms  p95={p95:6.1f}ms  ~{fps_eq:5.1f}fps  total={total:.2f}s")

def main():
    import torch
    from ultralytics import YOLO

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    print(f"\n{'='*70}")
    print(f"  Mining Dispatch — Profiling del Pipeline Completo")
    print(f"{'='*70}")
    print(f"  Device: {device} | imgsz: {IMGSZ} | frames: {TEST_FRAMES}")
    print(f"{'='*70}\n")

    # ── 1. Cargar modelo ──────────────────────────────────────────────────────
    print("[1/6] Cargando modelo YOLO…")
    t0 = time.perf_counter()
    model = YOLO(MODEL_PATH)
    print(f"      Cargado en {(time.perf_counter()-t0)*1000:.0f}ms\n")

    # ── 2. Abrir video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[2/6] Video: {w}x{h} @ {fps_video:.1f}fps | {total_frames} frames\n")

    # ── 3. Cargar BD ──────────────────────────────────────────────────────────
    print("[3/6] Conectando BD…")
    from src.database.db import get_session, migrate, RepositorioEventos
    migrate()

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("[4/6] Warmup (10 frames)…")
    for _ in range(10):
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); _, frame = cap.read()
        fr = cv2.resize(frame, (IMGSZ, IMGSZ))
        model.track(fr, persist=True, verbose=False, imgsz=IMGSZ,
                    conf=CONF, device=device, iou=0.7, max_det=20, agnostic_nms=True)

    # Reset tracker
    try:
        model.predictor.trackers[0].reset()
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── 5. Medir cada parte ───────────────────────────────────────────────────
    print(f"[5/6] Midiendo {TEST_FRAMES} frames…\n")

    t_read    : list[float] = []
    t_resize  : list[float] = []
    t_infer   : list[float] = []
    t_parse   : list[float] = []
    t_db      : list[float] = []
    t_total   : list[float] = []

    # Medir escritura BD por separado con dato sintético
    with get_session() as session:
        repo = RepositorioEventos(session)

        # Medir escritura BD
        for _ in range(20):
            t0 = time.perf_counter()
            repo.insertar({
                "tipo_maquina":      "camion",
                "clase_id":          "dump_truck",
                "confianza":         0.9,
                "numero_fotograma":  0,
                "timestamp":         "00:00:00.000",
                "tipo_evento_ciclo": "aparicion",
                "zona_nombre":       None,
            })
            t_db.append((time.perf_counter() - t0) * 1000)

    frame_count = 0
    with get_session() as session:
        repo = RepositorioEventos(session)

        while frame_count < TEST_FRAMES:
            t_frame_ini = time.perf_counter()

            # ── Lectura ──
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret: break
            t_read.append((time.perf_counter() - t0) * 1000)

            # ── Resize ──
            t0 = time.perf_counter()
            frame_r = cv2.resize(frame, (IMGSZ, IMGSZ))
            t_resize.append((time.perf_counter() - t0) * 1000)

            # ── Inferencia YOLO ──
            t0 = time.perf_counter()
            results = model.track(
                frame_r, persist=True, verbose=False,
                imgsz=IMGSZ, conf=CONF, device=device,
                iou=0.7, max_det=20, agnostic_nms=True,
            )
            t_infer.append((time.perf_counter() - t0) * 1000)

            # ── Parseo resultados ──
            t0 = time.perf_counter()
            detecciones = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                ids   = boxes.id
                if ids is not None:
                    scale_x = w / IMGSZ
                    scale_y = h / IMGSZ
                    for i, box in enumerate(boxes):
                        cls_name = results[0].names[int(box.cls)]
                        tid = int(ids[i])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detecciones.append({
                            "track_id":    tid,
                            "tipo_maquina": cls_name,
                            "confianza":   float(box.conf),
                            "bbox":        [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y],
                        })
            t_parse.append((time.perf_counter() - t0) * 1000)

            t_total.append((time.perf_counter() - t_frame_ini) * 1000)
            frame_count += 1

    cap.release()

    # ── 6. Resultados ─────────────────────────────────────────────────────────
    print(f"[6/6] Resultados:\n")
    print(f"  {'Etapa':<35} {'Media':>8}  {'p95':>8}  {'FPS eq':>8}  {'Total':>8}")
    print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    medir("1. Lectura frame (cap.read)",     t_read)
    medir("2. Resize frame",                  t_resize)
    medir("3. Inferencia YOLO + ByteTrack",  t_infer)
    medir("4. Parseo detecciones",            t_parse)
    medir("5. Escritura BD (INSERT)",         t_db)
    print()
    medir("★ TOTAL frame completo",           t_total)

    fps_real = 1000 / (sum(t_total)/len(t_total)) if t_total else 0
    infer_pct = (sum(t_infer)/sum(t_total)*100) if t_total else 0
    read_pct  = (sum(t_read) /sum(t_total)*100) if t_total else 0
    db_pct    = (sum(t_db)   /sum(t_total)*100) if t_db    else 0

    print(f"\n  FPS real del pipeline: {fps_real:.1f}")
    print(f"\n  Desglose del tiempo:")
    print(f"    Inferencia YOLO: {infer_pct:.0f}%")
    print(f"    Lectura video:   {read_pct:.0f}%")
    print(f"    BD (INSERT):     {db_pct:.0f}%")

    # Identificar cuello de botella
    etapas = [
        ("Inferencia YOLO", sum(t_infer)/len(t_infer) if t_infer else 0),
        ("Lectura frame",   sum(t_read) /len(t_read)  if t_read  else 0),
        ("BD INSERT",       sum(t_db)   /len(t_db)    if t_db    else 0),
        ("Parseo",          sum(t_parse)/len(t_parse) if t_parse else 0),
    ]
    cuello = max(etapas, key=lambda x: x[1])

    print(f"\n  🔴 Cuello de botella: {cuello[0]} ({cuello[1]:.1f}ms)")

    # Recomendaciones
    print(f"\n  💡 Recomendaciones:")
    if cuello[0] == "Inferencia YOLO":
        print(f"     → Subir TRACK_SKIP a 3 o 4")
        print(f"     → Bajar IMGSZ a 224 si la precisión lo permite")
        if not cuda:
            print(f"     → Usar GPU (CUDA) — la inferencia domina el tiempo")
    elif cuello[0] == "Lectura frame":
        print(f"     → El disco o el codec es lento")
        print(f"     → Convertir video a H.264 baseline con ffmpeg")
        print(f"     → ffmpeg -i input.mp4 -vcodec libx264 -preset ultrafast output.mp4")
    elif cuello[0] == "BD INSERT":
        print(f"     → Demasiados eventos escritos por frame")
        print(f"     → El geofencing está generando eventos en cada frame")
        print(f"     → Verificar que las zonas están desactivadas")

    print()

if __name__ == "__main__":
    main()
