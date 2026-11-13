"""
benchmark.py — Mide el rendimiento real del pipeline YOLO en diferentes
configuraciones sin API ni frontend.

Corre desde la raíz del proyecto:
    python benchmark.py

Prueba las siguientes combinaciones:
  1. PyTorch CPU  — imgsz=320, skip=1
  2. PyTorch CPU  — imgsz=416, skip=2
  3. PyTorch CUDA — imgsz=416, skip=1
  4. PyTorch CUDA — imgsz=640, skip=1
  5. ONNX CPU     — imgsz=320, skip=1
  6. ONNX CPU     — imgsz=416, skip=1

Para cada config mide:
  - FPS de inferencia (solo YOLO)
  - FPS total del pipeline (lectura + inferencia)
  - Latencia media por frame (ms)
  - Detecciones promedio por frame
"""
from __future__ import annotations

import os
import time
import cv2
import numpy as np

# ── Configuración ─────────────────────────────────────────────────────────────
VIDEO_PATH   = "data/raw/video.mp4"
MODEL_PATH   = "models/best.pt"
CONF         = 0.5
WARMUP_FRAMES = 10    # frames iniciales descartados (calentamiento)
TEST_FRAMES   = 100   # frames a medir por configuración

# ── Helpers ───────────────────────────────────────────────────────────────────

def barra(fps: float, max_fps: float = 60) -> str:
    pct   = min(fps / max_fps, 1.0)
    lleno = int(pct * 30)
    return f"[{'█' * lleno}{'░' * (30 - lleno)}] {fps:5.1f} fps"


def benchmark_pytorch(
    device: str,
    imgsz: int,
    skip: int,
    label: str,
) -> dict:
    """Mide rendimiento con backend PyTorch."""
    import torch
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  ✗ No se pudo abrir {VIDEO_PATH}")
        return {}

    # Warmup
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = cap.read()
        if frame is not None:
            fr = cv2.resize(frame, (imgsz, imgsz))
            model.track(fr, persist=True, verbose=False,
                        imgsz=imgsz, conf=CONF, device=device,
                        iou=0.7, max_det=20, agnostic_nms=True)

    # Reset tracker
    if hasattr(model, 'predictor') and model.predictor:
        try: model.predictor.trackers[0].reset()
        except: pass

    # Benchmark
    latencias   : list[float] = []
    detecciones : list[int]   = []
    t_total_ini  = time.perf_counter()

    frame_count = 0
    infer_count = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while frame_count < TEST_FRAMES:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret: break

        if frame_count % skip == 0:
            fr = cv2.resize(frame, (imgsz, imgsz))
            t0 = time.perf_counter()
            res = model.track(
                fr, persist=True, verbose=False,
                imgsz=imgsz, conf=CONF, device=device,
                iou=0.7, max_det=20, agnostic_nms=True,
            )
            lat = (time.perf_counter() - t0) * 1000
            latencias.append(lat)

            n_dets = len(res[0].boxes) if res and res[0].boxes else 0
            detecciones.append(n_dets)
            infer_count += 1

        frame_count += 1

    t_total = time.perf_counter() - t_total_ini
    cap.release()

    fps_total  = frame_count / t_total
    fps_infer  = infer_count / (sum(latencias) / 1000) if latencias else 0
    lat_media  = sum(latencias) / len(latencias) if latencias else 0
    lat_p95    = sorted(latencias)[int(len(latencias) * 0.95)] if latencias else 0
    dets_media = sum(detecciones) / len(detecciones) if detecciones else 0

    return {
        "label":      label,
        "fps_total":  fps_total,
        "fps_infer":  fps_infer,
        "lat_media":  lat_media,
        "lat_p95":    lat_p95,
        "dets_media": dets_media,
        "device":     device,
        "imgsz":      imgsz,
        "skip":       skip,
    }


def benchmark_onnx(
    imgsz: int,
    skip: int,
    label: str,
) -> dict:
    """Mide rendimiento con backend ONNX Runtime."""
    import onnxruntime as rt

    onnx_path = os.path.splitext(MODEL_PATH)[0] + ".onnx"
    if not os.path.exists(onnx_path):
        print(f"  ⚠  ONNX no encontrado en {onnx_path}")
        print(f"      Exportando (puede tardar 1-2 min)…")
        from ultralytics import YOLO
        m = YOLO(MODEL_PATH)
        m.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
        if not os.path.exists(onnx_path):
            print("  ✗  Exportación falló")
            return {}

    opts = rt.SessionOptions()
    opts.intra_op_num_threads  = os.cpu_count() or 4
    opts.inter_op_num_threads  = os.cpu_count() or 4
    opts.execution_mode        = rt.ExecutionMode.ORT_PARALLEL
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    session    = rt.InferenceSession(onnx_path, sess_options=opts,
                                     providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return {}

    def _infer(frame: np.ndarray) -> int:
        img = cv2.resize(frame, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        out = session.run(None, {input_name: img})
        preds = out[0][0]
        return int((preds[:, 4] >= CONF).sum())

    # Warmup
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = cap.read()
        if frame is not None:
            _infer(frame)

    # Benchmark
    latencias:   list[float] = []
    detecciones: list[int]   = []
    t_total_ini  = time.perf_counter()
    frame_count  = 0
    infer_count  = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while frame_count < TEST_FRAMES:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret: break

        if frame_count % skip == 0:
            t0  = time.perf_counter()
            n   = _infer(frame)
            lat = (time.perf_counter() - t0) * 1000
            latencias.append(lat)
            detecciones.append(n)
            infer_count += 1

        frame_count += 1

    t_total = time.perf_counter() - t_total_ini
    cap.release()

    fps_total  = frame_count / t_total
    fps_infer  = infer_count / (sum(latencias) / 1000) if latencias else 0
    lat_media  = sum(latencias) / len(latencias) if latencias else 0
    lat_p95    = sorted(latencias)[int(len(latencias) * 0.95)] if latencias else 0
    dets_media = sum(detecciones) / len(detecciones) if detecciones else 0

    return {
        "label":      label,
        "fps_total":  fps_total,
        "fps_infer":  fps_infer,
        "lat_media":  lat_media,
        "lat_p95":    lat_p95,
        "dets_media": dets_media,
        "device":     "onnx-cpu",
        "imgsz":      imgsz,
        "skip":       skip,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch

    cuda = torch.cuda.is_available()
    print(f"\n{'='*60}")
    print(f"  Mining Dispatch — Benchmark de Inferencia YOLO")
    print(f"{'='*60}")
    print(f"  Video:  {VIDEO_PATH}")
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  GPU:    {'✓ ' + torch.cuda.get_device_name(0) if cuda else '✗ No disponible'}")
    print(f"  CPU:    {os.cpu_count()} núcleos")
    print(f"  Frames: {TEST_FRAMES} por config (+ {WARMUP_FRAMES} warmup)")
    print(f"{'='*60}\n")

    configs = []

    # CPU configs — basadas en resultados anteriores
    configs.append(("pytorch-cpu", lambda: benchmark_pytorch(
        "cpu", 320, 1, "PyTorch CPU  | imgsz=320 | skip=1")))
    configs.append(("pytorch-cpu", lambda: benchmark_pytorch(
        "cpu", 320, 2, "PyTorch CPU  | imgsz=320 | skip=2")))
    configs.append(("pytorch-cpu", lambda: benchmark_pytorch(
        "cpu", 320, 3, "PyTorch CPU  | imgsz=320 | skip=3")))

    # ONNX configs — ahora con dynamic=True acepta cualquier imgsz
    configs.append(("onnx", lambda: benchmark_onnx(
        320, 1, "ONNX CPU     | imgsz=320 | skip=1")))
    configs.append(("onnx", lambda: benchmark_onnx(
        320, 2, "ONNX CPU     | imgsz=320 | skip=2")))

    # CUDA configs (solo si disponible)
    if cuda:
        configs.append(("cuda", lambda: benchmark_pytorch(
            "cuda", 320, 1, "PyTorch CUDA | imgsz=320 | skip=1")))
        configs.append(("cuda", lambda: benchmark_pytorch(
            "cuda", 416, 1, "PyTorch CUDA | imgsz=416 | skip=1")))
        configs.append(("cuda", lambda: benchmark_pytorch(
            "cuda", 320, 2, "PyTorch CUDA | imgsz=320 | skip=2")))

    resultados = []
    for i, (_, fn) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Ejecutando… ", end="", flush=True)
        try:
            r = fn()
            if r:
                resultados.append(r)
                print(f"✓  {r['fps_total']:5.1f} fps total")
            else:
                print("✗  falló")
        except Exception as e:
            print(f"✗  error: {e}")

    # ── Tabla de resultados ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    print(f"  {'Config':<38} {'FPS tot':>7} {'FPS inf':>7} {'Lat ms':>7} {'p95ms':>7} {'Dets':>5}")
    print(f"  {'-'*38} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")

    resultados.sort(key=lambda x: x["fps_total"], reverse=True)
    for r in resultados:
        print(
            f"  {r['label']:<38} "
            f"{r['fps_total']:>7.1f} "
            f"{r['fps_infer']:>7.1f} "
            f"{r['lat_media']:>7.1f} "
            f"{r['lat_p95']:>7.1f} "
            f"{r['dets_media']:>5.1f}"
        )

    print(f"{'='*60}")

    if resultados:
        best = resultados[0]
        print(f"\n  🏆 Mejor config: {best['label']}")
        print(f"     FPS total:    {best['fps_total']:.1f}")
        print(f"     Latencia:     {best['lat_media']:.1f} ms media | {best['lat_p95']:.1f} ms p95")
        print(f"\n  💡 Recomendación para video.py:")
        print(f"     TRACK_IMGSZ = {best['imgsz']}")
        print(f"     TRACK_SKIP  = {best['skip']}")
        device_hint = best['device']
        if device_hint == "cuda":
            print(f"     Backend:     PyTorch CUDA (ya configurado)")
        elif device_hint == "onnx-cpu":
            print(f"     Backend:     ONNX Runtime CPU (ya configurado con load_model())")
        else:
            print(f"     Backend:     PyTorch CPU")

    print()


if __name__ == "__main__":
    main()
