"""
analizar_video.py — Análisis offline de video y exportación del video anotado.

Procesa el video lo más rápido que permita el equipo (sin mostrar la UI ni
intentar reproducirlo en tiempo real) y escribe un archivo con los bounding
boxes de la maquinaria. El archivo resultante se reproduce fluido porque ya
contiene todos los fotogramas originales anotados.

Uso:
    python -m scripts.evaluation.analizar_video --video data/raw/video.mp4 \
                                            --model models/best.pt \
                                            --output data/output/video_anotado.mp4
"""
from __future__ import annotations

import argparse
import os
import queue
import time
from pathlib import Path

from src.config import Settings
from src.pipeline.runner import OpcionesAnalisis, Procesador


def procesar_video(
    video_path: str,
    model_path: str,
    output_path: str,
    confidence: float = 0.5,
    imgsz: int = 320,
    track_skip: int = 3,
    max_frames: int | None = None,
    device: str = "auto",
    guardar_db: bool = False,
) -> dict:
    """
    Analiza un video y guarda la versión anotada, sin interfaz gráfica.

    Returns:
        Diccionario con estadísticas del análisis.
    """
    print("=" * 60)
    print("MINING DISPATCH - ANÁLISIS OFFLINE DE VIDEO")
    print("=" * 60)

    config = Settings(
        model_path=model_path,
        confidence_threshold=confidence,
        track_skip=track_skip,
        track_imgsz=imgsz,
        forzar_cpu=(device == "cpu"),
    )

    opciones = OpcionesAnalisis(
        guardar_db=guardar_db,
        track_skip=track_skip,
        track_imgsz=imgsz,
        display_width=0,
        exportar_video=True,
        ruta_salida=output_path,
        max_frames=max_frames or 0,
    )

    procesador = Procesador(video_path, opciones, config)
    procesador.start()

    inicio = time.perf_counter()
    ultimo_reporte = 0.0
    stats = {
        "apariciones": 0,
        "desapariciones": 0,
        "zonas": 0,
    }

    try:
        while procesador.is_alive():
            # Drenar eventos para estadísticas sin bloquear.
            while True:
                try:
                    ev = procesador.eventos.get_nowait()
                except queue.Empty:
                    break
                if ev.tipo == "APARICION":
                    stats["apariciones"] += 1
                elif ev.tipo == "DESAPARICION":
                    stats["desapariciones"] += 1
                elif ev.tipo == "ZONA":
                    stats["zonas"] += 1

            ahora = time.perf_counter()
            if ahora - ultimo_reporte >= 1.0:
                ultimo_reporte = ahora
                est = procesador.estado
                total = est.total_frames or "?"
                print(
                    f"Frame {est.frame_actual}/{total} "
                    f"({est.progreso:5.1f}%) | {est.fps:5.1f} fps "
                    f"| activos={est.activos} | eventos={est.eventos}",
                    flush=True,
                )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario, deteniendo...")
        procesador.detener()

    procesador.join()
    procesador.detener()

    est = procesador.estado
    if est.error:
        raise RuntimeError(est.error)

    elapsed = time.perf_counter() - inicio
    salida = est.archivo_salida or output_path

    print("=" * 60)
    print("ANÁLISIS COMPLETADO")
    print("=" * 60)
    print(f"Frames procesados: {est.frames_procesados} de {est.total_frames or '?'}")
    if est.total_frames and est.frames_procesados < est.total_frames:
        print(
            "ADVERTENCIA: el video de salida quedó incompleto. "
            "Revisa que el video de entrada no esté dañado."
        )
    print(f"Tiempo total: {elapsed:.1f}s")
    print(f"Apariciones: {stats['apariciones']}")
    print(f"Desapariciones: {stats['desapariciones']}")
    print(f"Eventos de zona: {stats['zonas']}")
    print(f"Video anotado: {salida}")
    print("=" * 60)

    return {
        "frames": est.frames_procesados,
        "segundos": round(elapsed, 2),
        "archivo_salida": salida,
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analizar video offline y exportar el video anotado",
    )
    parser.add_argument("--video", required=True, help="Ruta al video de entrada")
    parser.add_argument("--model", required=True, help="Ruta al modelo YOLO (.pt)")
    parser.add_argument("--output", required=True, help="Ruta para guardar el video anotado")
    parser.add_argument("--confidence", type=float, default=0.5, help="Umbral de confianza")
    parser.add_argument("--imgsz", type=int, default=320, help="Resolución de inferencia")
    parser.add_argument(
        "--skip", type=int, default=3,
        help="Procesar 1 de cada N frames (más alto = más rápido)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limitar frames (pruebas)")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Dispositivo de inferencia (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--guardar-db", action="store_true",
        help="Guardar eventos en la base de datos (por defecto no)",
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video no encontrado: {args.video}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {args.model}")
    if os.path.normcase(os.path.abspath(args.output)) == os.path.normcase(
        os.path.abspath(args.video)
    ):
        raise ValueError(
            "La salida no puede ser el mismo archivo de entrada; "
            "elige otro nombre para no sobrescribir el video original."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    procesar_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        confidence=args.confidence,
        imgsz=args.imgsz,
        track_skip=args.skip,
        max_frames=args.max_frames,
        device=args.device,
        guardar_db=args.guardar_db,
    )


if __name__ == "__main__":
    main()
