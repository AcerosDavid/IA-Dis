"""
Utilidades de dibujo sobre frames OpenCV (BGR).
Estilo HUD avanzado / tecnología militar de análisis en tiempo real.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

# ── Paleta de colores (BGR) ───────────────────────────────────────────────────

COLORES: dict[str, tuple[int, int, int]] = {
    "camion":     (50, 255, 0),    # verde neón
    "excavadora": (255, 200, 0),   # cyan
    "pala":       (180, 50, 255),  # magenta
}
COLOR_DEFAULT = (255, 255, 255)

ESTADO_COLORES: dict[str, tuple[int, int, int]] = {
    "vacio":               (200, 200, 255),
    "cargado":             (50, 255, 0),
    "escarbando_cargando": (255, 220, 0),
    "descargando":         (50, 50, 255),
    "empty":               (200, 200, 200),
    "loaded":              (50, 255, 0),
    "dumping":             (50, 50, 255),
    "traveling_empty":     (255, 180, 180),
    "traveling_loaded":    (50, 200, 0),
    "waiting_empty":       (150, 150, 150),
    "being_loaded":        (200, 255, 0),
    "digging":             (255, 220, 0),
    "loading_truck":       (255, 255, 50),
    "traveling":           (255, 200, 200),
    "scooping":            (200, 50, 255),
    "dumping_truck":       (0, 150, 255),
}

# ── Símbolos de tipo de máquina ───────────────────────────────────────────────

TIPO_SIMBOLO: dict[str, str] = {
    "camion":     "TRK",
    "excavadora": "EXC",
    "pala":       "PAL",
}


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _draw_text_shadowed(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    shadow_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, shadow_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),          font, scale, color,        thickness,     cv2.LINE_AA)


def _draw_dashed_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 10,
    gap: int = 6,
) -> None:
    """Rectángulo con bordes punteados."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Horizontal top/bottom
    for x in range(x1, x2, dash + gap):
        xe = min(x + dash, x2)
        cv2.line(frame, (x, y1), (xe, y1), color, thickness)
        cv2.line(frame, (x, y2), (xe, y2), color, thickness)
    # Vertical left/right
    for y in range(y1, y2, dash + gap):
        ye = min(y + dash, y2)
        cv2.line(frame, (x1, y), (x1, ye), color, thickness)
        cv2.line(frame, (x2, y), (x2, ye), color, thickness)


def dibujar_tracking(
    frame:         np.ndarray,
    detecciones:   list[dict],
    numeros_flota: dict[int, str],
    estados:       dict[int, str] | None = None,
) -> None:
    """Dibuja detecciones estilo HUD avanzado con paneles de información."""
    for det in detecciones:
        tipo  = det["tipo_maquina"]
        color = COLORES.get(tipo, COLOR_DEFAULT)
        x1, y1, x2, y2 = map(int, det["bbox"])
        tid   = det.get("track_id", "?")
        w_box = x2 - x1
        h_box = y2 - y1
        conf  = det["confianza"]

        estado_maq = det.get("estado") or (estados.get(tid) if estados else None)
        color_bbox = ESTADO_COLORES.get(estado_maq, color) if estado_maq else color

        # ── Relleno semitransparente ──────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bbox, -1)
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

        # ── Borde exterior (doble) ────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bbox, 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                      (color_bbox[0] // 3, color_bbox[1] // 3, color_bbox[2] // 3), 1)

        # ── Esquinas estilo SmartPSS (más grandes) ────────────────────────────
        cs = min(w_box, h_box, 20)
        for (ox, oy, dx, dy) in [
            (x1, y1,  cs, 0), (x1, y1, 0,  cs),
            (x2, y1, -cs, 0), (x2, y1, 0,  cs),
            (x1, y2,  cs, 0), (x1, y2, 0, -cs),
            (x2, y2, -cs, 0), (x2, y2, 0, -cs),
        ]:
            cv2.line(frame, (ox, oy), (ox + dx, oy + dy), color_bbox, 3)

        # Punto en cada esquina
        for (cx2, cy2) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            cv2.circle(frame, (cx2, cy2), 3, color_bbox, -1)

        # ── Panel superior de información ─────────────────────────────────────
        simbolo = TIPO_SIMBOLO.get(tipo, tipo[:3].upper())
        numero  = numeros_flota.get(tid)
        id_txt  = f"#{numero}" if numero else f"ID:{tid}"
        label   = f"{simbolo} {id_txt}  {conf:.2f}"

        font      = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, 0.48, 1)
        pad       = 5
        px1       = x1
        py1       = y1 - th - 2 * pad - 1
        px2       = x1 + tw + 2 * pad + 2

        # Fondo del panel
        overlay2  = frame.copy()
        cv2.rectangle(overlay2, (px1, py1), (px2, y1), (10, 10, 10), -1)
        cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)

        # Borde coloreado del panel
        cv2.rectangle(frame, (px1, py1), (px2, y1), color_bbox, 1)
        # Acento lateral izquierdo (barra de color sólido)
        cv2.rectangle(frame, (px1, py1), (px1 + 3, y1), color_bbox, -1)

        _draw_text_shadowed(frame, label, (px1 + 7, y1 - pad),
                            font, 0.48, color_bbox, 1)

        # ── Barra de confianza debajo del panel ───────────────────────────────
        bar_y     = y1
        bar_total = tw + 2 * pad
        bar_fill  = int(bar_total * conf)
        cv2.rectangle(frame, (px1, bar_y), (px1 + bar_total, bar_y + 2), (30, 30, 30), -1)
        cv2.rectangle(frame, (px1, bar_y), (px1 + bar_fill, bar_y + 2), color_bbox, -1)

        # ── Badge de estado (debajo del bbox) ─────────────────────────────────
        if estado_maq:
            color_est  = ESTADO_COLORES.get(estado_maq, (200, 200, 200))
            label_est  = estado_maq.upper().replace("_", " ")
            (ew, eh), _ = cv2.getTextSize(label_est, font, 0.42, 1)
            ex1, ey1   = x1, y2 + 2
            ex2, ey2   = x1 + ew + 14, y2 + eh + 10

            overlay3 = frame.copy()
            cv2.rectangle(overlay3, (ex1, ey1), (ex2, ey2), (10, 10, 10), -1)
            cv2.addWeighted(overlay3, 0.80, frame, 0.20, 0, frame)
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), color_est, 1)
            cv2.rectangle(frame, (ex1, ey1), (ex1 + 3, ey2), color_est, -1)
            _draw_text_shadowed(frame, label_est, (ex1 + 8, ey2 - 4),
                                font, 0.42, color_est, 1)

        # ── Número de flota grande (centro del bbox) si OCR ──────────────────
        if numero:
            (nw, nh), _ = cv2.getTextSize(f"#{numero}", font, 1.2, 3)
            nx = x1 + (w_box - nw) // 2
            ny = y1 + (h_box + nh) // 2
            # Sombra
            cv2.putText(frame, f"#{numero}", (nx + 2, ny + 2),
                        font, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, f"#{numero}", (nx, ny),
                        font, 1.2, color_bbox, 3, cv2.LINE_AA)

        # ── Línea de ID desde la esquina superior izquierda ───────────────────
        cv2.line(frame, (x1, py1), (px1, py1), color_bbox, 1)


def dibujar_panel_flota(frame: np.ndarray, numeros: dict[int, str]) -> None:
    """Panel semitransparente con lista de flota detectada."""
    if not numeros:
        return

    h, w = frame.shape[:2]
    lineas = ["◈ FLOTA DETECTADA"] + [f"  {TIPO_SIMBOLO.get('camion', 'TRK')} {tid:>3} → #{num}"
                                       for tid, num in numeros.items()]
    font, fs, pad, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.37, 7, 17

    panel_h = pad * 2 + len(lineas) * lh
    panel_w = 185
    px1, py1 = w - panel_w - 10, 10
    px2, py2 = w - 10, py1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (5, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 180, 0), 1)
    cv2.rectangle(frame, (px1, py1), (px1 + 3, py2), (255, 180, 0), -1)

    for i, linea in enumerate(lineas):
        color = (255, 200, 0) if i == 0 else (200, 200, 200)
        cv2.putText(frame, linea, (px1 + pad + 2, py1 + pad + lh * (i + 1) - 2),
                    font, fs, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, linea, (px1 + pad + 2, py1 + pad + lh * (i + 1) - 2),
                    font, fs, color, 1, cv2.LINE_AA)


def dibujar_hud(frame: np.ndarray, texto: str, pausado: bool = False) -> None:
    """HUD superior izquierdo con información del sistema."""
    h, w = frame.shape[:2]

    # Panel de fondo HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(w, 320), 36), (5, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Borde inferior del HUD
    cv2.line(frame, (0, 36), (min(w, 320), 36), (0, 180, 255), 1)

    # Símbolo de grabación (círculo)
    cv2.circle(frame, (12, 18), 5, (0, 50, 255), -1)
    cv2.circle(frame, (12, 18), 5, (0, 100, 255), 1)

    _draw_text_shadowed(frame, texto, (25, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (200, 230, 255), 1)

    if pausado:
        # Banner de pausa centrado
        cx = w // 2
        label = "  ⏸  PAUSADO  "
        (pw, ph), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        bx1 = cx - pw // 2 - 10
        bx2 = cx + pw // 2 + 10
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (bx1, h // 2 - 25), (bx2, h // 2 + 20), (10, 10, 10), -1)
        cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (bx1, h // 2 - 25), (bx2, h // 2 + 20), (0, 200, 255), 2)
        _draw_text_shadowed(frame, "PAUSADO", (bx1 + 15, h // 2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

    # Marca de agua de esquina inferior derecha
    marca = "MINING DISPATCH AI"
    (mw, mh), _ = cv2.getTextSize(marca, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    cv2.putText(frame, marca, (w - mw - 8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 60, 80), 1, cv2.LINE_AA)


def dibujar_scanline(frame: np.ndarray, offset: int = 0, alpha: float = 0.08) -> None:
    """Efecto sutil de scanlines para aspecto de monitor avanzado."""
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)
    for y in range(offset % 2, h, 4):
        overlay[y, :] = (0, 0, 0)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def escalar_frame(frame: np.ndarray, ancho_max: int) -> np.ndarray:
    """Escala el frame manteniendo aspecto si supera el ancho máximo."""
    if ancho_max <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= ancho_max:
        return frame
    escala = ancho_max / w
    return cv2.resize(frame, (ancho_max, int(h * escala)), interpolation=cv2.INTER_AREA)
