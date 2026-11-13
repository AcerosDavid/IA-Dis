"""
Interfaz de escritorio (Tkinter) del sistema de despacho minero.
Diseño avanzado estilo HUD / tecnología de análisis en tiempo real.
"""
from __future__ import annotations

import math
import os
import queue
import sys
import time
from pathlib import Path

def _configurar_tcl_tk() -> None:
    base = Path(sys.base_prefix)
    if not os.environ.get("TCL_LIBRARY"):
        for candidato in (base / "tcl" / "tcl8.6", base / "lib" / "tcl8.6"):
            if (candidato / "init.tcl").exists():
                os.environ["TCL_LIBRARY"] = str(candidato)
                break
    if not os.environ.get("TK_LIBRARY"):
        for candidato in (base / "tcl" / "tk8.6", base / "lib" / "tk8.6"):
            if (candidato / "tk.tcl").exists():
                os.environ["TK_LIBRARY"] = str(candidato)
                break

_configurar_tcl_tk()

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np

from src.config import load_config
from src.database.db import RepositorioEventos, get_session, migrate
from src.pipeline.runner import EventoUI, OpcionesAnalisis, Procesador

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


# ── Paleta de colores ─────────────────────────────────────────────────────────

C = {
    # Fondos
    "bg_deep":    "#050810",
    "bg_main":    "#080d1a",
    "bg_panel":   "#0c1222",
    "bg_card":    "#101828",
    "bg_hover":   "#1a2540",
    # Neones principales
    "cyan":       "#00d4ff",
    "cyan_dim":   "#0088aa",
    "cyan_dark":  "#003344",
    "green":      "#00ff88",
    "green_dim":  "#00aa55",
    "green_dark": "#003322",
    "magenta":    "#ff00aa",
    "magenta_dim":"#aa0066",
    "yellow":     "#ffcc00",
    "yellow_dim": "#aa8800",
    "orange":     "#ff6600",
    "red":        "#ff2244",
    "red_dim":    "#882233",
    # Texto
    "text_hi":    "#e8f4ff",
    "text_mid":   "#7a9cbf",
    "text_lo":    "#3a5070",
    "text_green": "#00ff88",
    "text_cyan":  "#00d4ff",
    # Bordes
    "border":     "#1e3050",
    "border_hi":  "#00d4ff",
    "border_dim": "#0d1e35",
}

# Colores de tipo de evento
TIPO_COLOR = {
    "APARICION":   C["green"],
    "DESAPARICION": C["red"],
    "ZONA":        C["yellow"],
}


# ── Helpers de widgets ────────────────────────────────────────────────────────

def _frame(parent, **kw) -> tk.Frame:
    return tk.Frame(parent, bg=kw.pop("bg", C["bg_panel"]), **kw)


def _label(parent, text="", textvariable=None, font=None, fg=None, bg=None, **kw) -> tk.Label:
    return tk.Label(
        parent,
        text=text,
        textvariable=textvariable,
        font=font or ("Consolas", 10),
        fg=fg or C["text_mid"],
        bg=bg or C["bg_panel"],
        **kw,
    )


def _separator(parent, orient="horizontal", color=None) -> tk.Canvas:
    color = color or C["border"]
    if orient == "horizontal":
        c = tk.Canvas(parent, height=1, bg=C["bg_panel"], highlightthickness=0)
        c.bind("<Configure>", lambda e: (c.delete("all"), c.create_line(0, 0, e.width, 0, fill=color)))
    else:
        c = tk.Canvas(parent, width=1, bg=C["bg_panel"], highlightthickness=0)
        c.bind("<Configure>", lambda e: (c.delete("all"), c.create_line(0, 0, 0, e.height, fill=color)))
    return c


class _NeonButton(tk.Frame):
    """Botón con borde neón y efecto hover."""

    def __init__(self, parent, text, command, color=None, disabled=False, **kw):
        self._color  = color or C["cyan"]
        self._dim    = C["cyan_dim"]
        self._active = not disabled
        super().__init__(parent, bg=C["bg_panel"], bd=0, highlightthickness=0)

        self._inner = tk.Frame(self, bg=C["bg_card"], bd=0, highlightthickness=1,
                               highlightbackground=self._dim if not disabled else C["border"])
        self._inner.pack(fill="both", expand=True)

        self._lbl = tk.Label(
            self._inner, text=text,
            font=("Consolas", 9, "bold"),
            fg=(self._dim if not disabled else C["text_lo"]),
            bg=C["bg_card"],
            padx=14, pady=5, cursor="hand2" if not disabled else "arrow",
        )
        self._lbl.pack(fill="both", expand=True)

        self._cmd = command
        if not disabled:
            for w in (self._inner, self._lbl):
                w.bind("<Enter>",    self._on_enter)
                w.bind("<Leave>",    self._on_leave)
                w.bind("<Button-1>", self._on_click)
                w.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, _):
        if not self._active:
            return
        self._inner.configure(highlightbackground=self._color)
        self._lbl.configure(fg=self._color, bg=C["bg_hover"])
        self._inner.configure(bg=C["bg_hover"])

    def _on_leave(self, _):
        if not self._active:
            return
        self._inner.configure(highlightbackground=self._dim, bg=C["bg_card"])
        self._lbl.configure(fg=self._dim, bg=C["bg_card"])

    def _on_click(self, _):
        if self._active:
            self._lbl.configure(bg=self._color, fg=C["bg_deep"])
            self._inner.configure(bg=self._color)

    def _on_release(self, _):
        if self._active:
            self._on_enter(None)
            self._cmd()

    def set_state(self, state: str):
        enabled = state != "disabled"
        self._active = enabled
        for w in (self._inner, self._lbl):
            for ev in ("<Enter>", "<Leave>", "<Button-1>", "<ButtonRelease-1>"):
                w.unbind(ev)
        if enabled:
            for w in (self._inner, self._lbl):
                w.bind("<Enter>",    self._on_enter)
                w.bind("<Leave>",    self._on_leave)
                w.bind("<Button-1>", self._on_click)
                w.bind("<ButtonRelease-1>", self._on_release)
            self._inner.configure(highlightbackground=self._dim, bg=C["bg_card"])
            self._lbl.configure(fg=self._dim, bg=C["bg_card"], cursor="hand2")
        else:
            self._inner.configure(highlightbackground=C["border"], bg=C["bg_card"])
            self._lbl.configure(fg=C["text_lo"], bg=C["bg_card"], cursor="arrow")

    def set_text(self, text: str):
        self._lbl.configure(text=text)


class _StatCard(tk.Frame):
    """Tarjeta de estadística con valor grande y etiqueta pequeña."""

    def __init__(self, parent, label, color=None, width=100):
        self._color = color or C["cyan"]
        super().__init__(parent, bg=C["bg_card"], bd=0,
                         highlightthickness=1, highlightbackground=C["border"],
                         width=width)
        self.pack_propagate(False)

        tk.Label(self, text=label.upper(), font=("Consolas", 7),
                 fg=self._color, bg=C["bg_card"]).pack(pady=(6, 0))

        self._val_var = tk.StringVar(value="0")
        tk.Label(self, textvariable=self._val_var,
                 font=("Consolas", 20, "bold"),
                 fg=self._color, bg=C["bg_card"]).pack()

        self._sub_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._sub_var,
                 font=("Consolas", 7),
                 fg=C["text_lo"], bg=C["bg_card"]).pack(pady=(0, 6))

    def update(self, value: str, sub: str = ""):
        self._val_var.set(value)
        self._sub_var.set(sub)

    def flash(self):
        """Parpadeo rápido para indicar nuevo evento."""
        self.configure(highlightbackground=self._color)
        self.after(200, lambda: self.configure(highlightbackground=C["border"]))


class _FpsGraph(tk.Canvas):
    """Mini gráfica de barras con historial de FPS."""

    MAX_SAMPLES = 60

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["bg_card"], highlightthickness=0,
                         height=48, **kw)
        self._samples: list[float] = []
        self._max_fps = 30.0
        self.bind("<Configure>", lambda _: self._redraw())

    def push(self, fps: float):
        self._samples.append(fps)
        if len(self._samples) > self.MAX_SAMPLES:
            self._samples.pop(0)
        self._max_fps = max(30.0, max(self._samples) * 1.1)
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or not self._samples:
            return

        # Grid horizontal
        for y_pct in (0.25, 0.5, 0.75):
            y = int(h * (1 - y_pct))
            self.create_line(0, y, w, y, fill=C["border_dim"], dash=(2, 4))

        # Barras
        n = len(self._samples)
        bar_w = max(1, w // self.MAX_SAMPLES)
        for i, fps in enumerate(self._samples):
            x = i * bar_w
            bar_h = int((fps / self._max_fps) * h)
            # Gradiente de color según FPS
            ratio = fps / self._max_fps
            if ratio > 0.6:
                color = C["green"]
            elif ratio > 0.3:
                color = C["yellow"]
            else:
                color = C["red"]
            self.create_rectangle(x, h - bar_h, x + bar_w - 1, h,
                                  fill=color, outline="")

        # Valor actual
        cur = f"{self._samples[-1]:.1f} FPS"
        self.create_text(w - 4, 4, text=cur, anchor="ne",
                         font=("Consolas", 8), fill=C["text_cyan"])


class _LogTerminal(tk.Frame):
    """Terminal de log estilo hacker con scroll."""

    MAX_LINES = 200

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["bg_deep"], bd=0,
                         highlightthickness=1, highlightbackground=C["border"],
                         **kw)
        self._text = tk.Text(
            self,
            bg=C["bg_deep"], fg=C["green"],
            font=("Consolas", 8),
            state="disabled",
            wrap="word",
            bd=0, highlightthickness=0,
            insertbackground=C["green"],
            selectbackground=C["cyan_dark"],
        )
        sb = tk.Scrollbar(self, bg=C["bg_panel"], troughcolor=C["bg_deep"],
                          activebackground=C["cyan_dim"],
                          command=self._text.yview)
        self._text.configure(yscrollcommand=sb.set)

        # Tags de color
        self._text.tag_configure("ts",         foreground=C["text_lo"])
        self._text.tag_configure("aparicion",  foreground=C["green"])
        self._text.tag_configure("desaparicion", foreground=C["red"])
        self._text.tag_configure("zona",       foreground=C["yellow"])
        self._text.tag_configure("info",       foreground=C["cyan"])
        self._text.tag_configure("error",      foreground=C["magenta"])

        sb.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)

    def log(self, msg: str, tag: str = "info"):
        ts = time.strftime("%H:%M:%S")
        self._text.configure(state="normal")
        # Límite de líneas
        lines = int(self._text.index("end-1c").split(".")[0])
        if lines > self.MAX_LINES:
            self._text.delete("1.0", "10.0")
        self._text.insert("end", f"[{ts}] ", "ts")
        self._text.insert("end", msg + "\n", tag)
        self._text.configure(state="disabled")
        self._text.yview_moveto(1.0)


class _EventTable(tk.Frame):
    """Tabla de eventos con colores por tipo."""

    COLS = [
        ("hora",     "HORA",    80),
        ("tipo",     "TIPO",    90),
        ("maquina",  "MÁQ",     65),
        ("id",       "ID",      35),
        ("conf",     "CONF",    48),
        ("zona",     "ZONA",    90),
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["bg_panel"], **kw)

        # Cabecera
        hdr = tk.Frame(self, bg=C["bg_card"])
        hdr.pack(fill="x")
        for _, titulo, ancho in self.COLS:
            tk.Label(hdr, text=titulo, font=("Consolas", 7, "bold"),
                     fg=C["cyan"], bg=C["bg_card"],
                     width=max(1, ancho // 7), anchor="w").pack(side="left", padx=2, pady=3)

        _separator(self, color=C["border_hi"]).pack(fill="x")

        # Canvas + scrollbar para las filas
        container = tk.Frame(self, bg=C["bg_panel"])
        container.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(container, bg=C["bg_panel"],
                                 highlightthickness=0)
        sb = tk.Scrollbar(container, orient="vertical",
                          command=self._canvas.yview,
                          bg=C["bg_panel"], troughcolor=C["bg_deep"],
                          activebackground=C["cyan_dim"])
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._canvas, bg=C["bg_panel"])
        self._win_id = self._canvas.create_window((0, 0), window=self._inner,
                                                  anchor="nw")
        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        self._rows: list[tk.Frame] = []

    def _on_inner_configure(self, _):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self._canvas.itemconfigure(self._win_id, width=e.width)

    def insertar(self, ev: EventoUI):
        color = TIPO_COLOR.get(ev.tipo, C["text_mid"])
        conf_str = f"{ev.confianza:.2f}" if ev.confianza is not None else "—"
        vals = [
            ev.timestamp, ev.tipo, ev.maquina,
            str(ev.track_id), conf_str, ev.zona or "—",
        ]
        # Fila alternada
        odd = len(self._rows) % 2 == 0
        row_bg = C["bg_card"] if odd else C["bg_deep"]

        row = tk.Frame(self._inner, bg=row_bg)
        row.pack(fill="x")
        for val, (_, _, ancho) in zip(vals, self.COLS):
            tk.Label(row, text=val, font=("Consolas", 8),
                     fg=color, bg=row_bg,
                     width=max(1, ancho // 7), anchor="w").pack(side="left", padx=2, pady=1)

        self._rows.append(row)
        # Límite de 300 filas
        if len(self._rows) > 300:
            oldest = self._rows.pop(0)
            oldest.destroy()

        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

    def limpiar(self):
        for r in self._rows:
            r.destroy()
        self._rows.clear()


# ── Ventana principal ─────────────────────────────────────────────────────────

class AplicacionDesk(tk.Tk):
    """Ventana principal con diseño HUD avanzado."""

    POLL_MS = 40

    def __init__(self) -> None:
        super().__init__()
        self.config_app = load_config()

        self.title("MINING DISPATCH  //  AI VISION SYSTEM v2.0")
        self.geometry(self.config_app.window_geometry)
        self.minsize(1100, 700)
        self.configure(bg=C["bg_deep"])
        self._aplicar_icono()

        self._procesador: Procesador | None = None
        self._fuente: str | int | None = self.config_app.video_path or None
        self._photo: "ImageTk.PhotoImage | None" = None
        self._finalizado_avisado = False
        self._poll_programado = False
        self._t_inicio: float = 0.0

        # Animación del header
        self._anim_tick = 0
        self._scan_offset = 0

        self._crear_variables()
        self._construir_ui()
        self._animar_header()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

    # ── Icono ─────────────────────────────────────────────────────────────────

    def _aplicar_icono(self) -> None:
        icono = RAIZ_PROYECTO / "logo.ico"
        if sys.platform == "win32" and icono.exists():
            try:
                self.iconbitmap(str(icono))
            except tk.TclError:
                pass

    # ── Variables ─────────────────────────────────────────────────────────────

    def _crear_variables(self) -> None:
        self.var_fuente     = tk.StringVar(value=self._texto_fuente())
        self.var_estado     = tk.StringVar(value="SISTEMA LISTO")
        self.var_guardar_db = tk.BooleanVar(value=True)
        self.var_ocr        = tk.BooleanVar(value=self.config_app.ocr_habilitado)
        self.var_clasif     = tk.BooleanVar(value=self.config_app.clasificador_habilitado)
        self.var_skip       = tk.StringVar(value=str(self.config_app.track_skip))
        self.var_imgsz      = tk.StringVar(value=str(self.config_app.track_imgsz))

    # ── Construcción de la UI ─────────────────────────────────────────────────

    def _construir_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._construir_header()
        self._construir_cuerpo()
        self._construir_statusbar()

    # ── HEADER ────────────────────────────────────────────────────────────────

    def _construir_header(self) -> None:
        hdr = tk.Frame(self, bg=C["bg_main"], bd=0,
                       highlightthickness=1, highlightbackground=C["border"])
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        # Logo / título
        logo_f = tk.Frame(hdr, bg=C["bg_main"])
        logo_f.grid(row=0, column=0, padx=14, pady=8, sticky="w")

        tk.Label(logo_f, text="▶ MINING DISPATCH", font=("Consolas", 14, "bold"),
                 fg=C["cyan"], bg=C["bg_main"]).pack(side="left")
        tk.Label(logo_f, text="  AI VISION SYSTEM", font=("Consolas", 9),
                 fg=C["text_lo"], bg=C["bg_main"]).pack(side="left", padx=(4, 0))

        # Indicadores de estado (LEDs animados)
        self._leds_frame = tk.Frame(hdr, bg=C["bg_main"])
        self._leds_frame.grid(row=0, column=1, padx=10, pady=8, sticky="w")

        self._led_proc  = self._crear_led(self._leds_frame, "PROCESANDO")
        self._led_track = self._crear_led(self._leds_frame, "TRACKING")
        self._led_db    = self._crear_led(self._leds_frame, "BASE DATOS")

        # Fuente seleccionada
        src_f = tk.Frame(hdr, bg=C["bg_main"])
        src_f.grid(row=0, column=2, padx=14, pady=8, sticky="e")

        tk.Label(src_f, text="SRC:", font=("Consolas", 8),
                 fg=C["text_lo"], bg=C["bg_main"]).pack(side="left")
        tk.Label(src_f, textvariable=self.var_fuente,
                 font=("Consolas", 8), fg=C["cyan_dim"],
                 bg=C["bg_main"]).pack(side="left", padx=4)

        # Botones de control
        ctrl_f = tk.Frame(hdr, bg=C["bg_main"])
        ctrl_f.grid(row=0, column=3, padx=8, pady=6, sticky="e")

        _NeonButton(ctrl_f, "[ VIDEO ]", self._abrir_video,
                    color=C["cyan"]).pack(side="left", padx=3)
        _NeonButton(ctrl_f, "[ CÁMARA ]", self._usar_camara,
                    color=C["cyan"]).pack(side="left", padx=3)

        _separator_v = tk.Frame(ctrl_f, bg=C["border"], width=1)
        _separator_v.pack(side="left", fill="y", padx=6)

        self.btn_iniciar = _NeonButton(ctrl_f, "[ INICIAR ]", self._iniciar,
                                       color=C["green"])
        self.btn_iniciar.pack(side="left", padx=3)

        self.btn_pausar = _NeonButton(ctrl_f, "[ PAUSAR ]", self._pausar,
                                      color=C["yellow"])
        self.btn_pausar.pack(side="left", padx=3)
        self.btn_pausar.set_state("disabled")

        self.btn_detener = _NeonButton(ctrl_f, "[ DETENER ]", self._detener,
                                       color=C["red"])
        self.btn_detener.pack(side="left", padx=3)

        # Reloj en tiempo real
        self._reloj_var = tk.StringVar(value="00:00:00")
        tk.Label(hdr, textvariable=self._reloj_var,
                 font=("Consolas", 11, "bold"),
                 fg=C["text_lo"], bg=C["bg_main"],
                 padx=12).grid(row=0, column=4, sticky="e")
        self._tick_reloj()

    def _crear_led(self, parent, label: str) -> tk.Label:
        f = tk.Frame(parent, bg=C["bg_main"])
        f.pack(side="left", padx=8)
        led = tk.Label(f, text="●", font=("Consolas", 10),
                       fg=C["border"], bg=C["bg_main"])
        led.pack(side="left")
        tk.Label(f, text=label, font=("Consolas", 7),
                 fg=C["text_lo"], bg=C["bg_main"]).pack(side="left", padx=(2, 0))
        return led

    def _tick_reloj(self) -> None:
        self._reloj_var.set(time.strftime("%H:%M:%S"))
        self.after(1000, self._tick_reloj)

    # ── CUERPO ────────────────────────────────────────────────────────────────

    def _construir_cuerpo(self) -> None:
        cuerpo = tk.Frame(self, bg=C["bg_deep"])
        cuerpo.grid(row=1, column=0, sticky="nsew")
        cuerpo.columnconfigure(0, weight=3)
        cuerpo.columnconfigure(1, weight=0)
        cuerpo.rowconfigure(0, weight=1)

        self._construir_panel_video(cuerpo)
        self._construir_panel_derecho(cuerpo)

    # ── Panel de video ────────────────────────────────────────────────────────

    def _construir_panel_video(self, parent) -> None:
        video_outer = tk.Frame(parent, bg=C["bg_deep"],
                               highlightthickness=1, highlightbackground=C["border"])
        video_outer.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        video_outer.columnconfigure(0, weight=1)
        video_outer.rowconfigure(1, weight=1)

        # Mini barra superior del panel de video
        vid_hdr = tk.Frame(video_outer, bg=C["bg_card"])
        vid_hdr.grid(row=0, column=0, sticky="ew")

        tk.Label(vid_hdr, text="◈ FEED EN VIVO  //  ANÁLISIS YOLO",
                 font=("Consolas", 8, "bold"),
                 fg=C["cyan"], bg=C["bg_card"],
                 padx=8, pady=4).pack(side="left")

        self._scan_label = tk.Label(vid_hdr, text="",
                                    font=("Consolas", 8),
                                    fg=C["green_dim"], bg=C["bg_card"],
                                    padx=8)
        self._scan_label.pack(side="right")

        # Área de video
        self.video_canvas = tk.Canvas(
            video_outer,
            bg=C["bg_deep"],
            highlightthickness=0,
            cursor="crosshair",
        )
        self.video_canvas.grid(row=1, column=0, sticky="nsew")
        self._dibujar_placeholder()

        # Overlay de info en el frame (esquina superior izquierda del canvas)
        self._frame_var  = tk.StringVar(value="FRAME  —")
        self._prog_var   = tk.StringVar(value="PROGRESO  —")

    def _dibujar_placeholder(self) -> None:
        """Dibuja la cuadrícula del placeholder cuando no hay video."""
        self.video_canvas.delete("all")
        self.video_canvas.after(50, self._dibujar_grid_placeholder)

    def _dibujar_grid_placeholder(self) -> None:
        c = self.video_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            c.after(100, self._dibujar_grid_placeholder)
            return

        # Grid de puntos
        step = 40
        for x in range(0, w, step):
            for y in range(0, h, step):
                c.create_oval(x-1, y-1, x+1, y+1, fill=C["border_dim"], outline="")

        # Cruz central
        cx, cy = w // 2, h // 2
        size = 60
        c.create_line(cx - size, cy, cx + size, cy, fill=C["border"], width=1)
        c.create_line(cx, cy - size, cx, cy + size, fill=C["border"], width=1)
        c.create_oval(cx-4, cy-4, cx+4, cy+4, outline=C["cyan_dim"], width=1)

        # Texto central
        c.create_text(cx, cy + 30, text="SELECCIONA UN VIDEO O CÁMARA",
                      font=("Consolas", 11), fill=C["text_lo"])
        c.create_text(cx, cy + 50, text="PARA INICIAR EL ANÁLISIS",
                      font=("Consolas", 9), fill=C["text_lo"])

        # Marco esquinas
        m = 20
        for (x1, y1, x2, y2) in [
            (m, m, m+30, m), (m, m, m, m+30),
            (w-m-30, m, w-m, m), (w-m, m, w-m, m+30),
            (m, h-m, m+30, h-m), (m, h-m-30, m, h-m),
            (w-m-30, h-m, w-m, h-m), (w-m, h-m-30, w-m, h-m),
        ]:
            c.create_line(x1, y1, x2, y2, fill=C["cyan_dim"], width=2)

    # ── Panel derecho ─────────────────────────────────────────────────────────

    def _construir_panel_derecho(self, parent) -> None:
        panel = tk.Frame(parent, bg=C["bg_panel"], width=360)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        panel.pack_propagate(False)
        panel.columnconfigure(0, weight=1)

        # ── Tarjetas de métricas ──────────────────────────────────────────────
        self._construir_metricas(panel)

        _separator(panel, color=C["border"]).pack(fill="x", pady=4, padx=8)

        # ── Gráfica de FPS ────────────────────────────────────────────────────
        fps_hdr = tk.Frame(panel, bg=C["bg_panel"])
        fps_hdr.pack(fill="x", padx=8)
        tk.Label(fps_hdr, text="◈ RENDIMIENTO  //  FPS",
                 font=("Consolas", 8, "bold"),
                 fg=C["cyan"], bg=C["bg_panel"]).pack(side="left")

        self.fps_graph = _FpsGraph(panel)
        self.fps_graph.pack(fill="x", padx=8, pady=(2, 0))

        _separator(panel, color=C["border"]).pack(fill="x", pady=4, padx=8)

        # ── Opciones ──────────────────────────────────────────────────────────
        self._construir_opciones(panel)

        _separator(panel, color=C["border"]).pack(fill="x", pady=4, padx=8)

        # ── Tabla de eventos ──────────────────────────────────────────────────
        ev_hdr = tk.Frame(panel, bg=C["bg_panel"])
        ev_hdr.pack(fill="x", padx=8)
        tk.Label(ev_hdr, text="◈ EVENTOS DETECTADOS",
                 font=("Consolas", 8, "bold"),
                 fg=C["cyan"], bg=C["bg_panel"]).pack(side="left")
        self._ev_count_var = tk.StringVar(value="0 reg.")
        tk.Label(ev_hdr, textvariable=self._ev_count_var,
                 font=("Consolas", 7),
                 fg=C["text_lo"], bg=C["bg_panel"]).pack(side="right")

        self.event_table = _EventTable(panel)
        self.event_table.pack(fill="both", expand=True, padx=8, pady=(2, 0))

        # Botón ver BD
        _NeonButton(panel, "[ CARGAR EVENTOS DE BASE DE DATOS ]",
                    self._cargar_eventos_db, color=C["cyan"]).pack(
                        fill="x", padx=8, pady=6)

        # ── Log terminal ──────────────────────────────────────────────────────
        log_hdr = tk.Frame(panel, bg=C["bg_panel"])
        log_hdr.pack(fill="x", padx=8)
        tk.Label(log_hdr, text="◈ LOG SISTEMA",
                 font=("Consolas", 8, "bold"),
                 fg=C["green_dim"], bg=C["bg_panel"]).pack(side="left")

        self.terminal = _LogTerminal(panel)
        self.terminal.pack(fill="x", padx=8, pady=(2, 6))
        # Altura fija del terminal
        self.terminal.configure(height=90)

    def _construir_metricas(self, parent) -> None:
        met_hdr = tk.Frame(parent, bg=C["bg_panel"])
        met_hdr.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(met_hdr, text="◈ MÉTRICAS EN TIEMPO REAL",
                 font=("Consolas", 8, "bold"),
                 fg=C["cyan"], bg=C["bg_panel"]).pack(side="left")

        cards_row1 = tk.Frame(parent, bg=C["bg_panel"])
        cards_row1.pack(fill="x", padx=8, pady=2)

        self.card_activos = _StatCard(cards_row1, "ACTIVOS",  color=C["green"],  width=80)
        self.card_eventos = _StatCard(cards_row1, "EVENTOS",  color=C["yellow"], width=80)
        self.card_ciclos  = _StatCard(cards_row1, "CICLOS",   color=C["magenta"], width=80)
        self.card_fps     = _StatCard(cards_row1, "FPS",      color=C["cyan"],    width=80)

        for card in (self.card_activos, self.card_eventos, self.card_ciclos, self.card_fps):
            card.pack(side="left", padx=3, pady=2)

        # Barra de progreso custom
        prog_f = tk.Frame(parent, bg=C["bg_panel"])
        prog_f.pack(fill="x", padx=8, pady=(2, 0))

        tk.Label(prog_f, text="PROGRESO:", font=("Consolas", 7),
                 fg=C["text_lo"], bg=C["bg_panel"]).pack(side="left")
        self._prog_lbl = tk.Label(prog_f, text="0%",
                                   font=("Consolas", 7),
                                   fg=C["cyan"], bg=C["bg_panel"])
        self._prog_lbl.pack(side="right")

        self._prog_canvas = tk.Canvas(parent, height=6, bg=C["bg_deep"],
                                       highlightthickness=0)
        self._prog_canvas.pack(fill="x", padx=8, pady=(2, 0))
        self._prog_canvas.bind("<Configure>", lambda _: self._redraw_progress(0))
        self._prog_pct: float = 0.0

        # Frame info
        frame_f = tk.Frame(parent, bg=C["bg_panel"])
        frame_f.pack(fill="x", padx=8, pady=(2, 0))
        tk.Label(frame_f, text="FRAME:", font=("Consolas", 7),
                 fg=C["text_lo"], bg=C["bg_panel"]).pack(side="left")
        self._frame_lbl = tk.Label(frame_f, text="0 / —",
                                    font=("Consolas", 7, "bold"),
                                    fg=C["text_hi"], bg=C["bg_panel"])
        self._frame_lbl.pack(side="left", padx=4)

    def _redraw_progress(self, pct: float):
        c = self._prog_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2:
            return
        # Fondo
        c.create_rectangle(0, 0, w, h, fill=C["bg_deep"], outline="")
        # Barra
        bar_w = int(w * pct / 100)
        if bar_w > 0:
            # Gradiente: cyan → green según progreso
            c.create_rectangle(0, 0, bar_w, h, fill=C["cyan"], outline="")
        # Brillo en el extremo
        if bar_w > 4:
            c.create_rectangle(bar_w - 4, 0, bar_w, h,
                                fill=C["text_hi"], outline="")

    def _construir_opciones(self, parent) -> None:
        opt_hdr = tk.Frame(parent, bg=C["bg_panel"])
        opt_hdr.pack(fill="x", padx=8)
        tk.Label(opt_hdr, text="◈ CONFIGURACIÓN DEL PIPELINE",
                 font=("Consolas", 8, "bold"),
                 fg=C["cyan"], bg=C["bg_panel"]).pack(side="left")

        opt_f = tk.Frame(parent, bg=C["bg_card"],
                         highlightthickness=1, highlightbackground=C["border"])
        opt_f.pack(fill="x", padx=8, pady=(4, 0))

        # Checks
        chk_data = [
            ("▸ Guardar eventos en BD",        self.var_guardar_db, C["green"]),
            ("▸ OCR números de flota",         self.var_ocr,        C["yellow"]),
            ("▸ Clasificar estado (LSTM/CNN)", self.var_clasif,     C["magenta"]),
        ]
        for txt, var, col in chk_data:
            cf = tk.Frame(opt_f, bg=C["bg_card"])
            cf.pack(fill="x", padx=6, pady=1)
            cb = tk.Checkbutton(
                cf, text=txt, variable=var,
                font=("Consolas", 8),
                fg=col, bg=C["bg_card"],
                selectcolor=C["bg_deep"],
                activeforeground=col, activebackground=C["bg_hover"],
                bd=0, highlightthickness=0,
            )
            cb.pack(side="left")

        # Comboboxes
        combo_data = [
            ("SKIP FRAMES:", self.var_skip, ("1","2","3","4","5","6","8","10")),
            ("IMGSZ:", self.var_imgsz, ("160","320","480","640")),
        ]
        for lbl, var, vals in combo_data:
            cf = tk.Frame(opt_f, bg=C["bg_card"])
            cf.pack(fill="x", padx=6, pady=2)
            tk.Label(cf, text=lbl, font=("Consolas", 8),
                     fg=C["text_mid"], bg=C["bg_card"]).pack(side="left")

            # Combobox estilizado manualmente con un OptionMenu oscuro
            om = tk.OptionMenu(cf, var, *vals)
            om.configure(
                font=("Consolas", 8), fg=C["cyan"], bg=C["bg_deep"],
                activeforeground=C["cyan"], activebackground=C["bg_hover"],
                highlightthickness=0, bd=0, relief="flat",
                indicatoron=True,
            )
            om["menu"].configure(
                font=("Consolas", 8), fg=C["cyan"], bg=C["bg_deep"],
                activeforeground=C["bg_deep"], activebackground=C["cyan"],
            )
            om.pack(side="right")

    # ── Status bar ────────────────────────────────────────────────────────────

    def _construir_statusbar(self) -> None:
        sb = tk.Frame(self, bg=C["bg_main"],
                      highlightthickness=1, highlightbackground=C["border_dim"])
        sb.grid(row=2, column=0, sticky="ew")
        sb.columnconfigure(1, weight=1)

        # Indicador de estado (puntito + texto)
        self._status_dot = tk.Label(sb, text="●", font=("Consolas", 9),
                                     fg=C["green"], bg=C["bg_main"], padx=6)
        self._status_dot.grid(row=0, column=0, sticky="w")

        tk.Label(sb, textvariable=self.var_estado,
                 font=("Consolas", 9, "bold"),
                 fg=C["text_hi"], bg=C["bg_main"]).grid(row=0, column=1, sticky="w")

        tk.Label(sb, text=f"MODEL: {self.config_app.model_path}  //  DB: {self.config_app.db_url}",
                 font=("Consolas", 7),
                 fg=C["text_lo"], bg=C["bg_main"],
                 padx=12).grid(row=0, column=2, sticky="e")

    # ── Animaciones ───────────────────────────────────────────────────────────

    def _animar_header(self) -> None:
        """Animación continua: LEDs pulsantes y texto de scan."""
        self._anim_tick += 1
        t = self._anim_tick

        proc_vivo = self._procesador is not None and self._procesador.is_alive()
        pausado   = proc_vivo and (self._procesador.estado.pausado if proc_vivo else False)

        # LED de procesando: pulsa verde si activo
        if proc_vivo and not pausado:
            on = (t % 8) < 4
            self._led_proc.configure(fg=C["green"] if on else C["green_dark"])
        else:
            self._led_proc.configure(fg=C["border"])

        # LED de tracking: pulsa cyan si hay activos
        activos = int(self._procesador.estado.activos) if proc_vivo else 0
        if activos > 0:
            on = (t % 6) < 3
            self._led_track.configure(fg=C["cyan"] if on else C["cyan_dark"])
        else:
            self._led_track.configure(fg=C["border"])

        # LED de BD
        if proc_vivo and self.var_guardar_db.get():
            on = (t % 20) < 2
            self._led_db.configure(fg=C["yellow"] if on else C["yellow_dim"])
        else:
            self._led_db.configure(fg=C["border"])

        # Texto de scan
        chars = ["SCANNING", "SCAN·ING", "SCANN·NG", "SCANNI·G", "SCANNIN·",
                 "SCANNING", "·CANNING", "S·ANNING", "SC·NNING"]
        if proc_vivo and not pausado:
            self._scan_label.configure(text=chars[t % len(chars)])
        elif pausado:
            on = (t % 10) < 5
            self._scan_label.configure(
                text="⚠  PAUSADO" if on else "         ")
        else:
            # Idle: "STANDBY..."
            dots = "." * (1 + (t % 4))
            self._scan_label.configure(text=f"STANDBY{dots:<4}", fg=C["text_lo"])

        # Punto de status parpadeante
        if proc_vivo and not pausado:
            self._status_dot.configure(fg=C["green"] if (t % 8 < 4) else C["green_dark"])
        elif pausado:
            self._status_dot.configure(fg=C["yellow"] if (t % 10 < 5) else C["yellow_dim"])
        else:
            self._status_dot.configure(fg=C["text_lo"])

        self.after(120, self._animar_header)

    # ── Acciones del usuario ──────────────────────────────────────────────────

    @staticmethod
    def _es_camara(fuente: str | int) -> bool:
        if isinstance(fuente, int):
            return True
        if isinstance(fuente, str):
            v = fuente.strip().lower()
            return v.isdigit() or v.startswith(("rtsp://", "http://", "https://"))
        return False

    def _texto_fuente(self) -> str:
        if self._fuente is None:
            return "SIN FUENTE"
        if self._es_camara(self._fuente):
            return f"CAM:{self._fuente}"
        return Path(str(self._fuente)).name[:40]

    def _abrir_video(self) -> None:
        inicial = str(Path(self.config_app.video_path).parent) if self.config_app.video_path else str(Path.cwd())
        ruta = filedialog.askopenfilename(
            title="Seleccionar video",
            initialdir=inicial,
            filetypes=(("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Todos", "*.*")),
        )
        if not ruta:
            return
        self._fuente = ruta
        self.var_fuente.set(self._texto_fuente())
        self.var_estado.set("VIDEO CARGADO  //  PULSA INICIAR")
        self._dibujar_grid_placeholder()
        self.terminal.log(f"Video seleccionado: {ruta}", "info")

    def _usar_camara(self) -> None:
        defecto = str(self.config_app.camera_source) if self.config_app.camera_source else "0"
        valor = simpledialog.askstring(
            "Cámara", "Índice (0=webcam) o URL RTSP/HTTP:", initialvalue=defecto, parent=self,
        )
        if valor is None or not valor.strip():
            return
        valor = valor.strip()
        self._fuente = int(valor) if valor.isdigit() else valor
        self.var_fuente.set(self._texto_fuente())
        self.var_estado.set("CÁMARA SELECCIONADA  //  PULSA INICIAR")
        self.terminal.log(f"Cámara/stream: {valor}", "info")

    def _iniciar(self) -> None:
        if self._procesador is not None and self._procesador.is_alive():
            return
        if self._fuente is None:
            messagebox.showwarning("Sin fuente", "Selecciona un video o cámara primero.")
            return
        if Image is None:
            messagebox.showerror("Falta Pillow", "pip install pillow")
            return
        if not Path(self.config_app.model_path).exists():
            messagebox.showerror(
                "Modelo no encontrado",
                f"No existe: {self.config_app.model_path}\n\n"
                "Coloca best.pt en models/ o ajusta MODEL_PATH en .env.",
            )
            return

        opciones = OpcionesAnalisis(
            guardar_db      = self.var_guardar_db.get(),
            usar_ocr        = self.var_ocr.get(),
            usar_clasificador = self.var_clasif.get(),
            track_skip      = int(self.var_skip.get()),
            track_imgsz     = int(self.var_imgsz.get()),
            display_width   = self.config_app.display_width,
        )

        self.event_table.limpiar()
        self._finalizado_avisado = False
        self._t_inicio = time.perf_counter()
        self._procesador = Procesador(self._fuente, opciones, self.config_app)
        self._procesador.start()

        self.btn_pausar.set_state("normal")
        self.var_estado.set("INICIALIZANDO PIPELINE...")
        self.terminal.log("Pipeline iniciado.", "info")
        if not self._poll_programado:
            self._poll_programado = True
            self._poll()

    def _pausar(self) -> None:
        if self._procesador is None:
            return
        if self._procesador.estado.pausado:
            self._procesador.reanudar()
            self.btn_pausar.set_text("[ PAUSAR ]")
            self.var_estado.set("PROCESANDO...")
            self.terminal.log("Pipeline reanudado.", "info")
        else:
            self._procesador.pausar()
            self.btn_pausar.set_text("[ REANUDAR ]")
            self.var_estado.set("⚠  PAUSADO")
            self.terminal.log("Pipeline pausado.", "info")

    def _detener(self) -> None:
        if self._procesador is not None and self._procesador.is_alive():
            self._procesador.detener()
            self.var_estado.set("DETENIENDO...")
            self.terminal.log("Señal de parada enviada.", "info")

    def _cerrar(self) -> None:
        self._poll_programado = False
        if self._procesador is not None and self._procesador.is_alive():
            self._procesador.detener()
            self._procesador.join(timeout=2.0)
        self.destroy()

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        proc = self._procesador
        if proc is None:
            self._poll_programado = False
            return

        # Mostrar frame
        frame = proc.obtener_frame()
        if frame is not None:
            self._mostrar_frame(frame)

        # Actualizar estadísticas
        est = proc.estado
        total = est.total_frames

        self.card_activos.update(str(est.activos))
        self.card_eventos.update(str(est.eventos))
        self.card_ciclos.update(str(est.ciclos))
        self.card_fps.update(f"{est.fps:.1f}")
        self.fps_graph.push(est.fps)

        # Barra de progreso
        pct = est.progreso
        self._prog_pct = pct
        self._redraw_progress(pct)
        self._prog_lbl.configure(text=f"{pct:.1f}%")
        self._frame_lbl.configure(text=f"{est.frame_actual} / {total if total else '?'}")

        self.var_estado.set(est.mensaje.upper())

        # Drenar eventos
        self._drenar_eventos(proc)

        # Fin
        if proc.terminado and not self._finalizado_avisado:
            self._finalizado_avisado = True
            self.btn_pausar.set_state("disabled")
            self.btn_pausar.set_text("[ PAUSAR ]")
            elapsed = time.perf_counter() - self._t_inicio
            self.var_estado.set(f"ANÁLISIS COMPLETO  //  {elapsed:.1f}s")
            self.terminal.log(f"Análisis finalizado en {elapsed:.1f}s.", "aparicion")
            # Dibujar grid placeholder de nuevo
            self._dibujar_grid_placeholder()
            if est.error:
                self.terminal.log(f"ERROR: {est.error}", "error")
                messagebox.showerror("Error en el análisis", est.error)
            self._poll_programado = False
            return

        self.after(self.POLL_MS, self._poll)

    def _mostrar_frame(self, frame: np.ndarray) -> None:
        if Image is None:
            return

        c = self.video_canvas
        ancho = max(320, c.winfo_width())
        alto  = max(240, c.winfo_height())

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(rgb)
        imagen.thumbnail((ancho, alto), Image.Resampling.LANCZOS)

        self._photo = ImageTk.PhotoImage(imagen)
        c.delete("all")
        # Centrar la imagen en el canvas
        cx = ancho // 2
        cy = alto // 2
        c.create_image(cx, cy, image=self._photo, anchor="center")

        # Overlay de esquinas sobre el canvas
        iw, ih = imagen.size
        x0 = cx - iw // 2
        y0 = cy - ih // 2
        x1 = x0 + iw
        y1 = y0 + ih
        cs = 18  # tamaño de las esquinas
        for (px, py, dx, dy) in [
            (x0, y0,  cs, 0), (x0, y0, 0,  cs),
            (x1, y0, -cs, 0), (x1, y0, 0,  cs),
            (x0, y1,  cs, 0), (x0, y1, 0, -cs),
            (x1, y1, -cs, 0), (x1, y1, 0, -cs),
        ]:
            c.create_line(px, py, px + dx, py + dy, fill=C["cyan"], width=2)

    def _drenar_eventos(self, proc: Procesador) -> None:
        new = 0
        while True:
            try:
                ev = proc.eventos.get_nowait()
            except queue.Empty:
                break
            self.event_table.insertar(ev)
            tag = ev.tipo.lower() if ev.tipo.lower() in ("aparicion", "desaparicion", "zona") else "info"
            tipo_sym = {"APARICION": "▶", "DESAPARICION": "◀", "ZONA": "◈"}.get(ev.tipo, "·")
            self.terminal.log(
                f"{tipo_sym} {ev.tipo}  ID:{ev.track_id}  {ev.maquina or ''}  "
                f"{'conf=' + str(round(ev.confianza, 2)) if ev.confianza else ''}  "
                f"{ev.zona or ''}",
                tag,
            )
            new += 1
            if new > 0:
                self.card_eventos.flash()

        # Actualizar contador de filas
        n = len(self.event_table._rows)
        self._ev_count_var.set(f"{n} reg.")

    def _cargar_eventos_db(self) -> None:
        self.event_table.limpiar()
        try:
            migrate()
            with get_session() as session:
                eventos = RepositorioEventos(session).obtener_recientes(200)
                datos = [
                    EventoUI(
                        timestamp=e.timestamp,
                        tipo=(e.tipo_evento_ciclo or "evento").upper(),
                        maquina=e.tipo_maquina,
                        track_id=e.id,
                        confianza=e.confianza,
                        zona=e.zona_nombre or None,
                    )
                    for e in eventos
                ]
        except Exception as err:
            messagebox.showerror("Error de base de datos", str(err))
            self.terminal.log(f"Error BD: {err}", "error")
            return

        for ev in reversed(datos):
            self.event_table.insertar(ev)
        n = len(datos)
        self.var_estado.set(f"MOSTRANDO {n} EVENTOS DE LA BD")
        self.terminal.log(f"Cargados {n} eventos desde la base de datos.", "info")


def main() -> None:
    app = AplicacionDesk()
    app.mainloop()


if __name__ == "__main__":
    main()
