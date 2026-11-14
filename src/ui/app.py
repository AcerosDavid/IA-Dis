"""
Interfaz de escritorio para análisis de video con IA.

Una sola ventana: cargar video → analiza en segundo plano → muestra
el resultado ahí mismo. Sin pestañas, sin preguntar rutas, con historial.
"""
from __future__ import annotations

import os
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
        for candidato in (base / "tk" / "tk8.6", base / "lib" / "tk8.6"):
            if (candidato / "tk.tcl").exists():
                os.environ["TK_LIBRARY"] = str(candidato)
                break


_configurar_tcl_tk()

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from src.batch.jobs import GestorJobs
from src.config import load_config
from src.pipeline.runner import EventoUI

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


# ── Tema claro, simple y accesible ──────────────────────────────────────────────

T = {
    "bg":           "#eef1f6",
    "panel":        "#ffffff",
    "panel_alt":    "#f7f8fa",
    "border":       "#dfe3e8",
    "text":         "#1e293b",
    "text_dim":     "#64748b",
    "accent":       "#2563eb",
    "accent_light": "#dbeafe",
    "success":      "#16a34a",
    "success_light":"#dcfce7",
    "warning":      "#d97706",
    "warning_light":"#fef3c7",
    "danger":       "#dc2626",
    "danger_light": "#fee2e2",
    "track":        "#e2e8f0",
}

ESTADO_INFO = {
    "pendiente":       ("En espera",      T["text_dim"],   "◐"),
    "procesando":      ("Analizando...",   T["accent"],    "▶"),
    "generando_video": ("Generando video", T["accent"],    "◈"),
    "completado":      ("Completado",      T["success"],   "✓"),
    "error":           ("Error",           T["danger"],    "✗"),
    "cancelado":       ("Cancelado",       T["warning"],   "⊘"),
}

TIPO_COLOR = {
    "APARICION":   T["success"],
    "DESAPARICION": T["danger"],
    "ZONA":        T["warning"],
}


# ── Helpers simples ─────────────────────────────────────────────────────────────

def btn(parent, text, command, color="accent", width=None):
    """Botón plano limpio."""
    bg = T.get(color, T["accent"])
    b = tk.Button(parent, text=text, command=command, font=("Segoe UI", 9),
                  bg=bg, fg="white", activebackground=bg, activeforeground="white",
                  bd=0, relief="flat", padx=14, pady=7, cursor="hand2")
    if width:
        b.configure(width=width)
    return b


def lbl(parent, text="", font=("Segoe UI", 9), fg=None, bg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg or T["text"], bg=bg or T["panel"], **kw)


def seccion(parent, titulo):
    """Cabecera de sección con título y línea separadora."""
    f = tk.Frame(parent, bg=T["panel"])
    f.pack(fill="x")
    lbl(f, text=titulo, font=("Segoe UI", 9, "bold"), fg=T["text"]).pack(anchor="w")
    tk.Frame(f, bg=T["border"], height=1).pack(fill="x", pady=(3, 0))
    return f


# ── Ventana principal ──────────────────────────────────────────────────────────

class AplicacionDesk(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.config_app = load_config()

        self.title("Análisis de Video con IA")
        self.geometry("1200x720")
        self.minsize(900, 600)
        self.configure(bg=T["bg"])
        self._icono()

        # Estado
        self._gestor: GestorJobs | None = None
        self._job_actual: int | None = None
        self._poll_id = None
        self._t_inicio = 0.0
        self._cargando = False  # evita doble carga

        # Player
        self._player_cap = None
        self._player_playing = False
        self._player_frame = 0
        self._player_total = 0
        self._player_fps = 30.0
        self._player_after = None
        self._player_photo = None

        self._crear_ui()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

        # Inicializar gestor diferido
        self.after(100, self._init_gestor)

    def _icono(self) -> None:
        icono = RAIZ_PROYECTO / "logo.ico"
        if sys.platform == "win32" and icono.exists():
            try:
                self.iconbitmap(str(icono))
            except tk.TclError:
                pass

    def _init_gestor(self) -> None:
        try:
            self._gestor = GestorJobs(self.config_app)
            self._refrescar_historial()
            self._iniciar_poll()
        except Exception as e:
            print(f"[UI] Error inicializando gestor: {e}")

    # ── Construcción de la UI ──────────────────────────────────────────────────

    def _crear_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._toolbar()
        self._cuerpo()
        self._barra_eventos()

    def _toolbar(self) -> None:
        bar = tk.Frame(self, bg=T["panel"], height=54)
        bar.grid(row=0, column=0, sticky="ew")
        bar.pack_propagate(False)

        tk.Frame(bar, bg=T["border"], height=1).pack(side="bottom", fill="x")

        inner = tk.Frame(bar, bg=T["panel"])
        inner.pack(fill="x", padx=12, pady=8)

        self.b_cargar = btn(inner, "Cargar Video", self._cargar_video, "success")
        self.b_cargar.pack(side="left", padx=(0, 6))

        self.b_detener = btn(inner, "Detener", self._detener, "danger")
        self.b_detener.pack(side="left", padx=6)
        self.b_detener.configure(state="disabled", bg=T["border"],
                                   fg=T["text_dim"], activebackground=T["border"])

        self.b_eliminar = btn(inner, "Eliminar", self._eliminar, "danger")
        self.b_eliminar.pack(side="left", padx=6)
        self.b_eliminar.configure(state="disabled", bg=T["border"],
                                    fg=T["text_dim"], activebackground=T["border"])

        self.var_fuente = tk.StringVar(value="Sin video cargado")
        lbl(inner, textvariable=self.var_fuente, font=("Segoe UI", 9),
            fg=T["text_dim"]).pack(side="left", padx=16)

    def _cuerpo(self) -> None:
        cuerpo = tk.Frame(self, bg=T["bg"])
        cuerpo.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        cuerpo.columnconfigure(0, weight=3)
        cuerpo.columnconfigure(1, weight=0, minsize=300)
        cuerpo.rowconfigure(0, weight=1)

        self._panel_video(cuerpo)
        self._panel_lateral(cuerpo)

    # ── Panel de video (izquierda) ─────────────────────────────────────────────

    def _panel_video(self, parent) -> None:
        cont = tk.Frame(parent, bg=T["panel"], highlightthickness=1,
                       highlightbackground=T["border"])
        cont.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        cont.columnconfigure(0, weight=1)
        cont.rowconfigure(0, weight=1)

        # Canvas de video
        self.canvas = tk.Canvas(cont, bg="#0d1117", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._dibujar_placeholder()

        # Controles del reproductor
        ctrl = tk.Frame(cont, bg=T["panel_alt"])
        ctrl.grid(row=1, column=0, sticky="ew")
        ctrl.columnconfigure(2, weight=1)

        self.b_play = btn(ctrl, "▶", self._player_play_pause, "accent", width=3)
        self.b_play.grid(row=0, column=0, padx=6, pady=6)
        self.b_play.configure(state="disabled", bg=T["border"], fg=T["text_dim"],
                               activebackground=T["border"])

        self.b_stop = btn(ctrl, "■", self._player_stop, "danger", width=3)
        self.b_stop.grid(row=0, column=1, padx=(0, 6), pady=6)
        self.b_stop.configure(state="disabled", bg=T["border"], fg=T["text_dim"],
                               activebackground=T["border"])

        self.slider = tk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                               command=self._player_seek, bg=T["panel_alt"],
                               fg=T["text"], troughcolor=T["track"],
                               highlightthickness=0, bd=0, showvalue=False,
                               sliderrelief="flat")
        self.slider.grid(row=0, column=2, sticky="ew", padx=4, pady=6)

        self.var_tiempo = tk.StringVar(value="0:00 / 0:00")
        lbl(ctrl, textvariable=self.var_tiempo, font=("Segoe UI", 8),
            fg=T["text_dim"], bg=T["panel_alt"]).grid(row=0, column=3, padx=8)

    def _dibujar_placeholder(self) -> None:
        c = self.canvas
        c.delete("all")
        self.canvas.after(50, self._dibujar_placeholder_delayed)

    def _dibujar_placeholder_delayed(self) -> None:
        c = self.canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            self.canvas.after(100, self._dibujar_placeholder_delayed)
            return
        c.create_text(w // 2, h // 2 - 10,
                       text="Carga un video para comenzar",
                       font=("Segoe UI", 14), fill="#64748b")
        c.create_text(w // 2, h // 2 + 16,
                       text="El análisis se ejecutará en segundo plano",
                       font=("Segoe UI", 9), fill="#475569")

    def _dibujar_progreso(self, pct, frame, total, fps, activos) -> None:
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10:
            return
        cx, cy = w // 2, h // 2

        c.create_text(cx, cy - 50, text="Analizando video...",
                       font=("Segoe UI", 16, "bold"), fill="#60a5fa")

        # Barra de progreso
        bw = min(w - 160, 440)
        bx = cx - bw // 2
        by = cy - 10
        bh = 22
        c.create_rectangle(bx, by, bx + bw, by + bh, fill="#1e293b", outline="#334155")
        fw = int(bw * pct / 100)
        if fw > 0:
            c.create_rectangle(bx, by, bx + fw, by + bh, fill="#3b82f6", outline="")
        c.create_text(cx, by + bh // 2, text=f"{pct:.1f}%",
                       font=("Segoe UI", 10, "bold"), fill="white")

        total_s = str(total) if total else "?"
        c.create_text(cx, by + bh + 22,
                       text=f"Frame {frame} / {total_s}    {fps:.1f} FPS    Activos: {activos}",
                       font=("Segoe UI", 9), fill="#94a3b8")

    def _dibujar_completado(self) -> None:
        c = self.canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10:
            return
        c.create_text(w // 2, h // 2 - 10, text="Análisis completado",
                       font=("Segoe UI", 14, "bold"), fill="#4ade80")
        c.create_text(w // 2, h // 2 + 16,
                       text="Pulsa ▶ para ver el video procesado",
                       font=("Segoe UI", 9), fill="#64748b")

    # ── Panel lateral (derecha): métricas + configuración + historial ────────────

    def _panel_lateral(self, parent) -> None:
        panel = tk.Frame(parent, bg=T["panel"], highlightthickness=1,
                         highlightbackground=T["border"], width=300)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.pack_propagate(False)
        panel.columnconfigure(0, weight=1)

        # Scrollable
        canvas_s = tk.Canvas(panel, bg=T["panel"], highlightthickness=0)
        sb = tk.Scrollbar(panel, orient="vertical", command=canvas_s.yview,
                         bg=T["panel"], troughcolor=T["panel_alt"])
        canvas_s.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas_s.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas_s, bg=T["panel"])
        canvas_s.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                    lambda e: canvas_s.configure(scrollregion=canvas_s.bbox("all")))
        canvas_s.bind("<Configure>",
                        lambda e: canvas_s.itemconfigure("all", width=e.width))

        self._panel_metricas(inner)
        self._panel_config(inner)
        self._panel_historial(inner)

    def _panel_metricas(self, parent) -> None:
        seccion(parent, "Métricas en tiempo real")

        f = tk.Frame(parent, bg=T["panel"])
        f.pack(fill="x", padx=10, pady=8)

        # Progreso
        pf = tk.Frame(f, bg=T["panel"])
        pf.pack(fill="x", pady=2)
        lbl(pf, text="Progreso", font=("Segoe UI", 8), fg=T["text_dim"]).pack(side="left")
        self.var_progreso = tk.StringVar(value="0%")
        lbl(pf, textvariable=self.var_progreso, font=("Segoe UI", 8, "bold"),
            fg=T["accent"]).pack(side="right")

        self.bar_progreso = tk.Canvas(f, height=10, bg=T["track"], highlightthickness=0)
        self.bar_progreso.pack(fill="x", pady=(2, 6))

        # Métricas en grid 2x3 (frame separado para no mezclar pack y grid)
        grid_f = tk.Frame(f, bg=T["panel"])
        grid_f.pack(fill="x")
        grid_f.columnconfigure(0, weight=1)
        grid_f.columnconfigure(1, weight=1)

        datos = [
            ("Frames", "var_frames", T["text"]),
            ("Detecciones", "var_dets", T["success"]),
            ("Eventos", "var_eventos", T["warning"]),
            ("Activos", "var_activos", T["accent"]),
            ("Ciclos", "var_ciclos", T["text_dim"]),
            ("FPS", "var_fps", T["text"]),
        ]
        for i, (nombre, var_name, color) in enumerate(datos):
            row = i // 2
            col = i % 2
            card = tk.Frame(grid_f, bg=T["panel_alt"], highlightthickness=1,
                           highlightbackground=T["border"])
            card.grid(row=row, column=col, sticky="ew", padx=2, pady=2)
            lbl(card, text=nombre, font=("Segoe UI", 7), fg=T["text_dim"],
                bg=T["panel_alt"]).pack(anchor="w", padx=6, pady=(4, 0))
            var = tk.StringVar(value="0")
            setattr(self, var_name, var)
            lbl(card, textvariable=var, font=("Segoe UI", 16, "bold"),
                fg=color, bg=T["panel_alt"]).pack(anchor="w", padx=6, pady=(0, 4))

    def _panel_config(self, parent) -> None:
        seccion(parent, "Configuración")

        f = tk.Frame(parent, bg=T["panel"])
        f.pack(fill="x", padx=10, pady=8)

        # Skip frames
        rf = tk.Frame(f, bg=T["panel"])
        rf.pack(fill="x", pady=2)
        lbl(rf, text="Procesar 1 de cada N frames", font=("Segoe UI", 8),
            fg=T["text_dim"]).pack(side="left")
        self.var_skip = tk.StringVar(value=str(self.config_app.track_skip))
        ttk.Combobox(rf, textvariable=self.var_skip, width=4, state="readonly",
                      values=("1", "2", "3", "4", "5", "6", "8", "10"),
                      font=("Segoe UI", 8)).pack(side="right")

        # Imgsz
        rf2 = tk.Frame(f, bg=T["panel"])
        rf2.pack(fill="x", pady=2)
        lbl(rf2, text="Resolución de inferencia", font=("Segoe UI", 8),
            fg=T["text_dim"]).pack(side="left")
        self.var_imgsz = tk.StringVar(value=str(self.config_app.track_imgsz))
        ttk.Combobox(rf2, textvariable=self.var_imgsz, width=4, state="readonly",
                      values=("160", "320", "480", "640"),
                      font=("Segoe UI", 8)).pack(side="right")

        # Checks
        self.var_guardar = tk.BooleanVar(value=True)
        self.var_ocr = tk.BooleanVar(value=self.config_app.ocr_habilitado)
        self.var_clasif = tk.BooleanVar(value=self.config_app.clasificador_habilitado)

        for txt, var in [
            ("Guardar eventos en base de datos", self.var_guardar),
            ("Leer números de flota (OCR)", self.var_ocr),
            ("Clasificar estado (LSTM/CNN)", self.var_clasif),
        ]:
            tk.Checkbutton(f, text=txt, variable=var, font=("Segoe UI", 8),
                           fg=T["text"], bg=T["panel"], selectcolor=T["panel_alt"],
                           activebackground=T["panel"], activeforeground=T["text"],
                           bd=0, highlightthickness=0, anchor="w").pack(fill="x", pady=1)

    def _panel_historial(self, parent) -> None:
        seccion(parent, "Historial")

        self.hist_frame = tk.Frame(parent, bg=T["panel"])
        self.hist_frame.pack(fill="x", padx=10, pady=8)
        self._hist_items: dict[int, tk.Frame] = {}

    def _refrescar_historial(self) -> None:
        if self._gestor is None:
            return
        for item in self._hist_items.values():
            item.destroy()
        self._hist_items.clear()

        jobs = self._gestor.listar_jobs()
        if not jobs:
            lbl(self.hist_frame, text="Aún no hay análisis",
                font=("Segoe UI", 8), fg=T["text_dim"], bg=T["panel"]).pack(pady=8)
            return

        for job in jobs:
            self._crear_item_historial(job)

    def _crear_item_historial(self, job) -> None:
        estado_txt, estado_color, estado_sym = ESTADO_INFO.get(
            job["estado"], ("?", T["text_dim"], "?"))

        item = tk.Frame(self.hist_frame, bg=T["panel_alt"],
                        highlightthickness=1, highlightbackground=T["border"],
                        cursor="hand2")
        item.pack(fill="x", pady=2)

        # Fila superior
        top = tk.Frame(item, bg=T["panel_alt"])
        top.pack(fill="x", padx=6, pady=(3, 0))

        tk.Label(top, text=estado_sym, font=("Segoe UI", 9), fg=estado_color,
                 bg=T["panel_alt"]).pack(side="left")

        nombre = job["nombre"][:24]
        lbl(top, text=nombre, font=("Segoe UI", 8, "bold"), fg=T["text"],
            bg=T["panel_alt"]).pack(side="left", padx=4)

        lbl(top, text=estado_txt, font=("Segoe UI", 7), fg=estado_color,
            bg=T["panel_alt"]).pack(side="right")

        # Fila info
        bot = tk.Frame(item, bg=T["panel_alt"])
        bot.pack(fill="x", padx=6, pady=(0, 3))

        if job["total_eventos"]:
            lbl(bot, text=f"{job['total_eventos']} eventos", font=("Segoe UI", 7),
                fg=T["text_dim"], bg=T["panel_alt"]).pack(side="left")
        elif job["estado"] == "procesando":
            pct = job["progreso"] or 0
            lbl(bot, text=f"{pct:.0f}% procesando", font=("Segoe UI", 7),
                fg=T["accent"], bg=T["panel_alt"]).pack(side="left")

        if job["estado"] == "completado" and job["video_disponible"]:
            lbl(bot, text="✓ ver", font=("Segoe UI", 7), fg=T["success"],
                bg=T["panel_alt"]).pack(side="right")

        # Click para ver
        if job["estado"] == "completado" and job["video_disponible"]:
            for w in (item, top, bot):
                w.bind("<Button-1>", lambda e, jid=job["id"]: self._ver_historial(jid))

        self._hist_items[job["id"]] = item

    def _ver_historial(self, job_id: int) -> None:
        """Carga un video del historial en el reproductor."""
        if self._gestor is None:
            return
        jobs = self._gestor.listar_jobs()
        job = next((j for j in jobs if j["id"] == job_id), None)
        if job and job["ruta_salida"] and os.path.exists(job["ruta_salida"]):
            self._job_actual = job_id
            self.var_fuente.set(job["nombre"])
            self._player_cargar(job["ruta_salida"])
            self._refrescar_historial()

    # ── Barra de eventos (abajo) ───────────────────────────────────────────────

    def _barra_eventos(self) -> None:
        cont = tk.Frame(self, bg=T["panel"], highlightthickness=1,
                        highlightbackground=T["border"])
        cont.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        cont.columnconfigure(0, weight=1)

        seccion(cont, "Eventos detectados")

        cols = ("tipo", "maquina", "id", "conf")
        self.tabla = ttk.Treeview(cont, columns=cols, show="headings", height=6)
        for col, titulo, ancho in [
            ("tipo", "Tipo", 130),
            ("maquina", "Máquina", 120),
            ("id", "ID", 60),
            ("conf", "Confianza", 90),
        ]:
            self.tabla.heading(col, text=titulo)
            self.tabla.column(col, width=ancho, anchor="w")

        self.tabla.tag_configure("APARICION", foreground=T["success"])
        self.tabla.tag_configure("DESAPARICION", foreground=T["danger"])
        self.tabla.tag_configure("ZONA", foreground=T["warning"])

        sb = ttk.Scrollbar(cont, orient="horizontal", command=self.tabla.xview)
        self.tabla.configure(xscrollcommand=sb.set)
        self.tabla.pack(fill="x", padx=10, pady=(4, 0))
        sb.pack(fill="x", padx=10, pady=(0, 6))

    def _insertar_evento(self, ev: EventoUI) -> None:
        conf = f"{ev.confianza:.2f}" if ev.confianza is not None else "—"
        self.tabla.insert("", "end", values=(ev.tipo, ev.maquina or "—",
                                              str(ev.track_id), conf),
                          tags=(ev.tipo,))
        if len(self.tabla.get_children()) > 200:
            self.tabla.delete(self.tabla.get_children()[0])

    def _limpiar_eventos(self) -> None:
        for item in self.tabla.get_children():
            self.tabla.delete(item)

    # ── Acciones ────────────────────────────────────────────────────────────────

    def _cargar_video(self) -> None:
        if self._cargando:
            return
        if self._gestor is None:
            messagebox.showwarning("Espera", "El sistema aún se está inicializando.")
            return

        ruta = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=(("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Todos", "*.*")),
        )
        if not ruta:
            return

        if not Path(self.config_app.model_path).exists():
            messagebox.showerror(
                "Modelo no encontrado",
                f"No existe el modelo:\n{self.config_app.model_path}\n\n"
                "Coloca el archivo .pt en models/ o ajusta MODEL_PATH en .env.",
            )
            return

        self._cargando = True
        self._limpiar_eventos()
        try:
            info = self._gestor.crear_job(
                ruta_video=ruta,
                guardar_db=self.var_guardar.get(),
                usar_ocr=self.var_ocr.get(),
                usar_clasificador=self.var_clasif.get(),
                track_skip=int(self.var_skip.get()),
                track_imgsz=int(self.var_imgsz.get()),
            )
            self._job_actual = info["job_id"]
            self._t_inicio = time.perf_counter()
            self.var_fuente.set(info["nombre"])
            self.b_detener.configure(state="normal", bg=T["danger"], fg="white",
                                       activebackground=T["danger"])
            self.b_eliminar.configure(state="disabled", bg=T["border"], fg=T["text_dim"],
                                       activebackground=T["border"])
            self._dibujar_progreso(0, 0, info["total_frames"], 0, 0)
            self._refrescar_historial()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar el análisis:\n{e}")
        finally:
            self._cargando = False

    def _detener(self) -> None:
        if self._job_actual is None or self._gestor is None:
            return
        self._gestor.cancelar_job(self._job_actual)
        self.b_detener.configure(state="disabled", bg=T["border"], fg=T["text_dim"],
                                   activebackground=T["border"])

    def _eliminar(self) -> None:
        if self._job_actual is None or self._gestor is None:
            return
        if not messagebox.askyesno("Eliminar", "¿Eliminar este análisis y sus archivos?"):
            return
        self._gestor.eliminar_job(self._job_actual)
        self._job_actual = None
        self.var_fuente.set("Sin video cargado")
        self._player_detener()
        self._player_limpiar()
        self._dibujar_placeholder()
        self._limpiar_eventos()
        self.b_eliminar.configure(state="disabled", bg=T["border"], fg=T["text_dim"],
                                    activebackground=T["border"])
        self.b_detener.configure(state="disabled", bg=T["border"], fg=T["text_dim"],
                                   activebackground=T["border"])
        self._refrescar_historial()

    # ── Polling ─────────────────────────────────────────────────────────────────

    def _iniciar_poll(self) -> None:
        if self._poll_id is None:
            self._poll()

    def _poll(self) -> None:
        self._poll_id = None
        if self._gestor is not None and self._job_actual is not None:
            info = self._gestor.obtener_estado(self._job_actual)
            if info is not None:
                self._actualizar_metricas(info)

                if info.estado in ("pendiente", "procesando", "generando_video"):
                    self._dibujar_progreso(info.progreso, info.frame_actual,
                                            info.total_frames, info.fps, info.activos)
                    # Drenar eventos
                    for ev in self._gestor.drenar_eventos(self._job_actual):
                        self._insertar_evento(ev)

                elif info.estado == "completado":
                    self._on_completado(info)

                elif info.estado in ("error", "cancelado"):
                    self._on_error(info)

        # Refrescar historial siempre
        self._refrescar_historial()
        self._poll_id = self.after(500, self._poll)

    def _actualizar_metricas(self, info) -> None:
        self.var_progreso.set(f"{info.progreso:.1f}%")
        self._dibujar_barra(info.progreso)
        total_s = str(info.total_frames) if info.total_frames else "?"
        self.var_frames.set(f"{info.frame_actual}/{total_s}")
        self.var_dets.set(str(info.eventos))
        self.var_eventos.set(str(info.eventos))
        self.var_activos.set(str(info.activos))
        self.var_ciclos.set(str(info.ciclos))
        self.var_fps.set(f"{info.fps:.1f}")

    def _dibujar_barra(self, pct) -> None:
        c = self.bar_progreso
        c.delete("all")
        w = c.winfo_width()
        if w < 2:
            return
        fw = int(w * pct / 100)
        if fw > 0:
            c.create_rectangle(0, 0, fw, 10, fill=T["accent"], outline="")

    def _on_completado(self, info) -> None:
        self.b_detener.configure(state="disabled", bg=T["border"], fg=T["text_dim"],
                                   activebackground=T["border"])
        self.b_eliminar.configure(state="normal", bg=T["danger"], fg="white",
                                    activebackground=T["danger"])
        elapsed = time.perf_counter() - self._t_inicio
        self._dibujar_completado()
        # Cargar el video procesado en el reproductor
        if info.ruta_salida and os.path.exists(info.ruta_salida):
            self._player_cargar(info.ruta_salida)
        self._refrescar_historial()

    def _on_error(self, info) -> None:
        self.b_detener.configure(state="disabled", bg=T["border"], fg=T["text_dim"],
                                   activebackground=T["border"])
        self.b_eliminar.configure(state="normal", bg=T["danger"], fg="white",
                                    activebackground=T["danger"])
        if info.error:
            messagebox.showerror("Error en el análisis", info.error)
        self._dibujar_placeholder()
        self._refrescar_historial()

    # ── Reproductor ──────────────────────────────────────────────────────────────

    def _player_cargar(self, ruta: str) -> None:
        self._player_detener()
        self._player_limpiar()
        if self._player_cap is not None:
            self._player_cap.release()
        self._player_cap = cv2.VideoCapture(ruta)
        if not self._player_cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video procesado.")
            return
        self._player_total = int(self._player_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._player_fps = self._player_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._player_frame = 0
        self._player_mostrar(0)
        self._player_actualizar_tiempo()
        # Habilitar controles
        self.b_play.configure(state="normal", bg=T["accent"], fg="white",
                               activebackground=T["accent"])
        self.b_stop.configure(state="normal", bg=T["danger"], fg="white",
                               activebackground=T["danger"])

    def _player_mostrar(self, frame_idx: int) -> None:
        if self._player_cap is None or not self._player_cap.isOpened():
            return
        if Image is None:
            return
        self._player_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._player_cap.read()
        if not ret:
            return
        self._player_frame = frame_idx
        c = self.canvas
        w = max(320, c.winfo_width())
        h = max(240, c.winfo_height())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((w, h), Image.Resampling.BILINEAR)
        self._player_photo = ImageTk.PhotoImage(img)
        c.delete("all")
        c.create_image(w // 2, h // 2, image=self._player_photo, anchor="center")
        if self._player_total > 0:
            self.slider.set((frame_idx / self._player_total) * 100)

    def _player_play_pause(self) -> None:
        if self._player_cap is None:
            return
        if self._player_playing:
            self._player_playing = False
            self.b_play.configure(text="▶")
        else:
            self._player_playing = True
            self.b_play.configure(text="❚❚")
            self._player_tick()

    def _player_tick(self) -> None:
        if not self._player_playing or self._player_cap is None:
            return
        self._player_frame += 1
        if self._player_frame >= self._player_total:
            self._player_playing = False
            self.b_play.configure(text="▶")
            self._player_frame = 0
            return
        self._player_mostrar(self._player_frame)
        self._player_actualizar_tiempo()
        delay = max(1, int(1000 / self._player_fps))
        self._player_after = self.after(delay, self._player_tick)

    def _player_stop(self) -> None:
        self._player_detener()
        self._player_frame = 0
        if self._player_cap is not None:
            self._player_mostrar(0)
        self._player_actualizar_tiempo()

    def _player_detener(self) -> None:
        self._player_playing = False
        if self._player_after is not None:
            self.after_cancel(self._player_after)
            self._player_after = None
        if hasattr(self, "b_play"):
            self.b_play.configure(text="▶")

    def _player_seek(self, val: str) -> None:
        if self._player_cap is None or self._player_total == 0:
            return
        if self._player_playing:
            return  # no hacer seek mientras reproduce
        frame = int(float(val) / 100 * self._player_total)
        self._player_mostrar(frame)
        self._player_actualizar_tiempo()

    def _player_actualizar_tiempo(self) -> None:
        if self._player_cap is None or self._player_fps == 0:
            self.var_tiempo.set("0:00 / 0:00")
            return
        cur = self._player_frame / self._player_fps
        tot = self._player_total / self._player_fps if self._player_total else 0
        self.var_tiempo.set(f"{self._fmt(cur)} / {self._fmt(tot)}")

    @staticmethod
    def _fmt(seg: float) -> str:
        m = int(seg // 60)
        s = int(seg % 60)
        return f"{m}:{s:02d}"

    def _player_limpiar(self) -> None:
        self._player_playing = False
        if self._player_cap is not None:
            self._player_cap.release()
            self._player_cap = None
        if hasattr(self, "b_play"):
            self.b_play.configure(state="disabled", text="▶", bg=T["border"],
                                   fg=T["text_dim"], activebackground=T["border"])
            self.b_stop.configure(state="disabled", bg=T["border"],
                                   fg=T["text_dim"], activebackground=T["border"])
        self.slider.set(0)
        self.var_tiempo.set("0:00 / 0:00")

    # ── Cerrar ─────────────────────────────────────────────────────────────────

    def _cerrar(self) -> None:
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self._player_detener()
        if self._player_cap is not None:
            self._player_cap.release()
        if self._gestor is not None:
            self._gestor.shutdown()
        self.destroy()


def main() -> None:
    app = AplicacionDesk()
    app.mainloop()


if __name__ == "__main__":
    main()
