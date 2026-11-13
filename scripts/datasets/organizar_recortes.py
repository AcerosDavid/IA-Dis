"""
organizar_recortes.py — Paso 2: Clasifica los recortes con teclas de teclado.

Lee los recortes sin clasificar de data/muestras_sin_clasificar/
y te los muestra uno por uno en pantalla. Tú presionas una tecla
y el archivo se mueve a la carpeta del estado correcto.

═══════════════════════════════════════════════════════════════
  ESTADOS REDEFINIDOS
═══════════════════════════════════════════════════════════════
  El movimiento (quieto/andando) lo detecta el tracker automáticamente.
  El clasificador solo necesita saber el ESTADO DE CARGA de la máquina:

  EXCAVADORA / PALA:
    1 → escarbando_cargando   (cucharón en operación)
    2 → descargando           (volcando material)

  CAMIÓN:
    1 → vacio                 (tolva vacía)
    2 → cargado               (tolva con material)

═══════════════════════════════════════════════════════════════
  CONTROLES
═══════════════════════════════════════════════════════════════
    1 / 2   → clasificar en estado 1 o 2  (mueve el archivo)
    D       → descartar (mover a carpeta _descartado, no se entrena)
    S       → saltar sin clasificar (queda en sin_clasificar para después)
    ←  →    → ir al recorte anterior / siguiente sin clasificar
    Z       → deshacer última clasificación
    Q / ESC → guardar progreso y salir

═══════════════════════════════════════════════════════════════
  USO
═══════════════════════════════════════════════════════════════
    python organizar_recortes.py                        # todos los tipos
    python organizar_recortes.py --tipo camion          # solo camiones
    python organizar_recortes.py --tipo excavadora
    python organizar_recortes.py --origen data/muestras_sin_clasificar
    python organizar_recortes.py --destino data/estados

SALIDA (carpetas creadas automáticamente):
    data/estados/camion/vacio/
    data/estados/camion/cargado/
    data/estados/excavadora/escarbando_cargando/
    data/estados/excavadora/descargando/
    data/estados/pala/escarbando_cargando/
    data/estados/pala/descargando/
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ══════════════════════════════════════════════════════════════════════════════
#  TAXONOMÍA — estados por tipo de máquina
# ══════════════════════════════════════════════════════════════════════════════

ESTADOS: dict[str, dict[str, str]] = {
    "camion": {
        "1": "vacio",
        "2": "cargado",
    },
    "excavadora": {
        "1": "escarbando_cargando",
        "2": "descargando",
    },
    "pala": {
        "1": "escarbando_cargando",
        "2": "descargando",
    },
}

# Colores BGR por estado
_COLOR_ESTADO: dict[str, tuple[int, int, int]] = {
    "vacio":                (200, 200, 255),
    "cargado":              (0,   200,  50),
    "escarbando_cargando":  (0,   220, 255),
    "descargando":          (0,    50, 255),
}
_COLOR_TIPO: dict[str, tuple[int, int, int]] = {
    "camion":     (0,   255,  50),
    "excavadora": (0,   200, 255),
    "pala":       (255,  50, 180),
}

# Tamaño de la ventana de visualización
DISP_W = 960
DISP_H = 720


# ══════════════════════════════════════════════════════════════════════════════
#  ORGANIZADOR
# ══════════════════════════════════════════════════════════════════════════════

class Organizador:
    """
    Muestra recortes uno a uno y los mueve a la carpeta correcta
    según la tecla que presiones.
    """

    def __init__(
        self,
        origen_dir:  Path,
        destino_dir: Path,
        tipos:       list[str],
    ) -> None:
        self.origen_dir  = origen_dir
        self.destino_dir = destino_dir
        self.tipos       = tipos

        # Crear todas las carpetas destino de una vez
        self._crear_carpetas()

        # Cargar lista de archivos pendientes (todos los tipos solicitados)
        self._pendientes: list[Path] = self._cargar_pendientes()
        self._indice: int = 0

        # Historial para deshacer (lista de (ruta_origen, ruta_destino))
        self._historial: list[tuple[Path, Path]] = []

        # Contadores de clasificación
        self._conteo: dict[str, dict[str, int]] = {
            tipo: {est: 0 for est in mapa.values()}
            for tipo, mapa in ESTADOS.items()
        }
        self._descartados   = 0
        self._saltados      = 0
        self._ya_clasificados = 0

    # ── Ejecución ─────────────────────────────────────────────────────────────

    def ejecutar(self) -> None:
        if not self._pendientes:
            print("[INFO] No hay recortes pendientes de clasificar.")
            print(f"       Revisa: {self.origen_dir.resolve()}")
            return

        total = len(self._pendientes)
        print(f"\n[INFO] {total} recortes para clasificar")
        print(f"       Origen : {self.origen_dir.resolve()}")
        print(f"       Destino: {self.destino_dir.resolve()}\n")

        cv2.namedWindow("Organizador de Recortes", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Organizador de Recortes", DISP_W, DISP_H)

        while self._indice < len(self._pendientes):
            ruta_actual = self._pendientes[self._indice]

            # Detectar tipo desde la subcarpeta del origen
            tipo = ruta_actual.parent.name
            if tipo not in ESTADOS:
                # Archivo en raíz sin subcarpeta de tipo → saltar
                self._indice += 1
                continue

            # Cargar imagen
            img = cv2.imread(str(ruta_actual))
            if img is None:
                self._indice += 1
                continue

            # Construir frame de visualización
            frame_vis = self._construir_frame(
                img, tipo, ruta_actual, self._indice, len(self._pendientes)
            )
            cv2.imshow("Organizador de Recortes", frame_vis)

            # Esperar tecla (sin timeout — espera al usuario)
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key).lower() if 32 <= key <= 126 else ""

            # ── Clasificar en estado 1 o 2 ────────────────────────────────────
            if key_char in ("1", "2"):
                mapa   = ESTADOS[tipo]
                estado = mapa.get(key_char)
                if estado:
                    destino = self.destino_dir / tipo / estado / ruta_actual.name
                    self._mover(ruta_actual, destino)
                    self._conteo[tipo][estado] += 1
                    self._historial.append((destino, ruta_actual))
                    print(
                        f"  [{key_char}] {tipo}/{estado:<30} "
                        f"({self._conteo[tipo][estado]}) "
                        f"← {ruta_actual.name}"
                    )
                    self._indice += 1

            # ── Descartar ─────────────────────────────────────────────────────
            elif key_char == "d":
                destino = self.destino_dir / "_descartado" / tipo / ruta_actual.name
                self._mover(ruta_actual, destino)
                self._descartados += 1
                self._historial.append((destino, ruta_actual))
                print(f"  [D] Descartado: {ruta_actual.name}")
                self._indice += 1

            # ── Saltar ────────────────────────────────────────────────────────
            elif key_char == "s":
                self._saltados += 1
                print(f"  [S] Saltado: {ruta_actual.name}")
                self._indice += 1

            # ── Navegar ───────────────────────────────────────────────────────
            elif key in (81, 2) or key_char == "a":   # ← anterior
                self._indice = max(0, self._indice - 1)

            elif key in (83, 3) or key_char == "f":   # → siguiente
                self._indice = min(len(self._pendientes) - 1, self._indice + 1)

            # ── Deshacer ─────────────────────────────────────────────────────
            elif key_char == "z":
                self._deshacer(tipo)

            # ── Salir ─────────────────────────────────────────────────────────
            elif key in (ord("q"), 27):
                print("\n[INFO] Saliendo...")
                break

        cv2.destroyAllWindows()
        self._resumen_final()

    # ── Construcción del frame de visualización ───────────────────────────────

    def _construir_frame(
        self,
        img:           np.ndarray,
        tipo:          str,
        ruta:          Path,
        indice:        int,
        total:         int,
    ) -> np.ndarray:
        """
        Construye la pantalla de clasificación:
          - Imagen del recorte centrada y escalada (ocupa la mayor parte)
          - Panel inferior con instrucciones y estado de clasificación
        """
        color_tipo = _COLOR_TIPO.get(tipo, (200, 200, 200))
        mapa_est   = ESTADOS[tipo]

        # ── Escalar imagen manteniendo aspecto ────────────────────────────────
        h_img, w_img = img.shape[:2]
        area_h = DISP_H - 200   # reservar 200px para panel inferior
        area_w = DISP_W

        escala  = min(area_w / max(w_img, 1), area_h / max(h_img, 1))
        nw, nh  = int(w_img * escala), int(h_img * escala)
        img_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

        # Canvas negro
        canvas = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

        # Centrar imagen
        ox = (area_w - nw) // 2
        oy = (area_h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = img_res

        # Borde de color según tipo
        cv2.rectangle(canvas, (ox - 3, oy - 3), (ox + nw + 3, oy + nh + 3),
                      color_tipo, 3)

        # ── Panel inferior ────────────────────────────────────────────────────
        panel_y = area_h + 5
        cv2.rectangle(canvas, (0, panel_y), (DISP_W, DISP_H), (20, 20, 20), -1)
        cv2.line(canvas, (0, panel_y), (DISP_W, panel_y), color_tipo, 2)

        # Progreso
        pct_txt = f"{indice + 1}/{total}  ({(indice + 1) / max(total, 1) * 100:.0f}%)"
        barra_w = int(DISP_W * (indice + 1) / max(total, 1))
        cv2.rectangle(canvas, (0, panel_y + 1), (barra_w, panel_y + 5),
                      color_tipo, -1)

        # Nombre de archivo
        cv2.putText(canvas, ruta.name, (10, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, pct_txt, (DISP_W - 160, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_tipo, 1, cv2.LINE_AA)

        # Tipo de máquina
        tipo_txt = f"► {tipo.upper()}"
        cv2.putText(canvas, tipo_txt, (10, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, color_tipo, 2, cv2.LINE_AA)

        # Teclas de estado — dos botones grandes
        btn_w   = (DISP_W - 40) // 2
        btn_h   = 55
        btn_y   = panel_y + 65
        for idx_btn, (tecla, estado) in enumerate(mapa_est.items()):
            bx1 = 10 + idx_btn * (btn_w + 10)
            bx2 = bx1 + btn_w
            color_btn = _COLOR_ESTADO.get(estado, (200, 200, 200))
            n_cls = self._conteo[tipo].get(estado, 0)

            # Fondo semitransparente del botón
            overlay = canvas.copy()
            cv2.rectangle(overlay, (bx1, btn_y), (bx2, btn_y + btn_h), color_btn, -1)
            cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)
            cv2.rectangle(canvas, (bx1, btn_y), (bx2, btn_y + btn_h), color_btn, 2)

            # Texto del botón
            lbl = f"[{tecla}]  {estado.upper().replace('_', ' ')}   ({n_cls})"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tx = bx1 + (btn_w - tw) // 2
            ty = btn_y + (btn_h + th) // 2 - 2
            cv2.putText(canvas, lbl, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_btn, 2, cv2.LINE_AA)

        # Controles auxiliares en la esquina inferior derecha
        aux = "[D] Descartar  [S] Saltar  [←/→] Navegar  [Z] Deshacer  [Q] Salir"
        (tw, _), _ = cv2.getTextSize(aux, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, aux, (DISP_W - tw - 10, DISP_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

        # Contadores globales esquina inferior izquierda
        total_cl = sum(sum(v.values()) for v in self._conteo.values())
        info = f"Clasificados: {total_cl}  Descartados: {self._descartados}  Saltados: {self._saltados}"
        cv2.putText(canvas, info, (10, DISP_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 200, 140), 1, cv2.LINE_AA)

        return canvas

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cargar_pendientes(self) -> list[Path]:
        """Busca todos los .jpg/.png en las subcarpetas de tipo del origen."""
        archivos: list[Path] = []
        for tipo in self.tipos:
            carpeta = self.origen_dir / tipo
            if carpeta.exists():
                archivos += sorted(carpeta.glob("*.jpg"))
                archivos += sorted(carpeta.glob("*.png"))
        print(f"[INFO] Recortes encontrados: {len(archivos)}")
        return archivos

    def _mover(self, origen: Path, destino: Path) -> None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(origen), str(destino))

    def _deshacer(self, tipo_actual: str) -> None:
        """Revierte la última clasificación moviendo el archivo de vuelta."""
        if not self._historial:
            print("  [Z] Nada que deshacer.")
            return
        destino, origen_orig = self._historial.pop()
        if destino.exists():
            self._mover(destino, origen_orig)
            # Restar del conteo si corresponde
            estado = destino.parent.name
            tipo   = destino.parent.parent.name
            if tipo in self._conteo and estado in self._conteo[tipo]:
                self._conteo[tipo][estado] = max(0, self._conteo[tipo][estado] - 1)
            elif estado == "_descartado":
                self._descartados = max(0, self._descartados - 1)
            # Retroceder el índice para que vuelva a aparecer
            self._indice = max(0, self._indice - 1)
            # Recargar pendientes para que aparezca
            self._pendientes = self._cargar_pendientes()
            print(f"  [Z] Deshecho: {destino.name} → {origen_orig.parent.name}/")
        else:
            print(f"  [Z] Archivo no encontrado: {destino}")

    def _crear_carpetas(self) -> None:
        for tipo, mapa in ESTADOS.items():
            if tipo in self.tipos:
                for estado in mapa.values():
                    (self.destino_dir / tipo / estado).mkdir(parents=True, exist_ok=True)
        (self.destino_dir / "_descartado").mkdir(parents=True, exist_ok=True)

    def _resumen_final(self) -> None:
        total_cl = sum(sum(v.values()) for v in self._conteo.values())
        print(f"\n{'='*62}")
        print(f"  SESIÓN DE CLASIFICACIÓN COMPLETADA")
        print(f"{'='*62}")
        for tipo, conteo in self._conteo.items():
            if tipo not in self.tipos:
                continue
            total_tipo = sum(conteo.values())
            if total_tipo == 0:
                continue
            print(f"\n  {tipo.upper()}")
            for estado, n in conteo.items():
                barra = "█" * min(n // max(total_tipo // 20, 1), 30)
                dest  = self.destino_dir / tipo / estado
                print(f"    {estado:<30} {n:5d}  {barra}")
                print(f"    {'→':>5} {dest}")
        print(f"\n  Total clasificados : {total_cl}")
        print(f"  Descartados        : {self._descartados}")
        print(f"  Saltados           : {self._saltados}")
        pendientes_restantes = len([
            p for p in self._pendientes
            if p.exists()
        ])
        if pendientes_restantes > 0:
            print(f"  Pendientes aún     : {pendientes_restantes}")
            print(f"  (Vuelve a correr el script para continuar)")
        print(f"\n  SIGUIENTE PASO:")
        print(f"    python entrenar_clasificador.py")
        print(f"{'='*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Paso 2 — Clasifica recortes con teclas de teclado"
    )
    p.add_argument(
        "--origen",
        default="data/muestras_sin_clasificar",
        help="Carpeta con recortes sin clasificar (default: data/muestras_sin_clasificar)",
    )
    p.add_argument(
        "--destino",
        default="data/estados",
        help="Carpeta raíz de destino por estados (default: data/estados)",
    )
    p.add_argument(
        "--tipo",
        default="todos",
        choices=["camion", "excavadora", "pala", "todos"],
        help="Tipo de máquina a clasificar (default: todos)",
    )
    args = p.parse_args()

    origen  = Path(args.origen)
    destino = Path(args.destino)
    tipos   = list(ESTADOS.keys()) if args.tipo == "todos" else [args.tipo]

    if not origen.exists():
        print(f"[ERROR] Carpeta de origen no encontrada: {origen.resolve()}")
        print(f"        Ejecuta primero: python extraer_muestras.py")
        sys.exit(1)

    Organizador(
        origen_dir  = origen,
        destino_dir = destino,
        tipos       = tipos,
    ).ejecutar()


if __name__ == "__main__":
    main()
