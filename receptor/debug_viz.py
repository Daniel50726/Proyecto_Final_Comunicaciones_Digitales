# ─────────────────────────────────────────────────────────────
#  receptor/debug_viz.py  —  Depuración visual del pipeline
# ─────────────────────────────────────────────────────────────
#
#  En un entorno .py (a diferencia del notebook) no hay salida inline de
#  matplotlib.  Esta clase centraliza la depuración visual con tres modos:
#
#    "window" : cv2.imshow + waitKey   → inspección interactiva en escritorio
#    "save"   : escribe PNG en debug_out/ → headless / CI / tiempo real
#    "none"   : desactiva todo (modo producción / benchmark)
#
#  Cada etapa del pipeline produce UNA imagen BGR compuesta; esta clase decide
#  qué hacer con ella.  Así el código de cada etapa no se acopla al modo de
#  visualización elegido y la transición a tiempo real es trivial.
# ─────────────────────────────────────────────────────────────
import os
import time

import cv2
import numpy as np


class DebugViz:
    def __init__(self,
                 mode: str = "save",
                 out_dir: str = None,
                 wait_ms: int = 0,
                 max_panel_h: int = 520):
        """
        mode        : "window" | "save" | "none"
        out_dir     : carpeta destino en modo "save"
        wait_ms     : ms de espera en cv2.waitKey (0 = bloquea hasta tecla)
        max_panel_h : alto máximo de cada panel mostrado (escala para caber)
        """
        assert mode in ("window", "save", "none")
        self.mode = mode
        self.wait_ms = wait_ms
        self.max_panel_h = max_panel_h
        self.out_dir = out_dir or os.path.join(os.path.dirname(__file__), "debug_out")
        if mode == "save":
            os.makedirs(self.out_dir, exist_ok=True)
        self._counter = 0

    # ── API pública ───────────────────────────────────────────
    def show(self, tag: str, image: np.ndarray) -> None:
        """Muestra/guarda una imagen de depuración etiquetada `tag`."""
        if self.mode == "none" or image is None:
            return

        disp = self._fit(image)
        if self.mode == "window":
            cv2.imshow(tag, disp)
            cv2.waitKey(self.wait_ms)
        else:  # save
            self._counter += 1
            fname = f"{self._counter:02d}_{_safe(tag)}.png"
            path = os.path.join(self.out_dir, fname)
            cv2.imwrite(path, disp)
            print(f"  [debug] guardado → {os.path.relpath(path)}")

    def close(self) -> None:
        if self.mode == "window":
            cv2.destroyAllWindows()

    # ── Utilidades de dibujo reutilizables por las etapas ─────
    def _fit(self, image: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        if h <= self.max_panel_h:
            return image
        scale = self.max_panel_h / h
        return cv2.resize(image, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)


# ── Helpers de dibujo (BGR) usados por varias etapas ──────────

def to_bgr(img: np.ndarray) -> np.ndarray:
    """Garantiza 3 canales BGR a partir de gris o BGR."""
    if img is None:
        return None
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def hstack_panels(panels: list, sep: int = 6, bg: int = 30) -> np.ndarray:
    """Apila paneles BGR horizontalmente a una altura común, con separadores."""
    panels = [p for p in panels if p is not None]
    if not panels:
        return None
    h = max(p.shape[0] for p in panels)
    norm = []
    for p in panels:
        if p.shape[0] != h:
            scale = h / p.shape[0]
            p = cv2.resize(p, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
        norm.append(p)
    spacer = np.full((h, sep, 3), bg, np.uint8)
    out = norm[0]
    for p in norm[1:]:
        out = np.hstack([out, spacer, p])
    return out


def banner(image: np.ndarray, text: str,
           color=(255, 255, 255), bg=(40, 40, 40)) -> np.ndarray:
    """Añade una barra de título arriba de la imagen."""
    bar_h = 30
    w = image.shape[1]
    bar = np.full((bar_h, w, 3), bg, np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def _safe(tag: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in tag).strip("_").lower()
