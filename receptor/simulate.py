# ─────────────────────────────────────────────────────────────
#  receptor/simulate.py  —  Simulación de captura (test sin cámara)
# ─────────────────────────────────────────────────────────────
#
#  Reproduce las degradaciones del canal óptico pantalla→cámara para validar
#  el receptor sin hardware:  perspectiva (keystone), defocus, auto-exposure y
#  ruido de sensor.  Portado verbatim de FaseA.ipynb (Fase B, Celda 1).
#
#  En tiempo real esta función se reemplaza por cv2.VideoCapture(...).read().
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np


def simulate_capture(frame: np.ndarray,
                     angle_deg: float = 15.0,
                     noise_std: float = 8.0,
                     brightness: float = 1.0,
                     blur_k: int = 3,
                     bg_color: int = 80,
                     ambient: float = 0.0,
                     gradient: float = 0.0) -> np.ndarray:
    """
    Simula la imagen que capturaría la cámara.

    angle_deg : ángulo de inclinación respecto a la frontal (0 = frontal).
    ambient   : nivel aditivo constante (luz ambiente) sumado a toda la imagen.
    gradient  : 0..1, atenúa la iluminación de izquierda a derecha (luz lateral).
                Reproduce un gradiente ESPACIAL que sólo la calibración 2D
                (mapa a(x,y)/b(x,y)) puede compensar.
    """
    gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3 else frame.copy())
    h, w = gray.shape

    # — Distorsión de perspectiva (keystone realista por escorzo) —
    if abs(angle_deg) > 0.5:
        angle_rad = np.radians(angle_deg)
        d = w * 0.5 * np.sin(angle_rad) * 0.7          # desplazamiento horizontal
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[d, 0], [w - d, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        cap = cv2.warpPerspective(gray, M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=bg_color)
    else:
        cap = gray.copy()

    # — Desenfoque (defocus) —
    if blur_k >= 3:
        cap = cv2.GaussianBlur(cap, (blur_k | 1, blur_k | 1), 0)

    # — Variación de brillo (auto-exposure) + gradiente espacial + ambiente —
    if brightness != 1.0 or gradient > 0.0 or ambient != 0.0:
        capf = cap.astype(float) * brightness
        if gradient > 0.0:
            ramp = np.linspace(1.0 - gradient, 1.0, cap.shape[1])[None, :]
            capf *= ramp
        capf += ambient
        cap = np.clip(capf, 0, 255).astype(np.uint8)

    # — Ruido gaussiano de sensor —
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, cap.shape)
        cap = np.clip(cap.astype(float) + noise, 0, 255).astype(np.uint8)

    return cap
