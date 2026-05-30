# ─────────────────────────────────────────────────────────────
#  receptor/camera.py  —  Interfaz de cámara para tiempo real
# ─────────────────────────────────────────────────────────────
#
#  Dos problemas del canal real que se atacan aquí:
#
#  1. LATENCIA / COLA: cv2.VideoCapture acumula cuadros en un buffer interno;
#     si el procesamiento es más lento que la cámara, se procesan cuadros viejos
#     y la latencia crece sin límite.  `CameraStream` lee en un HILO aparte y
#     guarda SOLO el cuadro más reciente → el bucle de proceso siempre toma el
#     más nuevo y nunca se satura.  (Se complementa con BUFFERSIZE=1.)
#
#  2. AJUSTES AUTOMÁTICOS: exposición/foco/balance de blancos automáticos
#     introducen variaciones lentas que rompen la calibración.  `configure_camera`
#     intenta fijarlos con cap.set(); la semántica varía por backend/cámara, así
#     que se aplica en modo "mejor esfuerzo" y se reporta qué quedó fijado.
# ─────────────────────────────────────────────────────────────
import threading
import time

import cv2


def configure_camera(cap: cv2.VideoCapture, width: int = 1280, height: int = 720,
                     exposure: float = -6.0, verbose: bool = True) -> dict:
    """
    Intenta desactivar los automáticos y fijar resolución/exposición.
    Devuelve un dict con lo que realmente quedó aplicado (mejor esfuerzo).

    Nota sobre AUTO_EXPOSURE: en backend DSHOW (Windows) 0.25≈manual, 0.75≈auto;
    en V4L2 (Linux) 1=manual, 3=auto.  Se prueban ambos convenios.
    """
    applied = {}
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # minimizar cola interna

    # — Exposición manual —
    for val in (0.25, 1):                      # DSHOW=manual, V4L2=manual
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    # — Foco manual —
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # — Balance de blancos manual —
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    try:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    except Exception:
        pass

    for name, prop in [("width", cv2.CAP_PROP_FRAME_WIDTH),
                       ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                       ("fps", cv2.CAP_PROP_FPS),
                       ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
                       ("exposure", cv2.CAP_PROP_EXPOSURE),
                       ("autofocus", cv2.CAP_PROP_AUTOFOCUS),
                       ("auto_wb", cv2.CAP_PROP_AUTO_WB),
                       ("buffersize", cv2.CAP_PROP_BUFFERSIZE)]:
        applied[name] = cap.get(prop)

    if verbose:
        print("── Cámara configurada (mejor esfuerzo) ─────────")
        for k, v in applied.items():
            print(f"  {k:14s}: {v}")
    return applied


class CameraStream:
    """
    Lector de cámara en hilo: mantiene SIEMPRE el último cuadro disponible.
    Evita la acumulación de latencia en la cola interna de OpenCV.

    Uso:
        cam = CameraStream(0).start()
        frame = cam.read()          # último cuadro (o None si aún no hay)
        cam.stop()
    """

    def __init__(self, cam_id: int = 0, backend: int = None,
                 configure: bool = True, exposure: float = -6.0):
        self.cap = (cv2.VideoCapture(cam_id, backend) if backend is not None
                    else cv2.VideoCapture(cam_id))
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {cam_id}")
        if configure:
            configure_camera(self.cap, exposure=exposure)
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # esperar al primer cuadro
        t0 = time.time()
        while self._frame is None and time.time() - t0 < 5.0:
            time.sleep(0.01)
        return self

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self):
        """Devuelve (seq, frame) del último cuadro; seq permite saber si es nuevo."""
        with self._lock:
            if self._frame is None:
                return None, None
            return self._seq, self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
