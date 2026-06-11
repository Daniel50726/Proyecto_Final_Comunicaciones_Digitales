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

# Backends por nombre (los no disponibles en la plataforma se ignoran al abrir)
BACKENDS = {
    "any": getattr(cv2, "CAP_ANY", 0),
    "dshow": getattr(cv2, "CAP_DSHOW", 700),
    "msmf": getattr(cv2, "CAP_MSMF", 1400),
    "v4l2": getattr(cv2, "CAP_V4L2", 200),
}


def _safe_read(cap):
    """read() que captura cv2.error (MSMF puede lanzar en cuadros corruptos)."""
    try:
        return cap.read()
    except cv2.error:
        return False, None


def open_camera(cam_id: int, preferred: str = "any",
                width: int = 1280, height: int = 720, verbose: bool = True):
    """
    Abre la cámara probando el backend preferido y, si falla, cae a los demás.

    IMPORTANTE: la resolución se fija ANTES del primer read().  Cambiarla después
    de haber leído un cuadro provoca el bug de MSMF `_step >= minstep` (stride
    inconsistente).  Si fijar la resolución rompe la captura, se reintenta SIN
    forzarla (resolución nativa; el pipeline rectifica a canónico igualmente).
    Devuelve (cap, nombre_backend) o (None, None).
    """
    order = [preferred] + [b for b in ("any", "msmf", "dshow") if b != preferred]
    for name in order:
        flag = BACKENDS.get(name, BACKENDS["any"])
        for try_res in (True, False):        # 1º con resolución fijada, 2º nativa
            try:
                cap = cv2.VideoCapture(cam_id, flag)
                if not cap.isOpened():
                    cap.release()
                    break                    # este backend no abre → siguiente
                if try_res and width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except cv2.error:
                    pass
                ok, _ = _safe_read(cap)
                if ok:
                    if verbose:
                        res = "fijada" if try_res else "nativa"
                        print(f"  Cámara abierta: backend '{name}', resolución {res}.")
                    return cap, name
                cap.release()
            except Exception as e:
                if verbose:
                    print(f"  backend '{name}' no usable ({e}); probando otro…")
                break
    return None, None


def configure_camera(cap: cv2.VideoCapture, exposure: float = -6.0,
                     verbose: bool = True) -> dict:
    """
    Intenta desactivar los automáticos (exposición/foco/WB).  NO toca resolución
    ni buffer (eso se fija en open_camera, antes del primer read, para evitar el
    bug MSMF `_step >= minstep`).  Mejor esfuerzo: cada set en try; si la cámara
    no soporta una propiedad, cap.get() devuelve -1 y simplemente se compensa por
    software en la calibración.
    """
    applied = {}

    def trySet(prop, val):
        try:
            cap.set(prop, val)
        except cv2.error:
            pass

    # — Exposición manual (convenios DSHOW=0.25 / V4L2=1) —
    for val in (0.25, 1):
        trySet(cv2.CAP_PROP_AUTO_EXPOSURE, val)
    trySet(cv2.CAP_PROP_EXPOSURE, exposure)
    # — Foco manual —
    trySet(cv2.CAP_PROP_AUTOFOCUS, 0)
    # — Balance de blancos manual —
    trySet(cv2.CAP_PROP_AUTO_WB, 0)
    trySet(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

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

    def __init__(self, cam_id: int = 0, backend: str = "any",
                 configure: bool = True, exposure: float = -6.0):
        self.cap, self.backend = open_camera(cam_id, preferred=backend)
        if self.cap is None:
            raise RuntimeError(
                f"No se pudo abrir la cámara {cam_id} con ningún backend. "
                f"¿Está conectada / la usa otra app / permisos de cámara?")
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
        errors = 0
        while self._running:
            try:
                ok, frame = self.cap.read()
            except cv2.error:
                ok, frame = False, None        # cuadro corrupto (MSMF) → saltar
            if not ok or frame is None:
                errors += 1
                if errors > 200:               # cámara desconectada/colgada
                    self._running = False
                    break
                time.sleep(0.005)
                continue
            errors = 0
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
