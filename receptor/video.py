# ─────────────────────────────────────────────────────────────
#  receptor/video.py  —  Modo vídeo: recuperación de reloj de símbolo
# ─────────────────────────────────────────────────────────────
#
#  En modo cuadro único cada captura es un mensaje completo.  En modo VÍDEO la
#  pantalla muestra una SECUENCIA de tramas (un mensaje por periodo de símbolo)
#  y la cámara captura a OTRA tasa (camera_fps ≠ screen_fps).  Aparecen dos
#  problemas temporales que esta etapa resuelve:
#
#    • TEARING por rolling shutter: una captura cuyo tiempo de exposición cruza
#      una transición de pantalla mezcla la trama vieja (filas de arriba) con la
#      nueva (filas de abajo) → ilegible.
#    • RECUPERACIÓN DE RELOJ DE SÍMBOLO: hay que muestrear en el CENTRO de cada
#      periodo (captura estable, trama fresca), no en los bordes.  Se usa un
#      detector early-late sobre la señal de CAMBIO entre cuadros consecutivos:
#      el cambio es alto en las transiciones (bordes del símbolo) y mínimo en el
#      centro → se bloquea el muestreo a esos mínimos.
#
#  `VideoReceiver` es online (un cuadro a la vez) → sirve igual para la cámara
#  en vivo y para el stream simulado (test sin hardware).
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np

from .config import ModemConfig
from .channel_coding import ECCConfig
from .debug_viz import DebugViz
from .pipeline import ReceiverPipeline
from .simulate import simulate_capture
from .frame_builder import assemble_frame
from .stages import ROIStage, CalibrationStage, SyncStage, DemapStage


# ── Simulación de un stream de vídeo (test sin cámara) ────────
def simulate_video_stream(messages, config: ModemConfig, ecc: ECCConfig = None,
                          spc: float = 3.0, angle: float = 12.0,
                          noise: float = 8.0, brightness: float = 0.85,
                          tear: bool = True):
    """
    Genera la lista de cuadros que capturaría la cámara mostrando `messages` en
    pantalla, uno por periodo de símbolo.

    spc  : cuadros capturados por símbolo (= camera_fps · symbol_ms/1000).
    tear : si True, modela el desgarro por rolling shutter en las transiciones.
    """
    ecc = ecc or ECCConfig()
    tx = [assemble_frame(m, config, ecc)["frame"] for m in messages]
    H = config.frame_height
    n_slots = int(round(len(messages) * spc))
    frames = []
    for t in range(n_slots):
        s0, s1 = t / spc, (t + 1) / spc        # ventana de exposición (en símbolos)
        idx = int(s0)
        if idx >= len(tx):
            break
        base = tx[idx]
        b = int(np.floor(s1))
        if tear and b != idx and b < len(tx):  # la exposición cruza una transición
            frac = (b - s0) / (s1 - s0)         # parte expuesta antes del cambio
            split = int(np.clip(H * frac, 0, H))
            torn = tx[b].copy()
            torn[:split] = base[:split]         # arriba: trama vieja (rolling shutter)
            base = torn
        frames.append(simulate_capture(base, angle_deg=angle, noise_std=noise,
                                       brightness=brightness))
    return frames


# ── Recuperación de reloj de símbolo (early-late sobre el cambio) ─
class SymbolClock:
    """
    Detecta los instantes de muestreo óptimos en un stream.  Mantiene la señal
    de cambio d[t]=media|f_t − f_{t-1}| y, mediante un detector early-late,
    bloquea el muestreo en los MÍNIMOS de cambio (centro del símbolo), separados
    por al menos `min_gap` cuadros.  Las transiciones (picos de d) segmentan los
    símbolos.
    """

    def __init__(self, spc: float):
        self.spc = float(spc)
        self.min_gap = max(1, int(round(0.6 * spc)))
        self.prev = None
        self.d_hist = []
        self.best_idx = None      # índice del cuadro más estable del símbolo actual
        self.best_d = np.inf
        self.last_emit = -10 ** 9
        self.period_est = float(spc)
        self._trans = []          # tiempos de transición (para estimar periodo)

    def update(self, t: int, frame_small: np.ndarray):
        """Devuelve (d, sample_idx_o_None).  sample_idx = cuadro a decodificar."""
        if self.prev is None:
            self.prev = frame_small
            self.d_hist.append(0.0)
            self.best_idx, self.best_d = t, 0.0
            return 0.0, None
        d = float(np.mean(np.abs(frame_small.astype(np.int16) - self.prev.astype(np.int16))))
        self.prev = frame_small
        self.d_hist.append(d)

        # Umbral de transición adaptativo sobre una ventana reciente
        win = self.d_hist[-max(3, int(3 * self.spc)):]
        thr = max(4.0, 0.5 * max(win))

        sample = None
        if d > thr and (t - self.last_emit) >= self.min_gap:
            # transición → cerró el símbolo: el mejor (más estable) se decodifica
            if self.best_idx is not None:
                sample = self.best_idx
                self.last_emit = t
                self._trans.append(t)
                if len(self._trans) >= 2:
                    self.period_est = float(np.median(np.diff(self._trans)))
            self.best_idx, self.best_d = None, np.inf
        else:
            # dentro del símbolo: rastrear el cuadro de menor cambio (early-late)
            if d < self.best_d:
                self.best_d, self.best_idx = d, t
        return d, sample

    def flush(self):
        """Último símbolo pendiente al terminar el stream."""
        idx = self.best_idx
        self.best_idx = None
        return idx


# ── Receptor de vídeo (online) ────────────────────────────────
class VideoReceiver:
    """
    Procesa un stream cuadro a cuadro: recupera el reloj de símbolo, decodifica
    sólo los cuadros estables (centro de símbolo) con el pipeline B1–B4 y emite
    cada mensaje una vez (dedup de decodificaciones idénticas consecutivas).
    """

    def __init__(self, config: ModemConfig, ecc: ECCConfig = None,
                 spc: float = 3.0, downscale: int = 4):
        self.config = config
        self.ecc = ecc or ECCConfig()
        self.clock = SymbolClock(spc)
        self.downscale = downscale
        self.pipeline = ReceiverPipeline(
            config,
            [ROIStage(verbose=False), CalibrationStage(verbose=False),
             SyncStage(verbose=False), DemapStage(ecc=self.ecc, verbose=False)],
            viz=DebugViz(mode="none"))
        self.buffer = {}          # t → frame (para decodificar el cuadro elegido)
        self.t = -1
        self.last_text = None
        self.last_status = {}

    def _small(self, frame):
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (g.shape[1] // self.downscale, g.shape[0] // self.downscale))

    def push(self, frame: np.ndarray):
        """
        Inyecta un cuadro.  Devuelve un dict de evento cuando se decodifica un
        símbolo (o None).  Mantiene un buffer corto para poder decodificar el
        cuadro estable señalado por el reloj.
        """
        self.t += 1
        t = self.t
        self.buffer[t] = frame
        # limitar buffer
        for old in [k for k in self.buffer if k < t - int(4 * self.clock.spc) - 2]:
            self.buffer.pop(old, None)

        d, sample_idx = self.clock.update(t, self._small(frame))
        self.last_status = {"t": t, "change": round(d, 1),
                            "period_est": round(self.clock.period_est, 2)}
        if sample_idx is None or sample_idx not in self.buffer:
            return None
        return self._decode(self.buffer[sample_idx], sample_idx)

    def _decode(self, frame, sample_idx):
        ctx = self.pipeline.process(frame, verbose=False)
        text = ctx.text
        ev = {"sample_idx": sample_idx, "text": text,
              "roi_ok": ctx.stage_ok.get("ROI", False),
              "sync_peak": ctx.metrics.get("sync_peak", 0.0),
              "ecc_failed": ctx.metrics.get("ecc_failed_blocks", -1),
              "repeat": text == self.last_text}
        self.last_text = text
        return ev

    def finish(self):
        idx = self.clock.flush()
        if idx is not None and idx in self.buffer:
            return self._decode(self.buffer[idx], idx)
        return None
