# ─────────────────────────────────────────────────────────────
#  receptor/stages/sync.py  —  ETAPA 3: Sincronización  [B3]
# ─────────────────────────────────────────────────────────────
#
#  Tres sub-tareas (instrucciones Fase B):
#
#  1. SINCRONIZACIÓN DE TRAMA (preámbulo Gold) — núcleo de esta etapa, opera
#     sobre un cuadro estático.  Demodula las celdas del preámbulo y correlaciona
#     contra la secuencia Gold esperada:
#        • pico ≥ 0.7  → trama identificada (la grilla recuperada es correcta).
#        • lag = 0     → sin permutación espacial residual tras la homografía.
#     Es la prueba definitiva de que B1+B2 entregaron una grilla decodificable.
#
#  2. ROLLING SHUTTER — efecto ESPACIAL (filas a distinta altura se exponen en
#     instantes distintos).  Se caracteriza midiendo la deriva de brillo por
#     fila que queda tras la calibración (debe ser ≈0 si B2 la compensó).  Se
#     reporta `rs_slope` como diagnóstico; la compensación fina ya la absorbe
#     el mapa 2D b(x,y) de B2.
#
#  3. RECUPERADOR DE RELOJ DE SÍMBOLO (early-late gate) — es TEMPORAL: necesita
#     una serie de cuadros.  `early_late_gate()` queda implementada y lista para
#     el modo vídeo; en modo cuadro único no se invoca (no hay eje temporal).
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np

from ..layout import compute_frame_layout, sample_cells
from ..preamble import build_preamble, verify_preamble
from ..debug_viz import to_bgr, hstack_panels, banner, plot1d
from ..pipeline import PipelineStage, PipelineContext


def estimate_rolling_shutter(image: np.ndarray, config) -> dict:
    """
    Caracteriza el rolling shutter como la deriva de brillo MEDIO por fila de
    celdas.  Ajusta una recta brillo(fila) y devuelve su pendiente (niveles/fila)
    y la serie de medias por fila (para visualización).
    """
    cs, M, N = config.cell_size, config.M, config.N
    row_means = np.array([
        image[r * cs:(r + 1) * cs, :].mean() for r in range(M)], dtype=float)
    rows = np.arange(M, dtype=float)
    A = np.column_stack([rows, np.ones(M)])
    (slope, intercept), *_ = np.linalg.lstsq(A, row_means, rcond=None)
    return {"slope": float(slope), "intercept": float(intercept),
            "row_means": row_means}


def early_late_gate(intensity_ts: np.ndarray, sps: int,
                    mu: float = 0.0, gain: float = 0.05) -> dict:
    """
    Recuperador de reloj de símbolo (early-late gate) para MODO VÍDEO.

    intensity_ts : intensidad interpolada de una celda a lo largo del tiempo.
    sps          : muestras por símbolo (camera_fps · symbol_duration).
    Returns      : {"offset": fase de muestreo óptima, "samples": índices}.

    En modo cuadro único no se usa (no hay eje temporal); se incluye para la
    transición a captura de vídeo.
    """
    n = len(intensity_ts)
    offset = mu
    samples = []
    k = sps / 2.0 + offset
    while k < n - 1:
        idx = int(round(k))
        early = intensity_ts[max(0, idx - 1)]
        late = intensity_ts[min(n - 1, idx + 1)]
        err = (late - early) * np.sign(intensity_ts[idx] - 0.5)
        offset += gain * err
        samples.append(idx)
        k += sps + gain * err
    return {"offset": float(offset), "samples": np.array(samples, int)}


class SyncStage(PipelineStage):
    name = "Sincronizacion"
    required = True

    def __init__(self, peak_thr: float = 0.7, verbose: bool = True):
        self.peak_thr = peak_thr
        self.verbose = verbose

    def run(self, ctx: PipelineContext) -> bool:
        img = ctx.calibrated if ctx.calibrated is not None else ctx.warped
        if img is None:
            return False
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cfg = ctx.config

        # — Sincronización de trama por preámbulo Gold —
        layout = compute_frame_layout(cfg)
        preamble = build_preamble(cfg, layout)
        thr = int(ctx.calib["thr"]) if ctx.calib else 128
        pre_means, _ = sample_cells(img, preamble["preamble_cells"], cfg.cell_size)
        pre_u8 = np.clip(np.round(pre_means), 0, 255).astype(np.uint8)
        check = verify_preamble(pre_u8, preamble, cfg,
                                threshold=self.peak_thr, decision_thr=thr)

        # — Rolling shutter (diagnóstico espacial) —
        rs = estimate_rolling_shutter(img, cfg)

        ctx.sync = {**check, "preamble": preamble,
                    "rs_slope": rs["slope"], "row_means": rs["row_means"],
                    "clock": "N/A (cuadro único)"}
        ctx.metrics["sync_peak"] = round(check["peak"], 3)
        ctx.metrics["sync_lag"] = check["lag"]
        ctx.metrics["rs_slope_lvl_per_row"] = round(rs["slope"], 3)

        if self.verbose:
            print(f"  Preámbulo Gold     : pico={check['peak']:.3f}  "
                  f"lag={check['lag']}  "
                  f"{'✓ SYNC' if check['synced'] else '✗ NO SYNC'}")
            print(f"  Rolling shutter    : {rs['slope']:+.3f} niveles/fila "
                  f"(residual tras calibración)")
            print(f"  Reloj de símbolo   : N/A (cuadro único; activo en vídeo)")

        return bool(check["synced"])

    def draw_debug(self, ctx: PipelineContext):
        cfg = ctx.config
        s = ctx.sync
        pre = s["preamble"]

        # Panel 1: símbolos del preámbulo esperado vs recibido (tiras)
        exp_strip = pre["symbols"].reshape(1, -1).astype(np.uint8)
        img = ctx.calibrated if ctx.calibrated is not None else ctx.warped
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rec_means, _ = sample_cells(img, pre["preamble_cells"], cfg.cell_size)
        rec_strip = np.clip(np.round(rec_means), 0, 255).reshape(1, -1).astype(np.uint8)
        strips = np.vstack([np.repeat(exp_strip, 28, axis=0),
                            np.full((6, exp_strip.shape[1]), 60, np.uint8),
                            np.repeat(rec_strip, 28, axis=0)])
        strips = cv2.resize(to_bgr(strips), (460, 320), interpolation=cv2.INTER_NEAREST)
        p1 = banner(strips, "1. Preambulo: esperado (arriba) vs recibido (abajo)")

        # Panel 2: correlación cruzada circular (pico marcado)
        xc = s["xc"]
        L = len(xc)
        xc_c = np.roll(xc, L // 2)
        clr = (80, 230, 80) if s["synced"] else (80, 80, 230)
        p2 = plot1d([xc_c], [clr], hlines=[(self.peak_thr, (0, 180, 230)),
                                           (-self.peak_thr, (0, 180, 230))],
                    ymin=-1.05, ymax=1.05)
        p2 = banner(p2, f"2. Correlacion Gold  pico={s['peak']:.2f} lag={s['lag']}"
                        f"  {'SYNC' if s['synced'] else 'NO SYNC'}")

        # Panel 3: brillo medio por fila (diagnóstico rolling shutter)
        rm = s["row_means"]
        rows = np.arange(len(rm))
        fit = s["rs_slope"] * rows + (rm.mean() - s["rs_slope"] * rows.mean())
        p3 = plot1d([rm, fit], [(230, 200, 80), (80, 80, 230)])
        p3 = banner(p3, f"3. Brillo/fila (rolling shutter)  m={s['rs_slope']:+.2f}/fila")

        return hstack_panels([p1, p2, p3])
