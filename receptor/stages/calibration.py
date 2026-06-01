# ─────────────────────────────────────────────────────────────
#  receptor/stages/calibration.py  —  ETAPA 2: Calibración + AGC  [B2]
# ─────────────────────────────────────────────────────────────
#
#  Compensa las diferencias de brillo/contraste/offset introducidas por la
#  iluminación ambiental, el auto-exposure de la cámara y la respuesta de la
#  pantalla.  Modelo del canal:
#
#        y_rx(x,y) = a(x,y) · y_tx + b(x,y) + ruido
#
#  Los PILOTOS conocidos (0↔255 alternados) permiten invertirlo:
#        y_corr = (y_rx − b) / a   →  clip [0,255]
#
#  Dos modos (elegidos automáticamente según nº de pilotos disponibles):
#    • "2d"     : a(x,y) y b(x,y) son PLANOS ajustados por mínimos cuadrados
#                 sobre las posiciones de los pilotos → compensa gradientes
#                 ESPACIALES de iluminación (vignette, luz lateral).
#    • "global" : a, b escalares (fallback si hay pocos pilotos de algún nivel).
#
#  AGC + decisor de umbral adaptativo: tras calibrar, los niveles vuelven a
#  ~{0,255}; el umbral de decisión queda en el punto medio (≈128) y se expone
#  en `ctx.calib["thr"]` para que B4 lo use.
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np

from ..layout import compute_frame_layout, generate_pilot_values, sample_cells, cell_centers_px
from ..debug_viz import to_bgr, hstack_panels, banner
from ..pipeline import PipelineStage, PipelineContext


def _fit_plane(xy: np.ndarray, z: np.ndarray, n_iter: int = 2) -> np.ndarray:
    """
    Ajusta z ≈ c0 + c1·x + c2·y por LS ROBUSTO.  Tras un ajuste inicial,
    descarta los pilotos cuyo residual supera 3·MAD (típicamente celdas mal
    alineadas que muestrean a un vecino brillante) y reajusta.  Returns [c0,c1,c2].
    """
    A = np.column_stack([np.ones(len(xy)), xy[:, 0], xy[:, 1]])
    mask = np.ones(len(z), dtype=bool)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    for _ in range(n_iter):
        resid = z - A @ coef
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
        new_mask = np.abs(resid - np.median(resid)) <= 3.0 * 1.4826 * mad
        if new_mask.sum() < 4 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        coef, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
    return coef


def _eval_plane(coef: np.ndarray, W: int, H: int) -> np.ndarray:
    """Evalúa el plano c0+c1·x+c2·y sobre una grilla H×W."""
    ys, xs = np.mgrid[0:H, 0:W]
    return coef[0] + coef[1] * xs + coef[2] * ys


def _build_levelize_lut(received: np.ndarray, expected: np.ndarray) -> tuple:
    """
    Construye una LUT 0..255 que INVIERTE la respuesta no lineal del canal
    (gamma) usando los pilotos multi-nivel.  Para cada nivel TX conocido L se
    toma la MEDIANA de los pilotos recibidos a ese nivel → pares (recibido→L).
    Interpolación lineal a trozos entre esos puntos → un valor recibido se mapea
    al nivel TX que le corresponde, dejando los niveles equiespaciados otra vez.

    Returns (lut[256], puntos_recibidos, niveles_tx).
    """
    levels = np.unique(expected)
    R = np.array([np.median(received[expected == L]) for L in levels], dtype=float)
    L = levels.astype(float)
    # Forzar monotonía estricta de R (np.interp exige xp creciente)
    order = np.argsort(R)
    R, L = R[order], L[order]
    R = np.maximum.accumulate(R) + np.arange(len(R)) * 1e-3
    lut = np.interp(np.arange(256), R, L).clip(0, 255).astype(np.uint8)
    return lut, R, L


class CalibrationStage(PipelineStage):
    name = "Calibracion"
    required = True

    def __init__(self, min_pilots_per_level: int = 6, verbose: bool = True,
                 force_mode: str = None, warn_residual: float = 40.0):
        """force_mode: None=auto, "global"=escalar a,b, "2d"=mapa espacial.
        El modo "2d" es el que compensa gradientes/rolling shutter (ablación B5).
        warn_residual: umbral SOLO de advertencia (no detiene el pipeline)."""
        self.min_pilots_per_level = min_pilots_per_level
        self.verbose = verbose
        self.force_mode = force_mode
        self.warn_residual = warn_residual

    def run(self, ctx: PipelineContext) -> bool:
        if ctx.warped is None:
            return False
        cfg = ctx.config
        img = ctx.warped if ctx.warped.ndim == 2 else cv2.cvtColor(ctx.warped, cv2.COLOR_BGR2GRAY)
        W, H = cfg.frame_width, cfg.frame_height

        # 1 · Pilotos: posiciones, valores esperados (según esquema) y medidos
        layout = compute_frame_layout(cfg)
        pilots = layout["pilot"]
        expected = generate_pilot_values(len(pilots), cfg.scheme).astype(float)
        received, _ = sample_cells(img, pilots, cfg.cell_size)
        centers = cell_centers_px(pilots, cfg.cell_size)   # (x,y) por piloto

        # 4ASK: calibración NO LINEAL por LUT (invierte la gamma con los pilotos
        # multi-nivel) → los 4 niveles vuelven a 0/85/170/255 equiespaciados.
        if cfg.scheme == "4ASK":
            return self._run_4ask(ctx, cfg, img, pilots, expected, received)

        blk = expected < 128     # pilotos negros (TX=0)
        wht = ~blk               # pilotos blancos (TX=255)

        # 2 · Calibración global (siempre, como referencia/fallback)
        A = np.column_stack([expected, np.ones(len(expected))])
        (a_g, b_g), *_ = np.linalg.lstsq(A, received, rcond=None)

        # 3 · Mapa 2D si hay suficientes pilotos de cada nivel (o forzado)
        enough = (blk.sum() >= self.min_pilots_per_level and
                  wht.sum() >= self.min_pilots_per_level)
        if self.force_mode == "global":
            use_2d = False
        elif self.force_mode == "2d":
            use_2d = enough
        else:
            use_2d = enough
        if use_2d:
            cb = _fit_plane(centers[blk], received[blk])     # b(x,y) ← negros
            cw = _fit_plane(centers[wht], received[wht])     # blanco(x,y) ← blancos
            b_map = _eval_plane(cb, W, H)
            w_map = _eval_plane(cw, W, H)
            a_map = (w_map - b_map) / 255.0
            a_map = np.where(np.abs(a_map) < 1e-3, np.sign(a_map) * 1e-3 + 1e-9, a_map)
            calibrated = np.clip((img.astype(float) - b_map) / a_map, 0, 255)
            mode = "2d"
        else:
            a_map = np.full((H, W), a_g)
            b_map = np.full((H, W), b_g)
            calibrated = np.clip((img.astype(float) - b_g) / (a_g if abs(a_g) > 1e-9 else 1.0),
                                 0, 255)
            mode = "global"

        calibrated = calibrated.astype(np.uint8)

        # 4 · Residual: error de los pilotos tras calibrar (calidad de la calibración)
        cal_pilots, _ = sample_cells(calibrated, pilots, cfg.cell_size)
        residual = float(np.sqrt(np.mean((cal_pilots - expected) ** 2)))

        # 5 · AGC + umbral de decisión (punto medio de los niveles calibrados)
        thr = 128.0   # tras calibrar, {0,255} → decisor en el centro

        ctx.calibrated = calibrated
        ctx.calib = {"mode": mode, "a": float(a_g), "b": float(b_g),
                     "a_map": a_map, "b_map": b_map, "thr": thr,
                     "residual": residual,
                     "n_black": int(blk.sum()), "n_white": int(wht.sum())}
        ctx.metrics["calib_mode"] = mode
        ctx.metrics["calib_residual"] = round(residual, 2)
        ctx.metrics["calib_a"] = round(float(a_g), 4)
        ctx.metrics["calib_b"] = round(float(b_g), 2)

        if self.verbose:
            print(f"  Modo               : {mode}  "
                  f"(pilotos {int(blk.sum())}● / {int(wht.sum())}○)")
            print(f"  Ganancia/offset    : a={a_g:.4f}  b={b_g:.2f}")
            print(f"  Residual pilotos   : {residual:.2f}  (RMS, calibrado vs 0/255)")
            if residual >= self.warn_residual:
                print(f"  ⚠ residual alto (≥{self.warn_residual:.0f}): posible "
                      f"desalineación sub-celda de la grilla, blur/reflejos o "
                      f"contraste bajo. Se continúa (mejor esfuerzo); B3/B4 dirán "
                      f"si es decodificable.")

        # La calibración es de MEJOR ESFUERZO: basta con que sea invertible.
        # El residual alto se reporta como advertencia, pero NO detiene el
        # pipeline — los jueces reales de calidad son la sincronización (B3) y
        # el demapeo+ECC (B4).  Un residual alto suele ser desalineación de la
        # grilla, no un fallo de la calibración en sí.
        return abs(a_g) > 1e-3

    def _run_4ask(self, ctx, cfg, img, pilots, expected, received) -> bool:
        """Calibración 4ASK: LUT no lineal desde pilotos de 4 niveles (gamma)."""
        lut, R, L = _build_levelize_lut(received, expected)
        calibrated = lut[img.astype(np.uint8)]

        cal_pilots, _ = sample_cells(calibrated, pilots, cfg.cell_size)
        residual = float(np.sqrt(np.mean((cal_pilots - expected) ** 2)))

        ctx.calibrated = calibrated
        ctx.calib = {"mode": "4ask-lut", "thr": 128.0, "residual": residual,
                     "lut": lut, "recv_levels": R, "tx_levels": L,
                     "a": 1.0, "b": 0.0,
                     "a_map": np.ones(img.shape), "b_map": np.zeros(img.shape)}
        ctx.metrics["calib_mode"] = "4ask-lut"
        ctx.metrics["calib_residual"] = round(residual, 2)
        ctx.metrics["calib_levels_rx"] = [round(float(x), 1) for x in R]

        if self.verbose:
            print(f"  Modo               : 4ask-lut  (pilotos {len(received)}, 4 niveles)")
            print(f"  Niveles RX→TX      : "
                  f"{[round(float(x),1) for x in R]} → {[int(x) for x in L]}")
            print(f"  Residual pilotos   : {residual:.2f}  (RMS vs 0/85/170/255)")
            if residual >= self.warn_residual:
                print(f"  ⚠ residual alto: la gamma comprime los niveles oscuros; "
                      f"acerca la cámara / mejora la luz / sube --nsym.")
        return True

    def draw_debug(self, ctx: PipelineContext):
        cfg = ctx.config
        cal = ctx.calib
        layout = compute_frame_layout(cfg)
        pilots = layout["pilot"]
        expected = generate_pilot_values(len(pilots), cfg.scheme)
        centers = cell_centers_px(pilots, cfg.cell_size).astype(int)

        # Panel 1: warped con pilotos marcados (negros=azul, blancos=rojo)
        p1 = to_bgr(ctx.warped)
        for (x, y), e in zip(centers, expected):
            clr = (255, 120, 0) if e < 128 else (0, 0, 255)
            cv2.circle(p1, (x, y), 3, clr, -1, cv2.LINE_AA)
        p1 = banner(p1, "1. Rectificado + pilotos (azul=0, rojo=255)")

        # Panel 2: imagen calibrada
        p2 = banner(to_bgr(ctx.calibrated),
                    f"2. Calibrado [{cal['mode']}]  a={cal['a']:.3f} b={cal['b']:.1f}"
                    f"  resid={cal['residual']:.1f}")

        # Panel 3: mapa de offset b(x,y) como heatmap (gradiente de iluminación)
        bm = cal["b_map"]
        bm_norm = cv2.normalize(bm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p3 = cv2.applyColorMap(bm_norm, cv2.COLORMAP_JET)
        p3 = banner(p3, f"3. Mapa offset b(x,y)  [{bm.min():.0f}..{bm.max():.0f}]")

        return hstack_panels([p1, p2, p3])
