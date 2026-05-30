# ─────────────────────────────────────────────────────────────
#  receptor/stages/demap.py  —  ETAPA 4: Muestreo, demapeo + ECC  [B4]
# ─────────────────────────────────────────────────────────────
#
#  Cierra el pipeline:  imagen calibrada → texto.
#
#  1. MUESTREO ROBUSTO: por cada celda de datos se promedia el centro del
#     macropíxel tras un filtro de MEDIANA (suprime reflejos puntuales /
#     especulares del canal óptico) — `sample_cells_robust`.
#  2. DEMAPEO: símbolos → bits según el esquema, con el umbral adaptativo que
#     entregó la calibración (B2).
#  3. ECC (Reed-Solomon): corrige los errores residuales de byte (ráfagas).
#  4. bits → bytes → texto, quitando el padding 0x00 (rstrip).
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np

from ..layout import compute_frame_layout, sample_cells_robust
from ..preamble import build_preamble
from ..modulation import symbols_to_bits
from ..channel_coding import (ECCConfig, rs_decode_payload, bits_to_bytes)
from ..frame_builder import payload_capacity_bytes
from ..debug_viz import to_bgr, hstack_panels, banner, plot1d
from ..pipeline import PipelineStage, PipelineContext


class DemapStage(PipelineStage):
    name = "Demapeo"
    required = True

    def __init__(self, ecc: ECCConfig = None, verbose: bool = True):
        self.ecc = ecc or ECCConfig()
        self.verbose = verbose

    def run(self, ctx: PipelineContext) -> bool:
        img = ctx.calibrated if ctx.calibrated is not None else ctx.warped
        if img is None:
            return False
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cfg = ctx.config
        thr = int(ctx.calib["thr"]) if ctx.calib else 128

        layout = compute_frame_layout(cfg)
        preamble = build_preamble(cfg, layout)
        data_cells = preamble["data_positions"]

        # 1 · Muestreo robusto (media + mediana) de las celdas de payload
        payload_bytes = payload_capacity_bytes(cfg, len(data_cells))
        cells_per_byte = 16 if cfg.scheme == "BPSK_Manchester" else 4
        n_cells = payload_bytes * cells_per_byte
        values = sample_cells_robust(img, data_cells[:n_cells], cfg.cell_size)
        ctx.cell_values = values

        # 2 · Demapeo símbolos → bits → bytes
        syms = np.clip(np.round(values), 0, 255).astype(np.uint8)
        bits = symbols_to_bits(syms, cfg, thr)
        recv_bytes = bits_to_bytes(bits)[:payload_bytes]
        if len(recv_bytes) < payload_bytes:
            recv_bytes = np.pad(recv_bytes, (0, payload_bytes - len(recv_bytes)))
        ctx.symbols = syms
        ctx.bits = bits

        # 3 · ECC (Reed-Solomon)
        dec = rs_decode_payload(recv_bytes, payload_bytes, self.ecc)

        # 4 · bytes → texto
        text = dec["data"].rstrip(b"\x00").decode("utf-8", errors="replace")
        ctx.text = text
        ctx.metrics["ecc_scheme"] = self.ecc.scheme
        ctx.metrics["ecc_corrected"] = dec["n_corrected"]
        ctx.metrics["ecc_failed_blocks"] = dec["n_failed"]
        ctx.metrics["text"] = text

        # Métrica de calidad: nº de codewords irrecuperables
        ok = dec["n_failed"] == 0

        if self.verbose:
            print(f"  Muestreo           : {len(values)} celdas (media+mediana)")
            print(f"  ECC                : {self.ecc.scheme}  "
                  f"corregidos={dec['n_corrected']} bytes  "
                  f"bloques_fallidos={dec['n_failed']}")
            shown = text if len(text) <= 60 else text[:57] + "..."
            print(f"  Texto recuperado   : '{shown}'")

        return ok

    def draw_debug(self, ctx: PipelineContext):
        cfg = ctx.config
        img = ctx.calibrated if ctx.calibrated is not None else ctx.warped
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Panel 1: valores de celda muestreados (histograma bimodal esperado)
        vals = ctx.cell_values
        thr = int(ctx.calib["thr"]) if ctx.calib else 128
        hist = np.histogram(vals, bins=64, range=(0, 255))[0].astype(float)
        p1 = plot1d([hist], [(80, 220, 220)],
                    hlines=None, ymin=0)
        p1 = banner(p1, f"1. Histograma niveles muestreados (umbral={thr})")

        # Panel 2: tira de símbolos demodulados
        syms = ctx.symbols.reshape(1, -1).astype(np.uint8)
        strip = cv2.resize(to_bgr(np.repeat(syms, 40, axis=0)),
                           (460, 200), interpolation=cv2.INTER_NEAREST)
        p2 = banner(strip, f"2. Simbolos demodulados ({len(ctx.symbols)})")

        # Panel 3: texto recuperado sobre lienzo
        canvas = np.full((226, 460, 3), 25, np.uint8)
        txt = ctx.text or ""
        m = ctx.metrics
        cv2.putText(canvas, "TEXTO RECUPERADO:", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        for i in range(0, min(len(txt), 120), 24):
            cv2.putText(canvas, txt[i:i + 24], (12, 64 + (i // 24) * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"ECC {m.get('ecc_scheme')}: +{m.get('ecc_corrected',0)}B  "
                    f"fallos={m.get('ecc_failed_blocks',0)}",
                    (12, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 180, 230), 1, cv2.LINE_AA)
        p3 = banner(canvas, "3. Salida del receptor")

        return hstack_panels([p1, p2, p3])
