#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  benchmark_b5.py  —  ETAPA 5: Evaluación cuantitativa (BER)
# ─────────────────────────────────────────────────────────────
#
#  Demuestra cuantitativamente el requisito del checkpoint Fase B:
#  «la BER mejora al activar la corrección de perspectiva, la calibración de
#   color y la compensación de rolling shutter respecto al procesamiento crudo».
#
#  Ablación incremental (cada config añade una corrección sobre la anterior):
#    1. Crudo                  → sin perspectiva ni calibración (grilla nominal
#                                sobre la imagen capturada distorsionada).
#    2. + Perspectiva          → homografía (B1) → vista cenital, umbral fijo.
#    3. + Calib. global        → B1 + ganancia/offset ESCALAR (B2 modo global).
#    4. + Calib. 2D (rolling)  → B1 + mapa ESPACIAL a(x,y)/b(x,y): compensa
#                                gradientes e iluminación por fila (≈ rolling
#                                shutter residual).
#
#  Condiciones: ángulos {0°,15°,30°} × iluminación {uniforme, gradiente}.
#  La BER es de CANAL (pre-ECC): bits demodulados del payload vs bits
#  transmitidos conocidos.  Se promedia sobre varias realizaciones de ruido.
# ─────────────────────────────────────────────────────────────
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from receptor import (ModemConfig, ECCConfig, assemble_frame, simulate_capture,
                      compute_frame_layout, build_preamble, symbols_to_bits)
from receptor.layout import sample_cells_robust
from receptor.channel_coding import bytes_to_bits
from receptor.stages.roi import detect_roi
from receptor.stages.calibration import CalibrationStage
from receptor.pipeline import PipelineContext

ANGLES = [0.0, 15.0, 30.0]
N_TRIALS = 8
TX_TEXT = "PDS 2026 - benchmark BER Fase B receptor optico"

# Escenarios de iluminación: (nombre, brightness, ambient, gradient)
ILLUM = {
    "uniforme":  dict(brightness=0.85, ambient=0.0,  gradient=0.0),
    "gradiente": dict(brightness=0.60, ambient=40.0, gradient=0.5),
}

CONFIGS = ["Crudo", "+ Perspectiva", "+ Calib. global", "+ Calib. 2D (rolling)"]


def payload_geometry(config):
    layout = compute_frame_layout(config)
    pre = build_preamble(config, layout)
    return pre["data_positions"]


def decode_bits(image, config, data_cells, n_cells, thr=128):
    """Muestrea las celdas de payload y demodula a bits."""
    vals = sample_cells_robust(image, data_cells[:n_cells], config.cell_size)
    syms = np.clip(np.round(vals), 0, 255).astype(np.uint8)
    return symbols_to_bits(syms, config, thr)


def run_trial(fd, config, data_cells, n_cells, tx_bits, ang, illum):
    """Una realización: genera captura y evalúa BER de las 4 configuraciones."""
    cap = simulate_capture(fd["frame"], angle_deg=ang, noise_std=8.0, **illum)
    out = {}

    # 1 · Crudo: grilla nominal sobre la captura distorsionada
    rx = decode_bits(cap, config, data_cells, n_cells, thr=128)
    out["Crudo"] = ber(tx_bits, rx)

    # B1: homografía
    res = detect_roi(cap, config)
    if not res["success"]:
        out["+ Perspectiva"] = np.nan
        out["+ Calib. global"] = np.nan
        out["+ Calib. 2D (rolling)"] = np.nan
        return out, False

    warped = res["warped"]
    rx = decode_bits(warped, config, data_cells, n_cells, thr=128)
    out["+ Perspectiva"] = ber(tx_bits, rx)

    # B2: calibración (global y 2D)
    for mode, label in [("global", "+ Calib. global"), ("2d", "+ Calib. 2D (rolling)")]:
        ctx = PipelineContext(config=config, raw=cap)
        ctx.warped = warped
        CalibrationStage(verbose=False, force_mode=mode).run(ctx)
        thr = int(ctx.calib["thr"])
        rx = decode_bits(ctx.calibrated, config, data_cells, n_cells, thr)
        out[label] = ber(tx_bits, rx)
    return out, True


def ber(tx, rx):
    n = min(len(tx), len(rx))
    return float(np.mean(tx[:n] != rx[:n]))


def main():
    config = ModemConfig(scheme="BPSK_Manchester")
    ecc = ECCConfig(scheme="rs", nsym=16)
    fd = assemble_frame(TX_TEXT, config, ecc)
    data_cells = payload_geometry(config)
    n_cells = fd["n_payload_cells"]
    tx_bits = bytes_to_bits(fd["encoded"])[: (n_cells // config.chips_per_symbol)]

    print(f"Trama: {config.M}×{config.N}  payload={fd['payload_bytes']}B  "
          f"celdas={n_cells}  bits_canal={len(tx_bits)}  trials={N_TRIALS}")

    # results[illum][config] = lista de BER por ángulo
    results = {sc: {cfg: [] for cfg in CONFIGS} for sc in ILLUM}

    np.random.seed(2026)
    for sc, illum in ILLUM.items():
        print(f"\n══ Iluminación: {sc}  {illum} ══")
        header = "  ángulo │ " + " │ ".join(f"{c:>22s}" for c in CONFIGS)
        print(header)
        for ang in ANGLES:
            acc = {cfg: [] for cfg in CONFIGS}
            n_ok = 0
            for _ in range(N_TRIALS):
                out, ok = run_trial(fd, config, data_cells, n_cells, tx_bits, ang, illum)
                n_ok += ok
                for cfg in CONFIGS:
                    acc[cfg].append(out[cfg])
            row = {cfg: np.nanmean(acc[cfg]) for cfg in CONFIGS}
            for cfg in CONFIGS:
                results[sc][cfg].append(row[cfg])
            cells = " │ ".join(f"{row[c]:>22.4f}" for c in CONFIGS)
            print(f"  {ang:5.0f}° │ {cells}   (ROI ok {n_ok}/{N_TRIALS})")

    plot_results(results)


def plot_results(results):
    fig, axes = plt.subplots(1, len(ILLUM), figsize=(13, 5), sharey=True)
    colors = {"Crudo": "#d62728", "+ Perspectiva": "#ff7f0e",
              "+ Calib. global": "#1f77b4", "+ Calib. 2D (rolling)": "#2ca02c"}
    markers = {"Crudo": "x", "+ Perspectiva": "s",
               "+ Calib. global": "^", "+ Calib. 2D (rolling)": "o"}
    for ax, (sc, data) in zip(np.atleast_1d(axes), results.items()):
        for cfg in CONFIGS:
            ax.plot(ANGLES, data[cfg], marker=markers[cfg], color=colors[cfg],
                    lw=1.8, ms=8, label=cfg)
        ax.set_title(f"Iluminación {sc}", fontweight="bold")
        ax.set_xlabel("Ángulo de visión (°)")
        ax.set_xticks(ANGLES)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 0.55)
    np.atleast_1d(axes)[0].set_ylabel("BER de canal (pre-ECC)")
    np.atleast_1d(axes)[-1].legend(fontsize=9, loc="upper left")
    plt.suptitle("Fase B — BER vs ángulo por etapa de corrección (ablación)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = "benchmark_b5.png"
    plt.savefig(out, dpi=110)
    print(f"\n✓ Figura guardada → {out}")


if __name__ == "__main__":
    main()
