#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  run_receptor.py  —  Punto de entrada del receptor (Fase B)
# ─────────────────────────────────────────────────────────────
#
#  Orquesta el pipeline completo:
#    ROI → Calibración → Sincronización → Demapeo+ECC  (B1–B4).
#
#  Fuentes de imagen (--source):
#    sim    : genera una trama (frame_tx.png, o --text con ECC) y le aplica
#             simulate_capture (test sin cámara)
#    image  : carga una foto real con --path
#    camera : captura un cuadro de la webcam (--cam-id)
#
#  Depuración visual (--debug):
#    window : ventanas cv2.imshow (pulsa una tecla para avanzar)
#    save   : guarda PNGs en receptor/debug_out/   (default)
#    none   : sin salida visual
#
#  Ejemplos:
#    python run_receptor.py --source sim --angle 15 --debug window
#    python run_receptor.py --source sim --text "Hola PDS 2026" --ecc rs --angle 20
#    python run_receptor.py --source image --path captura.png --debug save
# ─────────────────────────────────────────────────────────────
import argparse
import os
import sys

import cv2

# Windows: la consola por defecto (cp1252) no imprime los caracteres Unicode
# (─, ✓, ✗, …) que usa el pipeline.  Forzar UTF-8 en stdout/stderr.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from receptor import (ModemConfig, DebugViz, ReceiverPipeline,
                       simulate_capture, ROIStage, CalibrationStage, SyncStage,
                       DemapStage, ECCConfig, assemble_frame)
from receptor.layout import compute_frame_layout, sample_cells


def load_frame(args, config) -> "tuple":
    """Devuelve (frame_capturado, frame_original_o_None, texto_tx_o_None)."""
    if args.source == "sim":
        if args.text is not None:
            # Generar trama con ECC desde texto (test end-to-end completo)
            ecc = ECCConfig(scheme=args.ecc, nsym=args.nsym)
            fd = assemble_frame(args.text, config, ecc)
            tx = fd["frame"]
        elif os.path.isfile(args.path):
            tx = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
        else:
            sys.exit(f"✗ No existe el frame transmitido: {args.path}\n"
                     f"  (usa --text para generar una trama, o genera frame_tx.png)")
        cap = simulate_capture(tx, angle_deg=args.angle, noise_std=args.noise,
                               brightness=args.brightness, blur_k=args.blur,
                               ambient=args.ambient, gradient=args.gradient)
        if args.occlude > 0:
            # Mancha/reflejo: bloque contiguo saturado → ráfaga de errores de byte
            h, w = cap.shape
            ow = int(w * args.occlude)
            x0 = w // 2
            cap[h // 4: h // 4 + h // 6, x0: x0 + ow] = 255
        return cap, tx, args.text

    if args.source == "image":
        img = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"✗ No se pudo leer la imagen: {args.path}")
        return img, None, None

    if args.source == "camera":
        cam = cv2.VideoCapture(args.cam_id)
        ok, frame = cam.read()
        cam.release()
        if not ok:
            sys.exit(f"✗ No se pudo capturar de la cámara {args.cam_id}")
        return frame, None, None

    sys.exit(f"✗ Fuente desconocida: {args.source}")


def build_pipeline(config: ModemConfig, viz: DebugViz,
                   ecc: ECCConfig) -> ReceiverPipeline:
    """Pipeline completo B1–B4."""
    stages = [
        ROIStage(reproj_thresh=8.0, verbose=True),     # B1
        CalibrationStage(verbose=True),                # B2
        SyncStage(verbose=True),                       # B3
        DemapStage(ecc=ecc, verbose=True),             # B4
    ]
    return ReceiverPipeline(config, stages, viz=viz)


def demo_calibration_gain(ctx, tx, config) -> None:
    """Demuestra cuantitativamente la mejora de BER que aporta la calibración:
    compara el error de bit en los centros de las celdas de datos, en el frame
    rectificado SIN calibrar vs CON calibrar, contra el frame transmitido real.
    (Checkpoint Fase B: 'la BER mejora al activar la calibración de color')."""
    if ctx.warped is None or ctx.calibrated is None or tx is None:
        return
    layout = compute_frame_layout(config)
    data = layout["data"]
    cs = config.cell_size

    truth, _ = sample_cells(tx, data, cs)              # niveles transmitidos
    raw, _ = sample_cells(ctx.warped, data, cs)        # rectificado, sin calibrar
    cal, _ = sample_cells(ctx.calibrated, data, cs)    # rectificado + calibrado
    thr = ctx.calib["thr"]

    truth_bits = truth >= 128
    ber_raw = float(np.mean((raw >= thr) != truth_bits))
    ber_cal = float(np.mean((cal >= thr) != truth_bits))

    print("\n── Demostración: BER en centros de celda (umbral={:.0f}) ──".format(thr))
    print(f"  Sin calibrar : BER = {ber_raw:.4f}  ({int(ber_raw*len(data))} / {len(data)} celdas)")
    print(f"  Calibrado    : BER = {ber_cal:.4f}  ({int(ber_cal*len(data))} / {len(data)} celdas)")
    mejora = "✓ mejora" if ber_cal < ber_raw else ("= igual" if ber_cal == ber_raw else "✗ empeora")
    print(f"  Resultado    : {mejora}")


def main():
    ap = argparse.ArgumentParser(description="Receptor del módem óptico (Fase B)")
    ap.add_argument("--source", choices=["sim", "image", "camera"], default="sim")
    ap.add_argument("--path", default="frame_tx.png",
                    help="ruta de la imagen (sim/image)")
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--debug", choices=["window", "save", "none"], default="save")
    # parámetros de simulación
    ap.add_argument("--angle", type=float, default=15.0)
    ap.add_argument("--noise", type=float, default=8.0)
    ap.add_argument("--brightness", type=float, default=0.85)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--ambient", type=float, default=0.0,
                    help="nivel de luz ambiente aditivo (sim)")
    ap.add_argument("--gradient", type=float, default=0.0,
                    help="gradiente espacial de iluminación 0..1 (sim)")
    ap.add_argument("--occlude", type=float, default=0.0,
                    help="mancha/reflejo que cubre esta fracción del ancho (sim) "
                         "→ ráfaga de errores que RS debe corregir")
    # configuración del módem (debe coincidir con el transmisor)
    ap.add_argument("--scheme", choices=["BPSK_Manchester", "4ASK"],
                    default="BPSK_Manchester")
    # payload + codificación de canal
    ap.add_argument("--text", default=None,
                    help="texto a transmitir (sim): genera la trama con ECC")
    ap.add_argument("--ecc", choices=["rs", "none"], default="rs",
                    help="código de canal del payload")
    ap.add_argument("--nsym", type=int, default=16,
                    help="bytes de paridad RS (corrige nsym/2 bytes por bloque)")
    args = ap.parse_args()

    config = ModemConfig(scheme=args.scheme)
    config.summary()
    ecc = ECCConfig(scheme=args.ecc, nsym=args.nsym)

    viz = DebugViz(mode=args.debug)
    cap, _tx, tx_text = load_frame(args, config)

    pipeline = build_pipeline(config, viz, ecc)
    ctx = pipeline.process(cap, verbose=True)

    if args.source == "sim":
        demo_calibration_gain(ctx, _tx, config)

    if tx_text is not None:
        match = (ctx.text == tx_text)
        print("\n── Verificación end-to-end ──────────────────")
        print(f"  TX : '{tx_text}'")
        print(f"  RX : '{ctx.text}'")
        print(f"  {'✓ TEXTO EXACTO' if match else '✗ DIFERENCIA'}")

    print("\n── Resumen ─────────────────────────────────")
    for stage, ok in ctx.stage_ok.items():
        mark = "✓" if ok else ("⏳" if ok is None else "✗")
        print(f"  {mark} {stage}")
    for k, v in ctx.metrics.items():
        print(f"     {k:18s}: {v}")


if __name__ == "__main__":
    main()
