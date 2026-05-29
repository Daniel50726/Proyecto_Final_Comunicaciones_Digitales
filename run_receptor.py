#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  run_receptor.py  —  Punto de entrada del receptor (Fase B)
# ─────────────────────────────────────────────────────────────
#
#  Orquesta el pipeline ROI → Calibración → Sincronización → Demapeo.
#  Por ahora SÓLO la etapa de ROI está implementada (B1); las demás son stubs.
#
#  Fuentes de imagen (--source):
#    sim    : carga frame_tx.png y le aplica simulate_capture (test sin cámara)
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
#    python run_receptor.py --source image --path captura.png --debug save
#    python run_receptor.py --source sim --angle 30 --debug save
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

from receptor import (ModemConfig, DebugViz, ReceiverPipeline,
                       simulate_capture, ROIStage)


def load_frame(args) -> "tuple":
    """Devuelve (frame_capturado, frame_original_o_None)."""
    if args.source == "sim":
        if not os.path.isfile(args.path):
            sys.exit(f"✗ No existe el frame transmitido: {args.path}\n"
                     f"  (genera frame_tx.png con la Fase A o usa --path)")
        tx = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
        cap = simulate_capture(tx, angle_deg=args.angle, noise_std=args.noise,
                               brightness=args.brightness, blur_k=args.blur)
        return cap, tx

    if args.source == "image":
        img = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"✗ No se pudo leer la imagen: {args.path}")
        return img, None

    if args.source == "camera":
        cam = cv2.VideoCapture(args.cam_id)
        ok, frame = cam.read()
        cam.release()
        if not ok:
            sys.exit(f"✗ No se pudo capturar de la cámara {args.cam_id}")
        return frame, None

    sys.exit(f"✗ Fuente desconocida: {args.source}")


def build_pipeline(config: ModemConfig, viz: DebugViz) -> ReceiverPipeline:
    """Por ahora sólo la etapa de ROI (B1).  Añadir las demás al implementarlas:

        from receptor import CalibrationStage, SyncStage, DemapStage
        stages = [ROIStage(...), CalibrationStage(), SyncStage(), DemapStage()]
    """
    stages = [
        ROIStage(reproj_thresh=8.0, verbose=True),
        # CalibrationStage(),   # B2
        # SyncStage(),          # B3
        # DemapStage(),         # B4
    ]
    return ReceiverPipeline(config, stages, viz=viz)


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
    # configuración del módem (debe coincidir con el transmisor)
    ap.add_argument("--scheme", choices=["BPSK_Manchester", "4ASK"],
                    default="BPSK_Manchester")
    args = ap.parse_args()

    config = ModemConfig(scheme=args.scheme)
    config.summary()

    viz = DebugViz(mode=args.debug)
    cap, _tx = load_frame(args)

    pipeline = build_pipeline(config, viz)
    ctx = pipeline.process(cap, verbose=True)

    print("\n── Resumen ─────────────────────────────────")
    for stage, ok in ctx.stage_ok.items():
        mark = "✓" if ok else ("⏳" if ok is None else "✗")
        print(f"  {mark} {stage}")
    for k, v in ctx.metrics.items():
        print(f"     {k:18s}: {v}")


if __name__ == "__main__":
    main()
