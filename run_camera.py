#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  run_camera.py  —  Modo VÍDEO en vivo (cámara o stream simulado)
# ─────────────────────────────────────────────────────────────
#
#  Bucle continuo que alimenta cuadros al VideoReceiver: recupera el reloj de
#  símbolo, decodifica los cuadros estables (centro de símbolo) y muestra un
#  overlay en vivo con el estado del enlace.
#
#  Fuentes:
#    --source camera   : webcam (cv2.VideoCapture), con calentamiento de auto-exp.
#    --source sim      : stream sintético desde --text (lista separada por '|')
#                        → para validar sin hardware.
#
#  Salida:
#    --debug window    : ventana cv2.imshow en vivo  (q = salir, s = guardar)
#    --debug save      : guarda cuadros anotados en receptor/debug_out/video/
#
#  Ejemplos:
#    python run_camera.py --source camera --spc 3
#    python run_camera.py --source sim --text "Hola|PDS 2026|Fase B|fin" --debug save
# ─────────────────────────────────────────────────────────────
import argparse
import os
import sys
import time

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from receptor import ModemConfig, ECCConfig
from receptor.video import VideoReceiver, simulate_video_stream


def draw_overlay(frame, receiver, last_event, frame_idx) -> np.ndarray:
    """Dibuja una barra de estado del enlace sobre el cuadro capturado."""
    disp = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    disp = disp.copy()
    h, w = disp.shape[:2]
    bar_h = 92
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    disp = cv2.addWeighted(overlay, 0.65, disp, 0.35, 0)

    st = receiver.last_status
    locked = last_event is not None
    clr = (80, 230, 80) if locked else (200, 200, 200)
    cv2.putText(disp, f"VIDEO RX  frame={frame_idx}  cambio={st.get('change',0):.0f}"
                      f"  periodo~{st.get('period_est',0):.1f}f",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    if last_event is not None:
        peak = last_event.get("sync_peak", 0.0)
        ecc = last_event.get("ecc_failed", -1)
        tag = "SYNC" if peak >= 0.7 else "no-sync"
        ecc_tag = "ECC ok" if ecc == 0 else f"ECC fail({ecc})"
        cv2.putText(disp, f"[{tag}]  peak={peak:.2f}  {ecc_tag}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 1, cv2.LINE_AA)
    txt = (receiver.last_text or "")[:48]
    cv2.putText(disp, f"RX: {txt}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 230, 120), 2, cv2.LINE_AA)
    return disp


def frame_source(args, config, ecc):
    """Generador de cuadros según la fuente."""
    if args.source == "sim":
        msgs = args.text.split("|") if args.text else ["Hola PDS 2026"]
        frames = simulate_video_stream(msgs, config, ecc, spc=args.spc,
                                       angle=args.angle, noise=args.noise,
                                       brightness=args.brightness, tear=True)
        print(f"  Stream simulado: {len(frames)} cuadros / {len(msgs)} mensajes")
        for f in frames:
            yield f
    else:  # camera
        cam = cv2.VideoCapture(args.cam_id)
        if not cam.isOpened():
            sys.exit(f"✗ No se pudo abrir la cámara {args.cam_id}")
        for _ in range(args.warmup):       # descartar cuadros de auto-exposición
            cam.read()
        try:
            while True:
                ok, f = cam.read()
                if not ok:
                    break
                yield f
        finally:
            cam.release()


def main():
    ap = argparse.ArgumentParser(description="Receptor de vídeo en vivo (Fase B)")
    ap.add_argument("--source", choices=["camera", "sim"], default="sim")
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--spc", type=float, default=3.0,
                    help="cuadros por símbolo (camera_fps · symbol_ms/1000)")
    ap.add_argument("--debug", choices=["window", "save"], default="window")
    ap.add_argument("--scheme", choices=["BPSK_Manchester", "4ASK"],
                    default="BPSK_Manchester")
    ap.add_argument("--nsym", type=int, default=16)
    # sim
    ap.add_argument("--text", default=None, help="mensajes separados por '|' (sim)")
    ap.add_argument("--angle", type=float, default=10.0)
    ap.add_argument("--noise", type=float, default=5.0)
    ap.add_argument("--brightness", type=float, default=0.85)
    args = ap.parse_args()

    config = ModemConfig(scheme=args.scheme)
    ecc = ECCConfig(scheme="rs", nsym=args.nsym)
    rx = VideoReceiver(config, ecc, spc=args.spc)

    save_dir = os.path.join("receptor", "debug_out", "video")
    if args.debug == "save":
        os.makedirs(save_dir, exist_ok=True)

    print("── Receptor de vídeo ─────────────────────────")
    decoded = []
    for i, frame in enumerate(frame_source(args, config, ecc)):
        ev = rx.push(frame)
        if ev is not None and not ev["repeat"]:
            decoded.append(ev["text"])
            print(f"  ► símbolo @cuadro {ev['sample_idx']}: "
                  f"peak={ev['sync_peak']:.2f} eccfail={ev['ecc_failed']} "
                  f"→ '{ev['text']}'")

        disp = draw_overlay(frame, rx, ev, i)
        if args.debug == "window":
            cv2.imshow("Video RX", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite(f"rx_frame_{i:04d}.png", disp)
        else:
            cv2.imwrite(os.path.join(save_dir, f"frame_{i:03d}.png"), disp)

    ev = rx.finish()
    if ev is not None and not ev["repeat"]:
        decoded.append(ev["text"])
        print(f"  ► (flush) → '{ev['text']}'")

    if args.debug == "window":
        cv2.destroyAllWindows()
    print(f"\n  Mensajes decodificados: {decoded}")


if __name__ == "__main__":
    main()
