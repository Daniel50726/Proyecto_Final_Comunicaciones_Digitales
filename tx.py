#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  tx.py  —  TRANSMISOR en tiempo real (Fase C)
# ─────────────────────────────────────────────────────────────
#
#  Despliega la secuencia de cuadros del protocolo en una ventana a PANTALLA
#  COMPLETA con FONDO OSCURO, en bucle continuo:
#
#      [ SYNC, DATA_0, DATA_1, …, DATA_{n-1}, EOM ]  →  repetir
#
#  Flujo lógico
#  ────────────
#  1. El mensaje se trocea en n cuadros DATA (cada uno protegido por RS) más un
#     SYNC inicial y un EOM final (protocol.build_tx_frames).
#  2. Cada cuadro se MANTIENE en pantalla `hold_ms` (≫ periodo de cámara) para
#     que el receptor capture ≥1 cuadro limpio aunque algunos salgan desgarrados
#     por rolling shutter en las transiciones (esos los descarta RS).
#  3. La secuencia se repite en bucle: un receptor que llega tarde o pierde un
#     cuadro lo recupera en la siguiente vuelta (reordenado por nº de secuencia).
#
#  Fondo oscuro: la grilla (1280×720) se centra sobre un lienzo negro a pantalla
#  completa → reduce deslumbramiento y luz difusa hacia la cámara.
#
#  Ejemplo:
#    python tx.py --text "mensaje largo..." --hold-ms 160 --nsym 16
#    python tx.py --file mensaje.txt --monitor 1
# ─────────────────────────────────────────────────────────────
import argparse
import sys
import time

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from receptor import ModemConfig, ECCConfig
from receptor.protocol import build_tx_frames, data_bytes_per_frame

WIN = "TX - Modem optico (ESC para salir)"


def letterbox_black(frame_gray, screen_w, screen_h):
    """Centra la grilla sobre un lienzo NEGRO a pantalla completa (fondo oscuro)."""
    canvas = np.zeros((screen_h, screen_w, 3), np.uint8)
    h, w = frame_gray.shape
    # escalar manteniendo aspecto si la grilla excede la pantalla
    scale = min(screen_w / w, screen_h / h, 1.0)
    if scale < 1.0:
        w2, h2 = int(w * scale), int(h * scale)
        frame_gray = cv2.resize(frame_gray, (w2, h2), interpolation=cv2.INTER_NEAREST)
        h, w = h2, w2
    y0, x0 = (screen_h - h) // 2, (screen_w - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Transmisor óptico en tiempo real")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="mensaje a transmitir")
    src.add_argument("--file", help="archivo de texto a transmitir")
    ap.add_argument("--scheme", choices=["BPSK_Manchester", "4ASK"],
                    default="BPSK_Manchester")
    ap.add_argument("--nsym", type=int, default=16, help="paridad RS por cuadro")
    ap.add_argument("--hold-ms", type=int, default=160,
                    help="ms que cada cuadro permanece en pantalla")
    ap.add_argument("--screen-w", type=int, default=1920)
    ap.add_argument("--screen-h", type=int, default=1080)
    ap.add_argument("--loops", type=int, default=0,
                    help="nº de repeticiones de la secuencia (0 = infinito)")
    args = ap.parse_args()

    text = args.text if args.text else open(args.file, encoding="utf-8").read()
    config = ModemConfig(scheme=args.scheme)
    ecc = ECCConfig(scheme="rs", nsym=args.nsym)

    frames = build_tx_frames(text, config, ecc)
    dpf = data_bytes_per_frame(config, ecc)
    n_data = sum(1 for f in frames if f["type"] == 0x01)
    print(f"── Transmisor ───────────────────────────────")
    print(f"  Mensaje        : {len(text)} caracteres")
    print(f"  Datos/cuadro   : {dpf} bytes  →  {n_data} cuadros DATA")
    print(f"  Secuencia      : {len(frames)} cuadros (SYNC + DATA + EOM)")
    print(f"  Hold           : {args.hold_ms} ms/cuadro")
    est = len(frames) * args.hold_ms / 1000.0
    print(f"  Duración/vuelta: {est:.1f} s")
    print(f"  Esquema/ECC    : {args.scheme} / RS(nsym={args.nsym})")
    print(f"  ESC en la ventana para salir.")

    cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    loop = 0
    try:
        while args.loops == 0 or loop < args.loops:
            for fr in frames:
                disp = letterbox_black(fr["frame"], args.screen_w, args.screen_h)
                # etiqueta discreta de estado (esquina, en gris tenue)
                cv2.putText(disp, f"{fr['label']}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
                cv2.imshow(WIN, disp)
                # mantener el cuadro `hold_ms`; waitKey procesa eventos de ventana
                t_end = time.time() + args.hold_ms / 1000.0
                while time.time() < t_end:
                    if (cv2.waitKey(5) & 0xFF) == 27:   # ESC
                        raise KeyboardInterrupt
            loop += 1
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print(f"\n  Transmisión detenida tras {loop} vuelta(s).")


if __name__ == "__main__":
    main()
