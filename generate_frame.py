#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  generate_frame.py  —  Transmisor: genera una trama (con/sin ECC)
# ─────────────────────────────────────────────────────────────
#
#  Construye la imagen PNG de una trama lista para MOSTRAR en pantalla y
#  fotografiar con la cámara.  A diferencia de frame_tx.png (Fase A, sin ECC),
#  aquí el payload puede ir protegido con Reed-Solomon, de modo que el receptor
#  corrija los errores del canal óptico y el texto salga limpio.
#
#  Ejemplos:
#    # trama con RS (recomendado para canal real)
#    python generate_frame.py --text "Hola Mundo!" --ecc rs --nsym 16 --out frame_rs.png
#
#    # luego: mostrar frame_rs.png en pantalla completa, fotografiarla y decodificar
#    python run_receptor.py --source image --path foto.png --ecc rs --nsym 16
#
#  IMPORTANTE: el receptor debe usar EXACTAMENTE los mismos --scheme/--ecc/--nsym
#  con que se generó la trama (la calibración/sync/demap dependen de ello).
# ─────────────────────────────────────────────────────────────
import argparse
import sys

import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from receptor import ModemConfig, ECCConfig, assemble_frame, payload_capacity_bytes
from receptor.layout import compute_frame_layout
from receptor.preamble import build_preamble


def main():
    ap = argparse.ArgumentParser(description="Generador de trama TX (con ECC)")
    ap.add_argument("--text", required=True, help="texto a transmitir")
    ap.add_argument("--out", default="frame_rs.png", help="PNG de salida")
    ap.add_argument("--scheme", choices=["BPSK_Manchester", "4ASK"],
                    default="BPSK_Manchester")
    ap.add_argument("--ecc", choices=["rs", "none"], default="rs")
    ap.add_argument("--nsym", type=int, default=16,
                    help="bytes de paridad RS (corrige nsym/2 bytes por bloque)")
    args = ap.parse_args()

    config = ModemConfig(scheme=args.scheme)
    ecc = ECCConfig(scheme=args.ecc, nsym=args.nsym)

    # Capacidad útil de texto (descontando preámbulo y paridad RS)
    layout = compute_frame_layout(config)
    data_cells = build_preamble(config, layout)["data_positions"]
    cap_bytes = payload_capacity_bytes(config, len(data_cells))
    usable = cap_bytes - args.nsym if args.ecc == "rs" else cap_bytes

    raw = args.text.encode("utf-8")
    if len(raw) > usable:
        print(f"⚠ El texto usa {len(raw)} bytes pero solo caben {usable} "
              f"(con ECC={args.ecc}, nsym={args.nsym}). Se truncará.")

    fd = assemble_frame(args.text, config, ecc)
    cv2.imwrite(args.out, fd["frame"])

    print(f"{'─' * 50}")
    print(f"  Trama generada     : {args.out}  "
          f"({config.frame_width}×{config.frame_height} px)")
    print(f"  Esquema            : {args.scheme}")
    print(f"  ECC                : {args.ecc}"
          + (f"  (nsym={args.nsym} → corrige hasta {args.nsym // 2} bytes/bloque)"
             if args.ecc == "rs" else ""))
    print(f"  Payload total      : {cap_bytes} bytes  "
          f"(texto útil ≈ {usable} bytes)")
    print(f"  Texto              : '{args.text}'  ({len(raw)} bytes)")
    print(f"{'─' * 50}")
    print(f"  Mostrar {args.out} en pantalla completa y fotografiar.")
    print(f"  Decodificar con:")
    print(f"    python run_receptor.py --source image --path FOTO.png "
          f"--scheme {args.scheme} --ecc {args.ecc}"
          + (f" --nsym {args.nsym}" if args.ecc == "rs" else ""))


if __name__ == "__main__":
    main()
