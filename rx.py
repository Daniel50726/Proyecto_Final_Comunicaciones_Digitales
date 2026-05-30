#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  rx.py  —  RECEPTOR en tiempo real (Fase C)
# ─────────────────────────────────────────────────────────────
#
#  Captura continua de la cámara y reensamblado del mensaje multi-cuadro.
#
#  Flujo lógico
#  ────────────
#  1. CameraStream (hilo) mantiene SIEMPRE el último cuadro → el bucle nunca
#     procesa cuadros viejos ni acumula latencia (mitiga el desacople 60Hz/30fps).
#  2. Por cada cuadro nuevo:  ROI (B1) → calibración (B2) → decodificar payload
#     a bytes con RS (Fase C).
#  3. ACEPTACIÓN POR RS: solo se integran cuadros cuyo RS decodifica sin fallos.
#     Los cuadros desgarrados por rolling shutter o borrosos por motion-blur
#     fallan RS y se descartan → no contaminan el mensaje (sin reloj de símbolo).
#  4. MessageAssembler ordena los trozos por nº de secuencia hasta tener todos
#     [0..total-1] o ver EOM → reconstruye el texto.  Repetidos se ignoran;
#     perdidos se recuperan en la siguiente vuelta del transmisor.
#
#  Ejemplo:
#    python rx.py --cam-id 0 --nsym 16
#    python rx.py --cam-id 0 --expect "texto esperado para medir BER"
# ─────────────────────────────────────────────────────────────
import argparse
import sys
import time

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from receptor import (ModemConfig, ECCConfig, DebugViz, ReceiverPipeline,
                      ROIStage, CalibrationStage)
from receptor.camera import CameraStream
from receptor.protocol import (decode_payload_bytes, parse_packet,
                               MessageAssembler, SYNC, DATA, EOM)

WIN = "RX - Modem optico (q/ESC para salir)"
_TYPE = {SYNC: "SYNC", DATA: "DATA", EOM: "EOM"}


def char_ber(rx_text: str, tx_text: str) -> float:
    """BER aproximada a nivel de bit comparando los bytes UTF-8 alineados."""
    a = rx_text.encode("utf-8", "replace")
    b = tx_text.encode("utf-8", "replace")
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff_bits = sum(bin(a[i] ^ b[i]).count("1") for i in range(n))
    diff_bits += 8 * abs(len(a) - len(b))     # bytes faltantes/sobrantes
    return diff_bits / (8 * max(len(a), len(b)))


def draw_overlay(frame, status):
    disp = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = disp.shape[:2]
    if w > 960:
        disp = cv2.resize(disp, (960, int(h * 960 / w)))
    bar = disp.copy()
    cv2.rectangle(bar, (0, 0), (disp.shape[1], 70), (20, 20, 20), -1)
    disp = cv2.addWeighted(bar, 0.6, disp, 0.4, 0)
    roi_clr = (80, 230, 80) if status["roi"] else (80, 80, 230)
    cv2.putText(disp, f"ROI {'OK' if status['roi'] else '--'}  "
                      f"fps={status['fps']:.1f}  proc={status['proc']}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_clr, 1, cv2.LINE_AA)
    cv2.putText(disp, f"último: {status['last']}   progreso: {status['prog']}",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return disp


def main():
    ap = argparse.ArgumentParser(description="Receptor óptico en tiempo real")
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--scheme", choices=["BPSK_Manchester", "4ASK"],
                    default="BPSK_Manchester")
    ap.add_argument("--nsym", type=int, default=16, help="paridad RS por cuadro")
    ap.add_argument("--exposure", type=float, default=-6.0)
    ap.add_argument("--no-config", action="store_true",
                    help="no tocar los ajustes automáticos de la cámara")
    ap.add_argument("--expect", default=None,
                    help="texto esperado (mide BER y verifica al completar)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="segundos máx. esperando completar el mensaje")
    ap.add_argument("--show", action="store_true", default=True)
    args = ap.parse_args()

    config = ModemConfig(scheme=args.scheme)
    ecc = ECCConfig(scheme="rs", nsym=args.nsym)

    # Pipeline ligero: solo geometría + calibración; el demapeo+RS lo hace
    # decode_payload_bytes para obtener los BYTES del paquete.
    pipeline = ReceiverPipeline(
        config, [ROIStage(verbose=False), CalibrationStage(verbose=False)],
        viz=DebugViz(mode="none"))
    asm = MessageAssembler()

    cam = CameraStream(args.cam_id, configure=not args.no_config,
                       exposure=args.exposure).start()
    print("── Receptor en marcha. Apunta a la pantalla transmisora. ──")

    last_seq_proc = -1
    n_proc = n_accept = 0
    t0 = time.time()
    t_first = None
    fps_t, fps_n, fps = time.time(), 0, 0.0
    status = {"roi": False, "fps": 0.0, "proc": 0, "last": "-", "prog": "0/?"}

    try:
        while True:
            seq, frame = cam.read()
            if frame is None:
                continue
            if seq == last_seq_proc:          # sin cuadro nuevo → no re-procesar
                if args.show:
                    _show_and_poll(frame, status)
                continue
            last_seq_proc = seq
            n_proc += 1

            ctx = pipeline.process(frame, verbose=False)
            status["roi"] = bool(ctx.stage_ok.get("ROI"))
            if ctx.calibrated is not None:
                data, fail = decode_payload_bytes(ctx.calibrated, config, ecc)
                if fail == 0:                 # cuadro íntegro (RS ok)
                    pkt = parse_packet(data)
                    if pkt is not None:
                        n_accept += 1
                        if t_first is None:
                            t_first = time.time()
                        status["last"] = f"{_TYPE.get(pkt['type'],'?')} seq={pkt['seq']}"
                        if asm.add(pkt):
                            break
                        status["prog"] = asm.progress()

            # FPS
            fps_n += 1
            if time.time() - fps_t >= 0.5:
                fps = fps_n / (time.time() - fps_t)
                fps_t, fps_n = time.time(), 0
            status["fps"], status["proc"] = fps, n_proc

            if args.show and _show_and_poll(frame, status):
                break
            if time.time() - t0 > args.timeout and t_first is None:
                print("  ⏱ timeout sin detectar transmisión."); break
            if t_first is not None and time.time() - t_first > args.timeout:
                print("  ⏱ timeout reensamblando el mensaje."); break
    finally:
        cam.stop()
        if args.show:
            cv2.destroyAllWindows()

    _report(asm, args, t_first, n_proc, n_accept)


def _show_and_poll(frame, status) -> bool:
    cv2.imshow(WIN, draw_overlay(frame, status))
    k = cv2.waitKey(1) & 0xFF
    return k in (27, ord("q"))


def _report(asm, args, t_first, n_proc, n_accept):
    print("\n── Resultado ────────────────────────────────")
    print(f"  Cuadros procesados : {n_proc}   aceptados (RS ok): {n_accept}")
    if asm.done:
        dt = (time.time() - t_first) if t_first else 0.0
        print(f"  ✓ MENSAJE COMPLETO ({len(asm.text)} chars, {dt:.1f} s)")
        print(f"  RX: '{asm.text}'")
        if args.expect is not None:
            ber = char_ber(asm.text, args.expect)
            ok = asm.text == args.expect
            print(f"  Exacto: {'✓' if ok else '✗'}   BER≈{ber:.2e}"
                  f"   {'(< 1e-4 ✓)' if ber < 1e-4 else ''}")
    else:
        print(f"  ✗ Mensaje incompleto — progreso {asm.progress()}")
        if asm.chunks:
            partial = b"".join(asm.chunks[i] for i in sorted(asm.chunks)
                               ).decode("utf-8", "replace")
            print(f"  Parcial: '{partial[:120]}'")


if __name__ == "__main__":
    main()
