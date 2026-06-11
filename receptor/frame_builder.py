# ─────────────────────────────────────────────────────────────
#  receptor/frame_builder.py  —  Ensamblador de trama TX (con ECC)
# ─────────────────────────────────────────────────────────────
#
#  Reproduce `assemble_frame` de Fase A reutilizando los módulos compartidos,
#  pero añade la capa de codificación de canal (Reed-Solomon) al payload.
#  Sirve para PROBAR el receptor end-to-end con ECC sin tocar el notebook:
#
#      texto → RS → bits → símbolos → grilla → [canal] → receptor → texto
#
#  Orden de pintado idéntico a Fase A:
#      canvas(128) → pilotos → preámbulo → payload → marcadores (último).
# ─────────────────────────────────────────────────────────────
import numpy as np

from .config import ModemConfig, PILOT_VALUE
from .layout import (compute_frame_layout, generate_pilot_values,
                     draw_markers_on_frame)
from .modulation import text_to_bits, bits_to_symbols
from .preamble import build_preamble
from .channel_coding import ECCConfig, rs_encode_payload, bytes_to_bits


def payload_capacity_bytes(config: ModemConfig, n_data_cells: int) -> int:
    """Bytes de payload que caben en `n_data_cells` según el esquema."""
    if config.scheme == "BPSK_Manchester":
        cells_per_byte = 8 * 2          # 8 bits/byte · 2 celdas/bit (Manchester)
    else:                               # 4ASK: 2 bits/símbolo
        cells_per_byte = 8 // 2         # 8 bits/byte ÷ 2 bits/celda
    return n_data_cells // cells_per_byte


def assemble_frame(text: str, config: ModemConfig,
                   ecc: ECCConfig = None) -> dict:
    """Construye la imagen de trama a partir de TEXTO (atajo de Fase B)."""
    fd = assemble_payload_frame(text.encode("utf-8"), config, ecc)
    fd["text"] = text
    return fd


def assemble_payload_frame(data: bytes, config: ModemConfig,
                           ecc: ECCConfig = None) -> dict:
    """
    Construye la imagen de trama a partir de BYTES de payload arbitrarios
    (p.ej. un paquete de protocolo de Fase C: cabecera + datos).  Idéntico a
    `assemble_frame` pero sin asumir que el payload es texto UTF-8.
    """
    ecc = ecc or ECCConfig()
    layout = compute_frame_layout(config)
    preamble = build_preamble(config, layout)
    data_cells = preamble["data_positions"]

    # — Payload: bytes → RS → bits → símbolos —
    payload_bytes = payload_capacity_bytes(config, len(data_cells))
    raw = bytes(data)
    encoded = rs_encode_payload(raw, payload_bytes, ecc)     # exactamente payload_bytes
    pay_bits = bytes_to_bits(encoded)
    pay_syms = bits_to_symbols(pay_bits, config)

    # — Canvas gris medio —
    frame = np.full((config.frame_height, config.frame_width),
                    PILOT_VALUE, dtype=np.uint8)
    cs = config.cell_size

    def paint(cells, values):
        for (r, c), v in zip(cells, values):
            frame[r * cs:(r + 1) * cs, c * cs:(c + 1) * cs] = v

    # — Pilotos —
    paint(layout["pilot"], generate_pilot_values(len(layout["pilot"]), config.scheme))
    # — Preámbulo —
    paint(preamble["preamble_cells"], preamble["symbols"])
    # — Payload —
    n = min(len(pay_syms), len(data_cells))
    paint(data_cells[:n], pay_syms[:n])
    # — Marcadores (último) —
    draw_markers_on_frame(frame, layout, config)

    return {"frame": frame, "config": config, "layout": layout,
            "preamble": preamble, "ecc": ecc, "data": raw,
            "payload_bytes": payload_bytes, "encoded": encoded,
            "n_payload_cells": n}
