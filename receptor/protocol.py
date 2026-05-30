# ─────────────────────────────────────────────────────────────
#  receptor/protocol.py  —  Protocolo de trama multi-cuadro (Fase C)
# ─────────────────────────────────────────────────────────────
#
#  Sobre el módem espacial de Fase A/B se monta un protocolo de NIVEL DE TRAMA
#  para transmitir mensajes que no caben en un solo cuadro:
#
#    ┌──────────── payload de UN cuadro (protegido por RS) ────────────┐
#    │  CABECERA (6 B)                          │  DATOS (length B)     │
#    │  type(1) seq(2) total(2) length(1)       │  trozo del mensaje    │
#    └──────────────────────────────────────────────────────────────────┘
#
#  Tipos de cuadro:
#    SYNC  : cuadro de sincronización (inicio de transmisión); lleva `total`.
#    DATA  : trozo `seq` del mensaje; también lleva `total` (redundancia).
#    EOM   : fin de mensaje.
#
#  La INTEGRIDAD por cuadro la garantiza Reed-Solomon: un cuadro se ACEPTA solo
#  si RS decodifica sin bloques fallidos.  Los cuadros desgarrados por rolling
#  shutter o borrosos fallan RS y se descartan automáticamente → no hace falta
#  un reloj de símbolo preciso, solo recolectar cuadros válidos por `seq` hasta
#  tener [0..total-1] (o ver EOM).  Esto reordena repetidos/perdidos.
# ─────────────────────────────────────────────────────────────
import struct

import numpy as np

from .config import ModemConfig
from .channel_coding import ECCConfig, rs_decode_payload, bits_to_bytes, plan_codewords
from .layout import compute_frame_layout, sample_cells_robust
from .modulation import symbols_to_bits
from .preamble import build_preamble
from .frame_builder import payload_capacity_bytes, assemble_payload_frame

# Tipos de cuadro
SYNC = 0xA5
DATA = 0x01
EOM = 0x5A

_HEADER = struct.Struct(">BHHB")   # type, seq, total, length  → 6 bytes
HEADER_LEN = _HEADER.size


# ── Capacidad de datos por cuadro ─────────────────────────────
def data_bytes_per_frame(config: ModemConfig, ecc: ECCConfig) -> int:
    """Bytes ÚTILES de mensaje por cuadro (descontando cabecera y paridad RS)."""
    layout = compute_frame_layout(config)
    data_cells = build_preamble(config, layout)["data_positions"]
    cap = payload_capacity_bytes(config, len(data_cells))
    _, usable, _ = plan_codewords(cap, ecc)     # bytes tras descontar paridad RS
    return max(1, usable - HEADER_LEN)


# ── Empaquetado / desempaquetado de la cabecera ───────────────
def pack_packet(ftype: int, seq: int, total: int, data: bytes = b"") -> bytes:
    return _HEADER.pack(ftype, seq, total, len(data)) + data


def parse_packet(payload: bytes) -> dict:
    """Bytes de payload decodificados → dict de paquete (o None si inválido)."""
    if len(payload) < HEADER_LEN:
        return None
    ftype, seq, total, length = _HEADER.unpack(payload[:HEADER_LEN])
    if ftype not in (SYNC, DATA, EOM):
        return None
    data = payload[HEADER_LEN:HEADER_LEN + length]
    if len(data) < length:
        return None
    return {"type": ftype, "seq": seq, "total": total,
            "length": length, "data": data}


# ── Lado TRANSMISOR: construir la secuencia de cuadros ────────
def build_tx_frames(text: str, config: ModemConfig, ecc: ECCConfig) -> list:
    """
    Mensaje → lista de cuadros (imágenes) a desplegar en orden:
        [SYNC, DATA_0, DATA_1, …, DATA_{n-1}, EOM]
    Cada elemento es dict {label, seq, frame}.
    """
    raw = text.encode("utf-8")
    dpf = data_bytes_per_frame(config, ecc)
    chunks = [raw[i:i + dpf] for i in range(0, len(raw), dpf)] or [b""]
    total = len(chunks)

    frames = []

    def add(label, ftype, seq, data=b""):
        pkt = pack_packet(ftype, seq, total, data)
        img = assemble_payload_frame(pkt, config, ecc)["frame"]
        frames.append({"label": label, "seq": seq, "type": ftype, "frame": img})

    add("SYNC", SYNC, 0)
    for i, ch in enumerate(chunks):
        add(f"DATA[{i}]", DATA, i, ch)
    add("EOM", EOM, total)
    return frames


# ── Lado RECEPTOR: decodificar el payload de un cuadro a bytes ─
def decode_payload_bytes(image: np.ndarray, config: ModemConfig,
                         ecc: ECCConfig, thr: int = 128) -> tuple:
    """
    Imagen rectificada+calibrada → (payload_bytes, rs_failed_blocks).
    rs_failed_blocks == 0  →  cuadro íntegro (aceptar).
    """
    layout = compute_frame_layout(config)
    data_cells = build_preamble(config, layout)["data_positions"]
    cap = payload_capacity_bytes(config, len(data_cells))
    cells_per_byte = 16 if config.scheme == "BPSK_Manchester" else 4
    n_cells = cap * cells_per_byte

    vals = sample_cells_robust(image, data_cells[:n_cells], config.cell_size)
    syms = np.clip(np.round(vals), 0, 255).astype(np.uint8)
    bits = symbols_to_bits(syms, config, thr)
    recv = bits_to_bytes(bits)[:cap]
    if len(recv) < cap:
        recv = np.pad(recv, (0, cap - len(recv)))
    dec = rs_decode_payload(recv, cap, ecc)
    return dec["data"], dec["n_failed"]


# ── Reensamblador de mensaje (acumula cuadros válidos) ────────
class MessageAssembler:
    """Acumula trozos de mensaje por número de secuencia hasta completarlo."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = None
        self.chunks = {}      # seq → bytes
        self.seen_eom = False
        self.done = False
        self.text = None

    def add(self, pkt: dict) -> bool:
        """Integra un paquete válido.  Devuelve True si el mensaje quedó completo."""
        if pkt is None or self.done:
            return self.done
        if pkt["type"] == SYNC:
            # nuevo mensaje: reiniciar si cambia el total
            if self.total is None:
                self.total = pkt["total"]
            return False
        if pkt["total"]:
            self.total = pkt["total"]
        if pkt["type"] == DATA:
            self.chunks[pkt["seq"]] = pkt["data"]
        elif pkt["type"] == EOM:
            self.seen_eom = True

        if self.total is not None and len(self.chunks) >= self.total \
                and all(i in self.chunks for i in range(self.total)):
            raw = b"".join(self.chunks[i] for i in range(self.total))
            self.text = raw.decode("utf-8", errors="replace")
            self.done = True
        return self.done

    def progress(self) -> str:
        got = len(self.chunks)
        tot = self.total if self.total is not None else "?"
        return f"{got}/{tot}" + (" +EOM" if self.seen_eom else "")
