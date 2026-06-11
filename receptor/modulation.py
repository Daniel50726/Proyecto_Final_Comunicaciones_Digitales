# ─────────────────────────────────────────────────────────────
#  receptor/modulation.py  —  Mapeo bits ↔ símbolos (compartido TX/RX)
# ─────────────────────────────────────────────────────────────
#
#  Demoduladores idénticos a los del transmisor (FaseA.ipynb, Celda 2).
#  Los usan B3 (demodular el preámbulo) y B4 (demodular el payload).
# ─────────────────────────────────────────────────────────────
import numpy as np

from .config import ModemConfig, ASK4_LEVELS

_4ASK_MAP = {(0, 0): ASK4_LEVELS[0], (0, 1): ASK4_LEVELS[1],
             (1, 1): ASK4_LEVELS[2], (1, 0): ASK4_LEVELS[3]}
_4ASK_INV = {int(v): k for k, v in _4ASK_MAP.items()}


# ── Texto ↔ bits ──────────────────────────────────────────────
def text_to_bits(text: str) -> np.ndarray:
    """UTF-8 → array plano de bits (MSB primero por byte)."""
    raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return np.unpackbits(raw)


def bits_to_text(bits: np.ndarray) -> str:
    """Inversa de text_to_bits. Trunca al byte completo más cercano."""
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n]).tobytes().decode("utf-8", errors="replace")


# ── BPSK + Manchester ─────────────────────────────────────────
def bpsk_manchester_modulate(bits: np.ndarray) -> np.ndarray:
    chips = np.empty(len(bits) * 2, dtype=np.uint8)
    chips[0::2] = bits          # chip par   = bit
    chips[1::2] = 1 - bits      # chip impar = complemento
    return (chips * 255).astype(np.uint8)


def bpsk_manchester_demodulate(symbols: np.ndarray,
                               threshold: int = 128) -> np.ndarray:
    """Símbolos grises → bits vía umbral + extracción del chip par."""
    chips = (symbols >= threshold).astype(np.uint8)
    return chips[0::2]


# ── 4-ASK con código Gray ─────────────────────────────────────
def ask4_modulate(bits: np.ndarray) -> np.ndarray:
    if len(bits) % 2:
        bits = np.append(bits, 0)
    return np.array([_4ASK_MAP[tuple(b)] for b in bits.reshape(-1, 2)],
                    dtype=np.uint8)


def ask4_demodulate(symbols: np.ndarray) -> np.ndarray:
    nearest_idx = np.argmin(
        np.abs(symbols.astype(int)[:, None] - ASK4_LEVELS.astype(int)), axis=1)
    nearest = ASK4_LEVELS[nearest_idx]
    return np.concatenate([_4ASK_INV[int(s)] for s in nearest]).astype(np.uint8)


# ── Interfaz unificada ────────────────────────────────────────
def bits_to_symbols(bits: np.ndarray, config: ModemConfig) -> np.ndarray:
    if config.scheme == "BPSK_Manchester":
        return bpsk_manchester_modulate(bits)
    if config.scheme == "4ASK":
        return ask4_modulate(bits)
    raise ValueError(f"Esquema desconocido: {config.scheme}")


def symbols_to_bits(symbols: np.ndarray, config: ModemConfig,
                    threshold: int = 128) -> np.ndarray:
    if config.scheme == "BPSK_Manchester":
        return bpsk_manchester_demodulate(symbols, threshold)
    if config.scheme == "4ASK":
        return ask4_demodulate(symbols)
    raise ValueError(f"Esquema desconocido: {config.scheme}")
