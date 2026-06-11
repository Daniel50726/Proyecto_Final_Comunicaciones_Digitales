# ─────────────────────────────────────────────────────────────
#  receptor/preamble.py  —  Preámbulo Gold y sincronización de trama
# ─────────────────────────────────────────────────────────────
#
#  Rol del preámbulo:  los finders (B1) resuelven DÓNDE está la grilla; el
#  preámbulo resuelve QUÉ celda es cuál dentro del área de datos (permutaciones
#  espaciales residuales) y confirma la identidad de la trama por su pico de
#  autocorrelación agudo.  Idéntico al transmisor (FaseA.ipynb, Celda 4).
# ─────────────────────────────────────────────────────────────
import numpy as np

from .config import ModemConfig
from .modulation import bits_to_symbols, symbols_to_bits

_TAPS_P1 = (2, 4)          # x^5+x^2+1
_TAPS_P2 = (1, 2, 3, 4)   # x^5+x^3+x^2+x+1


# ── LFSR de Fibonacci → m-secuencia ───────────────────────────
def m_sequence(n: int, rec_taps: tuple, seed=None) -> np.ndarray:
    """m-secuencia binaria de longitud 2^n − 1 (estado reciente-primero)."""
    if seed is None:
        seed = [1] + [0] * (n - 1)
    state = list(seed[:n])
    length = (1 << n) - 1
    out = np.empty(length, dtype=np.uint8)
    for i in range(length):
        out[i] = state[0]
        fb = 0
        for t in rec_taps:
            fb ^= state[t]
        state = [fb] + state[:-1]
    return out


def gold_sequence(n: int = 5, taps1=_TAPS_P1, taps2=_TAPS_P2,
                  delay: int = 7) -> np.ndarray:
    """Secuencia Gold = m1 ⊕ T^delay(m2)."""
    m1 = m_sequence(n, taps1)
    m2 = m_sequence(n, taps2)
    return (m1 ^ np.roll(m2, delay)).astype(np.uint8)


def circular_autocorr(seq: np.ndarray) -> np.ndarray:
    """Autocorrelación circular BPSK (0→−1,1→+1), normalizada (pico=1 en lag0)."""
    bpsk = 2.0 * seq.astype(float) - 1.0
    S = np.fft.fft(bpsk)
    corr = np.real(np.fft.ifft(S * np.conj(S)))
    return corr / corr[0]


# ── Construcción del preámbulo en la trama ────────────────────
def build_preamble(config: ModemConfig, layout: dict,
                   n_lfsr: int = 5, delay: int = 7) -> dict:
    """
    Reserva las primeras celdas de layout['data'] para el preámbulo.
    Returns: sequence, symbols, n_cells, preamble_cells, data_positions.
    """
    seq = gold_sequence(n_lfsr, delay=delay)
    if config.scheme == "4ASK" and len(seq) % 2 != 0:
        seq = seq[:-1]

    syms = bits_to_symbols(seq, config)
    n_cells = len(syms)
    free = layout["data"]
    if n_cells > len(free):
        raise ValueError(
            f"Preámbulo necesita {n_cells} celdas pero solo hay {len(free)}.")

    return {"sequence": seq, "symbols": syms, "n_cells": n_cells,
            "preamble_cells": free[:n_cells], "data_positions": free[n_cells:]}


# ── Verificación / sincronización de trama ────────────────────
def verify_preamble(received_syms: np.ndarray, preamble_info: dict,
                    config: ModemConfig, threshold: float = 0.7,
                    decision_thr: int = 128) -> dict:
    """
    Demodula los chips del preámbulo y calcula la correlación cruzada circular
    contra la secuencia Gold esperada.  Pico ≥ threshold → trama sincronizada.

    El LAG indica el desfase espacial residual: si lag≠0 hay una permutación de
    celdas no resuelta por la homografía (la trama está rotada/desplazada).
    """
    expected_seq = preamble_info["sequence"]
    received_bits = symbols_to_bits(received_syms.astype(np.uint8), config, decision_thr)

    n = min(len(expected_seq), len(received_bits))
    exp = 2.0 * expected_seq[:n].astype(float) - 1.0   # BPSK {0,1}→{-1,+1}
    rec = 2.0 * received_bits[:n].astype(float) - 1.0

    xc = np.real(np.fft.ifft(np.fft.fft(rec) * np.conj(np.fft.fft(exp))))
    xc_norm = xc / n
    peak = float(np.max(np.abs(xc_norm)))
    lag = int(np.argmax(np.abs(xc_norm)))

    return {"peak": peak, "lag": lag, "synced": peak >= threshold, "xc": xc_norm,
            "n_bits": n, "received_bits": received_bits[:n]}
