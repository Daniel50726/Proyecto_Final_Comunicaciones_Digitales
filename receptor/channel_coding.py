# ─────────────────────────────────────────────────────────────
#  receptor/channel_coding.py  —  Reed-Solomon sobre GF(2^8)
# ─────────────────────────────────────────────────────────────
#
#  Código de canal para proteger el payload contra los errores residuales del
#  canal óptico.  RS es orientado a BYTE → corrige RÁFAGAS de errores (varios
#  bytes consecutivos), que es la firma de los errores espaciales de este canal
#  (p.ej. la esquina BR mal alineada → símbolos contiguos corruptos en orden
#  raster).  Es además el ECC que usan los códigos QR, coherente con el diseño
#  de finders/grilla 2D de este módem.
#
#  RS(n,k) con nsym=n−k bytes de paridad corrige hasta ⌊nsym/2⌋ bytes erróneos
#  por bloque.  Implementación autocontenida (sin dependencias) en GF(2^8) con
#  polinomio primitivo 0x11D (α=2), el estándar de QR/DVB.
#
#  Esquema de payload DETERMINISTA (sin campo de longitud, como el resto del
#  módem): TX y RX calculan idéntico nº de palabras-código a partir de la
#  capacidad de la grilla; el mensaje se rellena con 0x00 y se recupera con
#  rstrip('\x00').
# ─────────────────────────────────────────────────────────────
from dataclasses import dataclass

import numpy as np

_PRIM = 0x11D   # x^8+x^4+x^3+x^2+1

# ── Tablas log/antilog de GF(2^8) ─────────────────────────────
_GF_EXP = [0] * 512
_GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= _PRIM
for _i in range(255, 512):
    _GF_EXP[_i] = _GF_EXP[_i - 255]


# ── Aritmética en GF(2^8) ─────────────────────────────────────
def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] + 255 - _GF_LOG[b]) % 255]


def _gf_pow(a, p):
    return _GF_EXP[(_GF_LOG[a] * p) % 255]


def _gf_poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for j in range(len(q)):
        for i in range(len(p)):
            r[i + j] ^= _gf_mul(p[i], q[j])
    return r


def _gf_poly_eval(poly, x):
    y = poly[0]
    for c in poly[1:]:
        y = _gf_mul(y, x) ^ c
    return y


def _gf_inverse(x):
    return _GF_EXP[255 - _GF_LOG[x]]


def _gf_poly_scale(p, x):
    return [_gf_mul(c, x) for c in p]


def _gf_poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i + len(r) - len(p)] = p[i]
    for i in range(len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r


def _gf_poly_div(dividend, divisor):
    out = list(dividend)
    for i in range(len(dividend) - (len(divisor) - 1)):
        coef = out[i]
        if coef != 0:
            for j in range(1, len(divisor)):
                if divisor[j] != 0:
                    out[i + j] ^= _gf_mul(divisor[j], coef)
    sep = -(len(divisor) - 1)
    return out[:sep], out[sep:]


# ── Codificación ──────────────────────────────────────────────
def _rs_generator_poly(nsym):
    g = [1]
    for i in range(nsym):
        g = _gf_poly_mul(g, [1, _gf_pow(2, i)])
    return g


def _rs_encode_msg(msg, nsym):
    """msg (lista de bytes) → palabra-código sistemática [msg | paridad]."""
    gen = _rs_generator_poly(nsym)
    out = list(msg) + [0] * nsym
    for i in range(len(msg)):
        coef = out[i]
        if coef != 0:
            for j in range(1, len(gen)):
                out[i + j] ^= _gf_mul(gen[j], coef)
    return list(msg) + out[len(msg):]


# ── Decodificación (síndrome + Berlekamp-Massey + Chien + Forney) ─
def _rs_calc_syndromes(msg, nsym):
    return [0] + [_gf_poly_eval(msg, _gf_pow(2, i)) for i in range(nsym)]


def _rs_find_error_locator(synd, nsym):
    err_loc, old_loc = [1], [1]
    for i in range(nsym):
        delta = synd[i + 1]
        for j in range(1, len(err_loc)):
            delta ^= _gf_mul(err_loc[-(j + 1)], synd[i + 1 - j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = _gf_poly_scale(old_loc, delta)
                old_loc = _gf_poly_scale(err_loc, _gf_inverse(delta))
                err_loc = new_loc
            err_loc = _gf_poly_add(err_loc, _gf_poly_scale(old_loc, delta))
    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    return err_loc


def _rs_find_errors(err_loc, nmess):
    errs = len(err_loc) - 1
    pos = []
    for i in range(nmess):
        if _gf_poly_eval(err_loc, _gf_pow(2, i)) == 0:
            pos.append(nmess - 1 - i)
    if len(pos) != errs:
        return None   # no se pudieron localizar todos
    return pos


def _rs_find_errata_locator(coef_pos):
    e_loc = [1]
    for i in coef_pos:
        e_loc = _gf_poly_mul(e_loc, _gf_poly_add([1], [_gf_pow(2, i), 0]))
    return e_loc


def _rs_find_error_evaluator(synd, err_loc, nsym):
    _, remainder = _gf_poly_div(_gf_poly_mul(synd, err_loc), [1] + [0] * (nsym + 1))
    return remainder


def _rs_correct_errata(msg, synd, err_pos):
    coef_pos = [len(msg) - 1 - p for p in err_pos]
    err_loc = _rs_find_errata_locator(coef_pos)
    err_eval = _rs_find_error_evaluator(synd[::-1], err_loc, len(err_loc) - 1)[::-1]

    X = [_gf_pow(2, -(255 - p)) for p in coef_pos]   # posiciones de error
    E = [0] * len(msg)
    for i, Xi in enumerate(X):
        Xi_inv = _gf_inverse(Xi)
        prime_tmp = 1
        for j in range(len(X)):
            if j != i:
                prime_tmp = _gf_mul(prime_tmp, 1 ^ _gf_mul(Xi_inv, X[j]))
        y = _gf_poly_eval(err_eval[::-1], Xi_inv)
        y = _gf_mul(Xi, y)
        E[err_pos[i]] = _gf_div(y, prime_tmp) if prime_tmp else 0
    return _gf_poly_add(msg, E)


def _rs_decode_msg(msg, nsym):
    """
    Devuelve (mensaje_corregido, n_errores) o (None, -1) si irrecuperable.
    Blindado: cualquier excepción interna (codeword patológica con más errores
    de los corregibles) se trata como bloque irrecuperable, nunca propaga.
    """
    msg = list(msg)
    synd = _rs_calc_syndromes(msg, nsym)
    if max(synd) == 0:
        return msg[:-nsym], 0           # sin errores
    try:
        err_loc = _rs_find_error_locator(synd, nsym)
        errs = len(err_loc) - 1
        if errs * 2 > nsym:
            return None, -1             # supera la capacidad de corrección
        err_pos = _rs_find_errors(err_loc[::-1], len(msg))
        if err_pos is None:
            return None, -1
        corrected = _rs_correct_errata(msg, synd, err_pos)
        if max(_rs_calc_syndromes(corrected, nsym)) != 0:
            return None, -1
        return corrected[:-nsym], len(err_pos)
    except (ZeroDivisionError, IndexError, ValueError):
        return None, -1


# ── Capa de payload determinista (chunking) ───────────────────
@dataclass
class ECCConfig:
    scheme: str = "rs"     # "rs" | "none"
    nsym: int = 16         # bytes de paridad por palabra-código (corrige nsym/2)
    cw: int = 255          # longitud de palabra-código (≤255)

    @property
    def k(self) -> int:
        return self.cw - self.nsym


def _effective_cw(total_bytes: int, ecc: ECCConfig) -> int:
    """Palabra-código efectiva: se adapta a la capacidad de la grilla.
    Si la capacidad < cw configurado, usa una sola palabra-código del tamaño
    disponible (RS funciona con cualquier n ≤ 255)."""
    return min(ecc.cw, total_bytes)


_SCRAMBLE_KEY = 0xC0DE51


def _scramble(arr: np.ndarray) -> np.ndarray:
    """
    Blanqueo de energía: XOR con una secuencia pseudoaleatoria DETERMINISTA.
    Rompe las regiones sólidas (p.ej. el relleno 0x00, que en 4ASK serían grandes
    manchas negras que engullen los finders) → el grid transmitido parece ruido
    moteado y la detección de finders es robusta.  XOR es su propia inversa, así
    que la misma función sirve para des-scramblear.  No mueve los errores entre
    bytes (preserva la corrección de ráfagas de RS).
    """
    arr = np.asarray(arr, np.uint8)
    ks = np.random.RandomState(_SCRAMBLE_KEY).randint(0, 256, len(arr)).astype(np.uint8)
    return (arr ^ ks).astype(np.uint8)


def plan_codewords(total_bytes: int, ecc: ECCConfig) -> tuple:
    """(nº palabras-código, bytes de datos utilizables, cw_efectiva)."""
    if ecc.scheme == "none":
        return 0, total_bytes, 0
    cw = _effective_cw(total_bytes, ecc)
    if cw <= ecc.nsym:
        return 0, total_bytes, cw     # no cabe la paridad → sin protección
    n_cw = total_bytes // cw
    k = cw - ecc.nsym
    return n_cw, n_cw * k, cw


def rs_encode_payload(data: bytes, total_bytes: int, ecc: ECCConfig) -> np.ndarray:
    """
    Mensaje → flujo de bytes de exactamente `total_bytes` listo para la grilla.
    Rellena con 0x00, parte en n_cw bloques de k, RS-codifica cada uno y
    rellena el resto con 0x00.  Determinista (TX y RX calculan igual).
    """
    n_cw, usable, cw = plan_codewords(total_bytes, ecc)
    out = np.zeros(total_bytes, np.uint8)
    if ecc.scheme == "none" or n_cw == 0:      # sin RS (o sin espacio para paridad)
        d = np.frombuffer(data[:total_bytes], np.uint8)
        out[:len(d)] = d
    else:
        k = cw - ecc.nsym
        msg = bytearray(data[:usable]) + bytearray(max(0, usable - len(data)))
        stream = bytearray()
        for c in range(n_cw):
            block = list(msg[c * k:(c + 1) * k])
            stream += bytearray(_rs_encode_msg(block, ecc.nsym))
        out[:len(stream)] = np.frombuffer(bytes(stream), np.uint8)
    return _scramble(out)                       # blanqueo (detección robusta)


def rs_decode_payload(recv: np.ndarray, total_bytes: int,
                      ecc: ECCConfig) -> dict:
    """
    Flujo de bytes recibido → bytes de datos corregidos + estadísticas.
    Returns: {"data": bytes, "n_corrected": int, "n_failed": int}.
    """
    recv = _scramble(np.asarray(recv, np.uint8))    # des-blanqueo (inverso del TX)
    if ecc.scheme == "none":
        return {"data": recv.tobytes(), "n_corrected": 0, "n_failed": 0}

    n_cw, _, cw = plan_codewords(total_bytes, ecc)
    if n_cw == 0:
        return {"data": recv.tobytes(), "n_corrected": 0, "n_failed": 0}
    k = cw - ecc.nsym
    data = bytearray()
    n_corr = n_fail = 0
    for c in range(n_cw):
        block = recv[c * cw:(c + 1) * cw]
        dec, n = _rs_decode_msg(block, ecc.nsym)
        if dec is None:
            n_fail += 1
            data += bytearray(bytes(block[:k]))     # mejor esfuerzo: datos crudos
        else:
            n_corr += n
            data += bytearray(bytes(dec))
    return {"data": bytes(data), "n_corrected": n_corr, "n_failed": n_fail}


# ── Conversión bits ↔ bytes ───────────────────────────────────
def bytes_to_bits(data: np.ndarray) -> np.ndarray:
    return np.unpackbits(np.asarray(data, np.uint8))


def bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n].astype(np.uint8))
