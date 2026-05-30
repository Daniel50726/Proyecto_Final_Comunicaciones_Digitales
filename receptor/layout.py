# ─────────────────────────────────────────────────────────────
#  receptor/layout.py  —  Layout determinista de la trama (compartido TX/RX)
# ─────────────────────────────────────────────────────────────
#
#  El mapa de celdas (marcadores, separadores, pilotos, datos) NO se transmite:
#  es DETERMINISTA y el receptor lo reconstruye localmente con solo conocer
#  (M, N, marker_cells, pilot_period).  Estas funciones son idénticas a las de
#  FaseA.ipynb (Celda 3) y las comparten las etapas B2 (calibración), B3
#  (sincronización) y B4 (demapeo).
#
#  Añade además helpers de MUESTREO de celdas sobre la imagen rectificada
#  (centro del macropíxel con margen anti-ISI) que B2/B4 reutilizan.
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np

from .config import ModemConfig

_PILOT_SEQ = np.array([0, 255, 0, 255], dtype=np.uint8)


# ── Patrón finder concéntrico (estilo QR) ─────────────────────
def generate_finder_pattern(size: int) -> np.ndarray:
    """Patrón binario size×size: 1=módulo oscuro, 0=módulo claro."""
    if size < 3:
        raise ValueError("marker_cells debe ser >= 3")
    pat = np.zeros((size, size), dtype=np.uint8)
    pat[0, :] = pat[-1, :] = 1
    pat[:, 0] = pat[:, -1] = 1
    if size >= 5:
        cs = size - 4
        pat[2:2 + cs, 2:2 + cs] = 1
    return pat


def render_finder_to_pixels(pattern: np.ndarray, cell_size: int) -> np.ndarray:
    """Upscaling por kron.  Convención imagen: oscuro(1)→0, claro(0)→255."""
    px = np.kron(pattern, np.ones((cell_size, cell_size), dtype=np.uint8))
    return ((1 - px) * 255).astype(np.uint8)


# ── Layout completo de la trama ───────────────────────────────
def compute_frame_layout(config: ModemConfig) -> dict:
    """
    Asigna cada celda (r,c) a: 'marker' | 'separator' | 'pilot' | 'data', más
    'corners' (esquinas sup-izq de los 3 finders).  Determinista y reproducible.
    """
    M, N, ms = config.M, config.N, config.marker_cells
    if N < 2 * ms + 1 or M < 2 * ms + 1:
        raise ValueError(
            f"Grilla {M}×{N} demasiado pequeña para marker_cells={ms}. "
            f"Mínimo M=N={2 * ms + 1}.")

    corners = [(0, 0), (0, N - ms), (M - ms, 0)]   # TL, TR, BL (BR libre)

    marker_set = set()
    for r0, c0 in corners:
        for dr in range(ms):
            for dc in range(ms):
                marker_set.add((r0 + dr, c0 + dc))

    sep_set = set()
    for r0, c0 in corners:
        for dr in range(-1, ms + 1):
            for dc in range(-1, ms + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < M and 0 <= c < N and (r, c) not in marker_set:
                    sep_set.add((r, c))

    reserved = marker_set | sep_set
    free = [(r, c) for r in range(M) for c in range(N) if (r, c) not in reserved]

    pilot_positions, data_positions = [], []
    for idx, pos in enumerate(free):
        (pilot_positions if idx % config.pilot_period == 0
         else data_positions).append(pos)

    return {"marker": marker_set, "separator": sep_set,
            "pilot": pilot_positions, "data": data_positions, "corners": corners}


def generate_pilot_values(n: int) -> np.ndarray:
    """n valores piloto alternando 0↔255 (idéntico a TX)."""
    return _PILOT_SEQ[np.arange(n) % len(_PILOT_SEQ)]


def draw_markers_on_frame(frame: np.ndarray, layout: dict,
                          config: ModemConfig) -> np.ndarray:
    """Escribe finders + separadores blancos sobre `frame` (in-place)."""
    pattern_px = render_finder_to_pixels(
        generate_finder_pattern(config.marker_cells), config.cell_size)
    cs = config.cell_size
    sz = config.marker_cells * cs
    for (r0, c0) in layout["corners"]:
        y0, x0 = r0 * cs, c0 * cs
        frame[y0:y0 + sz, x0:x0 + sz] = pattern_px
    for (r, c) in layout["separator"]:
        y0, x0 = r * cs, c * cs
        frame[y0:y0 + cs, x0:x0 + cs] = 255
    return frame


# ── Muestreo de celdas sobre la imagen rectificada ────────────
def cell_centers_px(positions: list, cell_size: int) -> np.ndarray:
    """Centros (x,y) en píxeles de una lista de celdas (r,c)."""
    if not positions:
        return np.empty((0, 2), dtype=float)
    rc = np.asarray(positions, dtype=float)
    return np.column_stack([(rc[:, 1] + 0.5) * cell_size,    # x
                            (rc[:, 0] + 0.5) * cell_size])    # y


def sample_cells(image: np.ndarray, positions: list,
                 cell_size: int, margin: int = None) -> tuple:
    """
    (media, σ) de la zona interior de cada celda (r,c) en `image`.
    margin = cell_size//8 ≈ 12.5% de recorte por lado → evita ISI espacial.
    Returns: (means, stds) ndarrays alineadas con `positions`.
    """
    if margin is None:
        margin = max(1, cell_size // 8)
    if not positions:
        return np.array([]), np.array([])
    means, stds = [], []
    for r, c in positions:
        patch = image[r * cell_size + margin:(r + 1) * cell_size - margin,
                      c * cell_size + margin:(c + 1) * cell_size - margin].astype(float)
        means.append(patch.mean())
        stds.append(patch.std())
    return np.asarray(means), np.asarray(stds)


def sample_cells_robust(image: np.ndarray, positions: list,
                        cell_size: int, margin: int = None) -> np.ndarray:
    """
    Muestreo ROBUSTO del centro del macropíxel (B4): aplica filtro de MEDIANA
    al parche interior para suprimir reflejos puntuales/especulares antes de
    promediar.  Devuelve un valor por celda.  Más resistente que `sample_cells`
    frente a brillos puntuales del canal óptico.
    """
    if margin is None:
        margin = max(1, cell_size // 8)
    ksize = max(3, (cell_size // 4) | 1)   # impar
    vals = []
    for r, c in positions:
        patch = image[r * cell_size + margin:(r + 1) * cell_size - margin,
                      c * cell_size + margin:(c + 1) * cell_size - margin]
        if patch.size == 0:
            vals.append(0.0)
            continue
        k = min(ksize, patch.shape[0] | 1, patch.shape[1] | 1)
        med = cv2.medianBlur(patch.astype(np.uint8), k) if k >= 3 else patch
        vals.append(float(med.mean()))
    return np.asarray(vals)
