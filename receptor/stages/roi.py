# ─────────────────────────────────────────────────────────────
#  receptor/stages/roi.py  —  ETAPA 1: Detección de ROI + homografía
# ─────────────────────────────────────────────────────────────
#
#  Localiza la pantalla transmisora mediante los 3 finder patterns (estilo QR),
#  estima una homografía de PERSPECTIVA real y rectifica la grilla a vista
#  cenital canónica (frame_width × frame_height).
#
#  Estrategia (jerarquía de contornos de los finders):
#    Tras THRESH_BINARY_INV el anillo oscuro → blanco y forma cuadrados
#    concéntricos:  A=borde exterior, B=borde interior del anillo, C=bloque
#    central.  B SIEMPRE queda rodeado de blanco → su contorno es el más fiable
#    y se identifica por su TAMAÑO ESPERADO (proyectivamente estable).
#
#  Homografía:  se usan las 4 esquinas del anillo interior B de cada finder
#  (12 correspondencias) → perspectiva real (no afín como una regla de
#  paralelogramo sobre 3 centros).
#
#  Lógica numérica portada verbatim de FaseA.ipynb (Fase B, Celda 1).
#  La visualización matplotlib se reemplaza por dibujo cv2 (tiempo real).
# ─────────────────────────────────────────────────────────────
from dataclasses import dataclass

import cv2
import numpy as np

from ..debug_viz import to_bgr, hstack_panels, banner
from ..pipeline import PipelineStage, PipelineContext


# ── Preprocesamiento ──────────────────────────────────────────
def preprocess_for_detection(frame: np.ndarray,
                             blur_k: int = 5,
                             block_sz: int = 51,
                             C_offset: int = 10) -> tuple:
    """Gris → blur → umbral adaptativo INV → morfología.  Returns (gray, binary)."""
    gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3 else frame.copy())

    blurred = cv2.GaussianBlur(gray, (blur_k | 1, blur_k | 1), 0)

    block_sz = max(21, block_sz | 1)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,          # oscuro → 255 (anillos de finders)
        blockSize=block_sz, C=C_offset)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    return gray, binary


# ── Candidatos a finder (jerarquía de contornos) ──────────────
@dataclass
class FinderCandidate:
    center: np.ndarray
    area: float
    contour: np.ndarray
    score: float
    depth: int


def _centroid(cnt: np.ndarray) -> np.ndarray:
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1:
        x, y, w, h = cv2.boundingRect(cnt)
        return np.array([x + w / 2.0, y + h / 2.0])
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])


def _squareness(cnt: np.ndarray) -> float:
    """1.0 = cuadrado perfecto.  minAreaRect → invariante a rotación/perspectiva."""
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    return float(min(w, h) / max(w, h)) if max(w, h) > 0 else 0.0


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como [TL, TR, BR, BL] en el espacio imagen."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _finder_quad(cnt: np.ndarray) -> np.ndarray:
    """4 esquinas ordenadas del contorno del finder (≈ cuadrado)."""
    peri = cv2.arcLength(cnt, True)
    for eps in (0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return _order_quad(approx)
    return _order_quad(cv2.boxPoints(cv2.minAreaRect(cnt)))


def find_finder_candidates(binary: np.ndarray,
                           area_min: int = 300,
                           area_max: int = 120_000,
                           min_square: float = 0.55,
                           min_solid: float = 0.70) -> list:
    """Contornos tipo-anillo (cuadrados, sólidos, con hueco), por score desc."""
    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []

    H = hier[0]   # (N,4): [next, prev, child, parent]
    out = []
    for i, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if not (area_min <= area <= area_max):
            continue
        sq = _squareness(cnt)
        if sq < min_square:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solid = area / hull_area if hull_area > 1 else 0.0
        if solid < min_solid:
            continue
        child_idx = H[i][2]
        if child_idx == -1:                 # un anillo DEBE tener hueco interior
            continue
        grandchild_idx = H[child_idx][2]
        depth = 2 if grandchild_idx != -1 else 1
        score = sq * 0.35 + solid * 0.35 + (0.30 if depth == 2 else 0.0)
        out.append(FinderCandidate(_centroid(cnt), area, cnt, score, depth))

    out.sort(key=lambda c: -c.score)
    return out


def group_by_center(candidates: list, min_dist: float) -> list:
    """Agrupa los contornos concéntricos (A/B/C) de un mismo finder por centro."""
    groups = []
    for cand in candidates:
        for grp in groups:
            if np.linalg.norm(cand.center - grp[0].center) < min_dist:
                grp.append(cand)
                break
        else:
            groups.append([cand])
    return groups


def select_finder_groups(candidates: list, min_dist: float, n: int = 3):
    """Los n grupos (finders) más fiables, por score de su mejor miembro."""
    groups = group_by_center(candidates, min_dist)
    if len(groups) < n:
        return None
    groups.sort(key=lambda grp: -max(c.score for c in grp))
    return groups[:n]


def _pick_inner_ring(group: list, expected_side: float) -> FinderCandidate:
    """Elige el anillo interior B (tamaño más cercano al esperado)."""
    return min(group, key=lambda c: abs(np.sqrt(c.area) - expected_side))


# ── Asignación de esquinas / rol de cada finder ───────────────
def assign_corners(groups: list, config) -> dict:
    cs, ms = config.cell_size, config.marker_cells
    W, H = float(config.frame_width), float(config.frame_height)

    centers = np.array([grp[0].center for grp in groups], dtype=float)
    tl_idx = int(np.argmin(centers[:, 0] + centers[:, 1]))
    rest = [i for i in range(3) if i != tl_idx]
    tr_idx, bl_idx = ((rest[0], rest[1])
                      if centers[rest[0], 0] >= centers[rest[1], 0]
                      else (rest[1], rest[0]))

    role_grp = {"TL": groups[tl_idx], "TR": groups[tr_idx], "BL": groups[bl_idx]}
    tl, tr, bl = centers[tl_idx], centers[tr_idx], centers[bl_idx]

    Cc = ms * cs / 2.0
    d_canon = {"TL": W - 2 * Cc, "TR": W - 2 * Cc, "BL": H - 2 * Cc}
    d_cap = {"TL": np.linalg.norm(tr - tl),
             "TR": np.linalg.norm(tr - tl),
             "BL": np.linalg.norm(bl - tl)}

    finders = {}
    for role in ("TL", "TR", "BL"):
        scale = d_cap[role] / d_canon[role] if d_canon[role] > 0 else 1.0
        finders[role] = _pick_inner_ring(role_grp[role],
                                         expected_side=(ms - 2) * cs * scale)

    # Caja externa extrapolada (sólo visualización del ROI)
    C = (ms * cs) / 2.0
    vx = (tr - tl) / (W - 2 * C)
    vy = (bl - tl) / (H - 2 * C)
    tl_out = tl - C * vx - C * vy
    tr_out = tr + C * vx - C * vy
    bl_out = bl - C * vx + C * vy
    br_out = tr_out + bl_out - tl_out

    return {"TL": tl, "TR": tr, "BL": bl, "BR": tr + bl - tl,
            "TL_out": tl_out, "TR_out": tr_out,
            "BL_out": bl_out, "BR_out": br_out, "finders": finders}


# ── Homografía de perspectiva + rectificación ─────────────────
def compute_homography(corners: dict, config) -> tuple:
    cs, ms = config.cell_size, config.marker_cells
    W, H = float(config.frame_width), float(config.frame_height)

    Cc = ms * cs / 2.0                   # centro del finder ↔ borde (canónico)
    ih = (ms - 2) * cs / 2.0             # semilado del anillo interior B
    off = np.array([[-ih, -ih], [ih, -ih], [ih, ih], [-ih, ih]], np.float32)
    canon_center = {"TL": (Cc, Cc), "TR": (W - Cc, Cc), "BL": (Cc, H - Cc)}

    src, dst = [], []
    for role in ("TL", "TR", "BL"):
        src.append(_finder_quad(corners["finders"][role].contour))
        dst.append(np.array(canon_center[role], np.float32) + off)
    src = np.vstack(src).astype(np.float32)
    dst = np.vstack(dst).astype(np.float32)

    H_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H_mat is None:
        return None, np.inf
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H_mat).reshape(-1, 2)
    reproj = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
    return H_mat, reproj


def warp_to_canonical(image: np.ndarray, H: np.ndarray, config) -> np.ndarray:
    return cv2.warpPerspective(image, H,
                               (config.frame_width, config.frame_height),
                               flags=cv2.INTER_LINEAR)


# ── Pipeline de detección (función pura, reutilizable) ────────
def detect_roi(frame: np.ndarray, config,
               reproj_thresh: float = 8.0, verbose: bool = False) -> dict:
    """frame → dict(success, warped, H, corners, reproj_err, n_found, quality,
    gray, binary, candidates)."""
    cs, ms = config.cell_size, config.marker_cells
    finder_px = ms * cs
    inner_px = (ms - 2) * cs
    block_sz = max(21, (finder_px // 3) | 1)
    area_min = int(inner_px ** 2 * 0.10)     # tolera fuerte compresión de B
    area_max = int(finder_px ** 2 * 4)
    min_dist = float(finder_px * 0.6)

    # Borde para que findContours no recorte finders pegados al borde del frame.
    pad = cs
    frame_pad = cv2.copyMakeBorder(frame, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=128)

    gray_pad, binary_pad = preprocess_for_detection(frame_pad, block_sz=block_sz)
    gray = gray_pad[pad:-pad, pad:-pad]
    binary = binary_pad[pad:-pad, pad:-pad]

    cands_pad = find_finder_candidates(binary_pad,
                                       area_min=area_min, area_max=area_max)
    candidates = [
        FinderCandidate(c.center - pad, c.area,
                        (c.contour - pad).astype(c.contour.dtype), c.score, c.depth)
        for c in cands_pad]

    groups_pad = select_finder_groups(cands_pad, min_dist)

    if verbose:
        ng = len(group_by_center(cands_pad, min_dist))
        print(f"  Candidatos brutos  : {len(cands_pad)}")
        print(f"  Finders agrupados  : {ng}")

    if groups_pad is None:
        n = len(group_by_center(cands_pad, min_dist))
        return dict(success=False, warped=None, H=None, corners=None,
                    reproj_err=np.inf, n_found=n, quality=0.0,
                    gray=gray, binary=binary, candidates=candidates)

    corners_pad = assign_corners(groups_pad, config)
    H_pad, reproj = compute_homography(corners_pad, config)
    success = H_pad is not None and reproj <= reproj_thresh

    # H del espacio padded → original:  H = H_pad · T_pad
    if H_pad is not None:
        T_pad = np.array([[1., 0., float(pad)],
                          [0., 1., float(pad)],
                          [0., 0., 1.]], dtype=np.float64)
        H_mat = H_pad @ T_pad
    else:
        H_mat = None

    corners = {k: (np.asarray(v, dtype=float) - pad)
               for k, v in corners_pad.items() if k != "finders"}
    corners["finders"] = {
        name: FinderCandidate(f.center - pad, f.area,
                              (f.contour - pad).astype(f.contour.dtype),
                              f.score, f.depth)
        for name, f in corners_pad["finders"].items()}

    warped = (warp_to_canonical(gray if frame.ndim == 2 else frame, H_mat, config)
              if success else None)

    finder_score = float(np.mean([f.score for f in corners["finders"].values()]))
    reproj_score = float(np.clip(1.0 - reproj / reproj_thresh, 0.0, 1.0))
    quality = finder_score * 0.5 + reproj_score * 0.5 if success else 0.0

    if verbose:
        print(f"  Error reproyección : {reproj:.2f} px")
        print(f"  Quality score      : {quality:.3f}")
        print(f"  Estado             : {'✓ OK' if success else '✗ FALLO'}")

    return dict(success=success, warped=warped, H=H_mat, corners=corners,
                reproj_err=reproj, n_found=3 if groups_pad else 0,
                quality=quality, gray=gray, binary=binary, candidates=candidates)


# ── Etapa del pipeline (envoltura OO + depuración cv2) ────────
# BGR
_ROLE_CLR = {"TL": (68, 68, 255), "TR": (255, 136, 68), "BL": (0, 153, 255)}


class ROIStage(PipelineStage):
    name = "ROI"
    required = True

    def __init__(self, reproj_thresh: float = 8.0, verbose: bool = True):
        self.reproj_thresh = reproj_thresh
        self.verbose = verbose

    def run(self, ctx: PipelineContext) -> bool:
        res = detect_roi(ctx.raw, ctx.config,
                         reproj_thresh=self.reproj_thresh, verbose=self.verbose)
        ctx.roi = res
        ctx.gray = res["gray"]
        ctx.binary = res["binary"]
        ctx.warped = res["warped"]
        ctx.metrics["roi_reproj_px"] = res["reproj_err"]
        ctx.metrics["roi_quality"] = res["quality"]
        ctx.metrics["roi_n_found"] = res["n_found"]
        return res["success"]

    def draw_debug(self, ctx: PipelineContext):
        """Compone 3 paneles BGR: capturado+ROI | binarización+candidatos | warped+grid."""
        res, cfg = ctx.roi, ctx.config

        # — Panel 1: imagen capturada con ROI y finders —
        p1 = to_bgr(ctx.raw)
        cor = res["corners"]
        if cor is not None:
            quad = np.int32([cor["TL_out"], cor["TR_out"],
                             cor["BR_out"], cor["BL_out"]])
            cv2.polylines(p1, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)
            for name, f in cor["finders"].items():
                clr = _ROLE_CLR[name]
                cx, cy = f.center.astype(int)
                cv2.circle(p1, (cx, cy), 7, clr, -1, cv2.LINE_AA)
                cv2.circle(p1, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)
                ipts = np.int32(_finder_quad(f.contour))
                cv2.polylines(p1, [ipts], True, clr, 1, cv2.LINE_AA)
                cv2.putText(p1, name, (cx + 10, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2, cv2.LINE_AA)
            bx, by = np.int32(cor["BR"])
            cv2.drawMarker(p1, (bx, by), (255, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 16, 2)
        st = (f"OK  reproj={res['reproj_err']:.1f}px  Q={res['quality']:.2f}"
              if res["success"] else f"FALLO  {res['n_found']}/3 finders")
        p1 = banner(p1, f"1. Capturado  [{st}]")

        # — Panel 2: binarización + contornos candidatos —
        p2 = to_bgr(res["binary"])
        for cand in res["candidates"][:30]:
            cnt = cand.contour.reshape(-1, 1, 2).astype(np.int32)
            clr = (0, 255, 0) if cand.score > 0.7 else (0, 255, 255)
            cv2.polylines(p2, [cnt], True, clr, 1, cv2.LINE_AA)
        p2 = banner(p2, f"2. Binarizacion ({len(res['candidates'])} cand, verde>0.7)")

        # — Panel 3: frame rectificado con grilla —
        if res["warped"] is not None:
            p3 = to_bgr(res["warped"])
            cs = cfg.cell_size
            for k in range(0, cfg.frame_width + 1, cs):
                cv2.line(p3, (k, 0), (k, cfg.frame_height), (0, 200, 0), 1)
            for k in range(0, cfg.frame_height + 1, cs):
                cv2.line(p3, (0, k), (cfg.frame_width, k), (0, 200, 0), 1)
            p3 = banner(p3, f"3. Rectificado {cfg.frame_width}x{cfg.frame_height}")
        else:
            p3 = np.full((cfg.frame_height, cfg.frame_width, 3), 20, np.uint8)
            cv2.putText(p3, "Rectificacion fallida (<3 finders)",
                        (20, cfg.frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 240), 2, cv2.LINE_AA)
            p3 = banner(p3, "3. Rectificado [FALLO]")

        return hstack_panels([p1, p2, p3])
