# ─────────────────────────────────────────────────────────────
#  receptor/stages/calibration.py  —  ETAPA 2: Calibración + AGC  [B2 — STUB]
# ─────────────────────────────────────────────────────────────
#
#  CONTRATO (a implementar en Fase B2):
#    Entrada : ctx.warped (frame rectificado), ctx.config
#    Salida  : ctx.calib  = {"a": float, "b": float, "gain_map": ndarray|None,
#                            "thr": float}
#              ctx.calibrated = imagen con brillo/contraste compensados
#
#  Plan:
#    1. Reconstruir layout determinista: compute_frame_layout(config) → pilotos.
#    2. Leer celdas piloto conocidas (0↔255 alternados, PILOT_VALUE=128 base).
#    3. Regresión LS:  y_rx = a·y_tx + b  →  a (ganancia), b (offset).
#    4. AGC software + decisor de umbral adaptativo (compensa deriva lenta).
#    5. Opcional: mapa 2D local de ganancia/offset para gradientes espaciales.
# ─────────────────────────────────────────────────────────────
from ..pipeline import PipelineStage, PipelineContext


class CalibrationStage(PipelineStage):
    name = "Calibracion"
    required = True

    def run(self, ctx: PipelineContext) -> bool:
        raise NotImplementedError(
            "B2: calibración fotométrica + AGC (pilotos → a,b → umbral adaptativo)")
