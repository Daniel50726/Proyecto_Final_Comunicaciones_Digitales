# ─────────────────────────────────────────────────────────────
#  receptor/stages/sync.py  —  ETAPA 3: Sincronización  [B3 — STUB]
# ─────────────────────────────────────────────────────────────
#
#  CONTRATO (a implementar en Fase B3):
#    Entrada : ctx.calibrated (o ctx.warped), ctx.config
#    Salida  : ctx.sync = {"rs_slope": float,   # rolling shutter (px → Δt)
#                          "peak": float,        # correlación del preámbulo
#                          "lag": int,           # desfase espacial residual
#                          "locked": bool}
#
#  Plan:
#    1. Rolling shutter: alinear temporalmente celdas a distinta altura
#       (modelar desfase lineal con la fila y compensarlo).
#    2. Recuperador de reloj de símbolo: early-late gate sobre la intensidad
#       interpolada (relevante en captura de vídeo, no en cuadro estático).
#    3. Sincronización de trama: correlación cruzada circular con el preámbulo
#       Gold (n=5, TAPS_P1=(2,4), TAPS_P2=(1,2,3,4)); pico ≥ 0.7 → trama OK.
# ─────────────────────────────────────────────────────────────
from ..pipeline import PipelineStage, PipelineContext


class SyncStage(PipelineStage):
    name = "Sincronizacion"
    required = True

    def run(self, ctx: PipelineContext) -> bool:
        raise NotImplementedError(
            "B3: rolling shutter + early-late gate + correlación Gold del preámbulo")
