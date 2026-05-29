# ─────────────────────────────────────────────────────────────
#  receptor/stages/demap.py  —  ETAPA 4: Muestreo, demapeo + ECC  [B4 — STUB]
# ─────────────────────────────────────────────────────────────
#
#  CONTRATO (a implementar en Fase B4):
#    Entrada : ctx.calibrated, ctx.sync, ctx.config
#    Salida  : ctx.cell_values, ctx.symbols, ctx.bits, ctx.text
#
#  Plan:
#    1. Muestreo robusto: por celda, promediar el centro del macropíxel
#       (margen ≈ cell_size//8 para evitar ISI espacial) + filtro de mediana
#       para suprimir reflejos puntuales.
#    2. Demapeo según esquema:
#         BPSK_Manchester → chip par (>=128) = bit.
#         4ASK            → cuantizar al nivel Gray más cercano {0,85,170,255}.
#    3. Codificación de canal (≥1 técnica de control de errores):
#         opciones válidas → bloque con detección / convolucional / Reed-Solomon / LDPC.
#    4. bits → texto con rstrip('\x00') (sin campo de longitud explícito).
# ─────────────────────────────────────────────────────────────
from ..pipeline import PipelineStage, PipelineContext


class DemapStage(PipelineStage):
    name = "Demapeo"
    required = True

    def run(self, ctx: PipelineContext) -> bool:
        raise NotImplementedError(
            "B4: muestreo (media+mediana) → demapeo → ECC → bits → texto")
