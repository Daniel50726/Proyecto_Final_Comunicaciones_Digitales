# ─────────────────────────────────────────────────────────────
#  receptor/pipeline.py  —  Núcleo del pipeline de recepción
# ─────────────────────────────────────────────────────────────
#
#  Arquitectura:  un objeto `PipelineContext` (mutable) fluye a través de una
#  lista ordenada de `PipelineStage`.  Cada etapa lee lo que necesita del
#  contexto, escribe su resultado y devuelve True/False según haya tenido
#  éxito.  El orquestador `ReceiverPipeline` corta en seco si una etapa
#  obligatoria falla y delega la depuración visual en `DebugViz`.
#
#  Este diseño desacopla:
#    • el ORDEN del pipeline (aquí)
#    • la LÓGICA de cada etapa (stages/*.py)
#    • la VISUALIZACIÓN (debug_viz.py)
#
#  → migrar a tiempo real = sustituir la fuente del primer frame y poner
#    DebugViz(mode="none"); el resto del pipeline no cambia.
# ─────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import ModemConfig
from .debug_viz import DebugViz


# ── Contexto que fluye por el pipeline ────────────────────────
@dataclass
class PipelineContext:
    config: ModemConfig
    raw: np.ndarray                              # frame capturado (gris o BGR)

    # — Etapa 1: ROI / rectificación —
    gray: Optional[np.ndarray] = None
    binary: Optional[np.ndarray] = None
    roi: Optional[dict] = None                   # dict de detect_roi
    warped: Optional[np.ndarray] = None          # frame rectificado canónico

    # — Etapa 2: calibración fotométrica —
    calib: Optional[dict] = None                 # {a, b, gain_map, ...}
    calibrated: Optional[np.ndarray] = None

    # — Etapa 3: sincronización —
    sync: Optional[dict] = None                  # {rolling_shutter, peak, lag, ...}

    # — Etapa 4: muestreo + demapeo + ECC —
    cell_values: Optional[np.ndarray] = None
    symbols: Optional[np.ndarray] = None
    bits: Optional[np.ndarray] = None
    text: Optional[str] = None

    # — Estado del pipeline —
    stage_ok: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


# ── Etapa abstracta ───────────────────────────────────────────
class PipelineStage(ABC):
    name: str = "stage"
    required: bool = True   # si True y falla, el pipeline se detiene

    @abstractmethod
    def run(self, ctx: PipelineContext) -> bool:
        """Procesa el contexto in-place; devuelve True si tuvo éxito."""
        ...

    def draw_debug(self, ctx: PipelineContext) -> Optional[np.ndarray]:
        """Devuelve una imagen BGR de depuración (o None si no aplica)."""
        return None


# ── Orquestador ───────────────────────────────────────────────
class ReceiverPipeline:
    def __init__(self,
                 config: ModemConfig,
                 stages: list,
                 viz: DebugViz = None):
        self.config = config
        self.stages = stages
        self.viz = viz or DebugViz(mode="none")

    def process(self, frame: np.ndarray, verbose: bool = True) -> PipelineContext:
        ctx = PipelineContext(config=self.config, raw=frame)

        for stage in self.stages:
            if verbose:
                print(f"\n── Etapa: {stage.name} {'─' * (40 - len(stage.name))}")
            try:
                ok = stage.run(ctx)
            except NotImplementedError as e:
                if verbose:
                    print(f"  ⏳ pendiente: {e}")
                ctx.stage_ok[stage.name] = None
                break
            ctx.stage_ok[stage.name] = ok

            dbg = stage.draw_debug(ctx)
            if dbg is not None:
                self.viz.show(stage.name, dbg)

            if not ok and stage.required:
                if verbose:
                    print(f"  ✗ {stage.name} falló — pipeline detenido.")
                break

        self.viz.close()
        return ctx
