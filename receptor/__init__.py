"""Receptor del módem óptico espacio-temporal (Fase B).

Pipeline modular:  ROI → Calibración → Sincronización → Demapeo+ECC.
"""
from .config import ModemConfig
from .debug_viz import DebugViz
from .pipeline import ReceiverPipeline, PipelineContext, PipelineStage
from .simulate import simulate_capture
from .stages import (ROIStage, detect_roi,
                     CalibrationStage, SyncStage, DemapStage)

__all__ = ["ModemConfig", "DebugViz", "ReceiverPipeline", "PipelineContext",
           "PipelineStage", "simulate_capture", "ROIStage", "detect_roi",
           "CalibrationStage", "SyncStage", "DemapStage"]
