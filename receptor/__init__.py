"""Receptor del módem óptico espacio-temporal (Fase B).

Pipeline modular:  ROI → Calibración → Sincronización → Demapeo+ECC.
"""
from .config import ModemConfig
from .debug_viz import DebugViz
from .pipeline import ReceiverPipeline, PipelineContext, PipelineStage
from .simulate import simulate_capture
from .layout import compute_frame_layout, generate_pilot_values, sample_cells
from .modulation import (text_to_bits, bits_to_text,
                         bits_to_symbols, symbols_to_bits)
from .preamble import gold_sequence, build_preamble, verify_preamble
from .channel_coding import ECCConfig, rs_encode_payload, rs_decode_payload
from .frame_builder import (assemble_frame, assemble_payload_frame,
                            payload_capacity_bytes)
from .protocol import (build_tx_frames, decode_payload_bytes, parse_packet,
                       pack_packet, MessageAssembler, data_bytes_per_frame,
                       SYNC, DATA, EOM)
from .stages import (ROIStage, detect_roi,
                     CalibrationStage, SyncStage, DemapStage)

__all__ = ["ModemConfig", "DebugViz", "ReceiverPipeline", "PipelineContext",
           "PipelineStage", "simulate_capture", "compute_frame_layout",
           "generate_pilot_values", "sample_cells", "text_to_bits",
           "bits_to_text", "bits_to_symbols", "symbols_to_bits",
           "gold_sequence", "build_preamble", "verify_preamble",
           "ECCConfig", "rs_encode_payload", "rs_decode_payload",
           "assemble_frame", "assemble_payload_frame", "payload_capacity_bytes",
           "build_tx_frames", "decode_payload_bytes", "parse_packet",
           "pack_packet", "MessageAssembler", "data_bytes_per_frame",
           "SYNC", "DATA", "EOM",
           "ROIStage", "detect_roi", "CalibrationStage", "SyncStage", "DemapStage"]
