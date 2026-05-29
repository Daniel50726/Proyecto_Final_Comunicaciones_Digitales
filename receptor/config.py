# ─────────────────────────────────────────────────────────────
#  receptor/config.py  —  Configuración compartida TX/RX
# ─────────────────────────────────────────────────────────────
#
#  Esta clase es el ÚNICO contrato que el receptor necesita compartir con el
#  transmisor (Fase A).  El layout de la trama (marcadores, pilotos, preámbulo,
#  datos) es DETERMINISTA: se reconstruye localmente con `compute_frame_layout`
#  a partir de (M, N, marker_cells, pilot_period, scheme).  NO se transmite.
#
#  Mantener estos defaults idénticos a los de FaseA.ipynb (Celda 1).
# ─────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Literal

import numpy as np

# ── Constantes de modulación (idénticas a Fase A) ─────────────
ASK4_LEVELS  = np.array([0, 85, 170, 255], dtype=np.uint8)   # 4-ASK + Gray
BPSK_CHIPS   = {0: [0, 1], 1: [1, 0]}                        # Manchester
PILOT_VALUE  = 128                                            # gris medio conocido
MARKER_WHITE = 255
MARKER_BLACK = 0


@dataclass
class ModemConfig:
    # ── Grilla de datos ───────────────────────────────────────
    M: int = 36           # filas de macropíxeles
    N: int = 64           # columnas de macropíxeles
    cell_size: int = 20   # píxeles por lado de cada celda

    # ── Modulación ────────────────────────────────────────────
    scheme: Literal["BPSK_Manchester", "4ASK"] = "BPSK_Manchester"

    # ── Estructura de trama ───────────────────────────────────
    marker_cells: int = 7   # tamaño del finder pattern en celdas
    pilot_period: int = 4   # insertar piloto cada N celdas de datos

    # ── Temporización ─────────────────────────────────────────
    symbol_duration_ms: int = 100
    screen_fps: int = 60
    camera_fps: int = 30

    # ── Propiedades derivadas ─────────────────────────────────
    @property
    def frame_width(self) -> int:
        return self.N * self.cell_size

    @property
    def frame_height(self) -> int:
        return self.M * self.cell_size

    @property
    def bits_per_symbol(self) -> int:
        return 1 if self.scheme == "BPSK_Manchester" else 2

    @property
    def chips_per_symbol(self) -> int:
        """Chips que ocupa un símbolo en la grilla (Manchester duplica)."""
        return 2 if self.scheme == "BPSK_Manchester" else 1

    @property
    def total_cells(self) -> int:
        return self.M * self.N

    @property
    def reserved_cells(self) -> int:
        return 3 * (self.marker_cells ** 2)

    @property
    def available_data_cells(self) -> int:
        return self.total_cells - self.reserved_cells

    def summary(self) -> None:
        print(f"{'─' * 46}")
        print(f"  Esquema         : {self.scheme}")
        print(f"  Grilla          : {self.M}×{self.N}  ({self.total_cells} celdas)")
        print(f"  Tamaño celda    : {self.cell_size}×{self.cell_size} px")
        print(f"  Imagen canónica : {self.frame_width}×{self.frame_height} px")
        print(f"  Celdas libres   : {self.available_data_cells}")
        print(f"  Marcadores      : {self.marker_cells} celdas/lado")
        print(f"{'─' * 46}")
