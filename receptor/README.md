# Receptor — Fase B (Módem óptico espacio-temporal)

Receptor del módem óptico **pantalla → cámara** del proyecto de PDS 2026-1.
Localiza, rectifica, calibra, sincroniza y decodifica los cuadros capturados
por la cámara **sin conocimiento a priori** de la posición ni orientación de la
pantalla.

A diferencia de la Fase A (notebook), la Fase B es un **paquete `.py` modular**
pensado para la transición a tiempo real: un pipeline por etapas desacoplado de
la fuente de imagen y de la visualización.

---

## Pipeline

```
captura (cámara / simulada)
     │
     ▼  ┌──────────────────────────────────────────────────────────┐
        │  B1  ROIStage         detección de finders + homografía   │  → vista cenital
        │  B2  CalibrationStage calibración fotométrica + AGC (2D)  │  → niveles {0,255}
        │  B3  SyncStage        preámbulo Gold + rolling shutter    │  → trama sincronizada
        │  B4  DemapStage       muestreo robusto + demapeo + RS     │  → texto
        └──────────────────────────────────────────────────────────┘
     │
     ▼
   texto recuperado
```

Cada etapa es un `PipelineStage` con `run(ctx) -> bool` y `draw_debug(ctx)`.
Un `PipelineContext` mutable fluye por todas; el orquestador `ReceiverPipeline`
corta si una etapa obligatoria falla y delega la depuración visual en `DebugViz`.

**Migrar a tiempo real** = cambiar la fuente del primer cuadro y poner
`DebugViz(mode="none")`; el resto del pipeline no cambia.

---

## Etapas

| Etapa | Archivo | Qué hace |
|-------|---------|----------|
| **B1** Detección de ROI | `stages/roi.py` | Detecta los 3 finders (jerarquía de contornos → anillo interior B), estima una **homografía de perspectiva** (12 puntos) y rectifica a vista cenital. Refuerzo: **flat-field + CLAHE** (`normalize_illumination`) para tolerar gradientes de iluminación. |
| **B2** Calibración + AGC | `stages/calibration.py` | Invierte el canal `y = a·x + b` con los pilotos. Modo **2D**: ajusta planos `a(x,y)`, `b(x,y)` (compensa gradientes espaciales / rolling shutter residual); modo **global** de respaldo. Umbral de decisión adaptativo. |
| **B3** Sincronización | `stages/sync.py` | **Sincronización de trama** por correlación circular con el preámbulo **Gold** (pico ≥ 0.7, lag = 0). Diagnóstico de **rolling shutter** (deriva de brillo por fila). `early_late_gate` listo para modo vídeo. |
| **B4** Muestreo + ECC | `stages/demap.py` | **Muestreo robusto** (media + filtro de mediana, suprime reflejos), demapeo con umbral de B2, **Reed-Solomon** (corrige ráfagas) y reconstrucción de texto. |

---

## Módulos compartidos

| Módulo | Contenido |
|--------|-----------|
| `config.py` | `ModemConfig` — contrato TX/RX (M=36, N=64, cell_size=20, marker_cells=7). |
| `layout.py` | Layout determinista de la trama (`compute_frame_layout`), pilotos, finders, muestreo de celdas (`sample_cells`, `sample_cells_robust`). |
| `modulation.py` | `text↔bits`, `bits↔símbolos`, demodulación BPSK-Manchester / 4-ASK. |
| `preamble.py` | Secuencias `m_sequence` / `gold_sequence` (n=5), `build_preamble`, `verify_preamble`. |
| `channel_coding.py` | **Reed-Solomon** sobre GF(2⁸) autocontenido (`ECCConfig`, `rs_encode_payload`, `rs_decode_payload`). |
| `frame_builder.py` | `assemble_frame` (TX) — genera tramas con ECC para test end-to-end. |
| `pipeline.py` | `PipelineContext`, `PipelineStage`, `ReceiverPipeline`. |
| `debug_viz.py` | `DebugViz` (window / save / none) + helpers de dibujo cv2 (`plot1d`, `hstack_panels`, `banner`). |
| `simulate.py` | `simulate_capture` — perspectiva, ruido, brillo, gradiente, ambiente (test sin cámara). |
| `video.py` | Modo vídeo: `simulate_video_stream`, `SymbolClock` (reloj de símbolo), `VideoReceiver`. |

---

## Layout determinista (clave del diseño)

El mapa de celdas (marcadores, separadores, pilotos, datos) **no se transmite**:
TX y RX lo calculan localmente con `compute_frame_layout(config)` a partir de
`(M, N, marker_cells, pilot_period)`. Solo hace falta compartir la `ModemConfig`.

---

## Codificación de canal: Reed-Solomon

Se eligió **RS sobre GF(2⁸)** porque:

- Los errores de este canal son **en ráfaga** (espaciales: una mancha/reflejo o
  la esquina BR mal alineada corrompe bytes contiguos en orden raster). RS es
  orientado a byte → corrige ráfagas mejor que un código orientado a bit.
- Es el ECC de los **códigos QR**, coherente con el diseño de finders/grilla 2D.

`RS(n,k)` con `nsym = n−k` bytes de paridad corrige hasta `⌊nsym/2⌋ bytes` por
bloque. La palabra-código se adapta a la capacidad de la grilla.

> Nota: RS es ineficiente con errores **dispersos** de ruido (cada bit malo =
> un byte a corregir). En canal ruidoso, subir `--nsym`.

---

## Uso

### Requisitos
```
python -m pip install numpy opencv-python matplotlib
```

### Receptor de cuadro único — `run_receptor.py`
```bash
# Decodificar frame_tx.png (generado por Fase A) a varios ángulos
python run_receptor.py --source sim --angle 15 --debug window

# End-to-end con texto propio + Reed-Solomon
python run_receptor.py --source sim --text "Hola PDS 2026" --ecc rs --angle 20

# Demostrar el valor del ECC: una mancha que RS corrige
python run_receptor.py --source sim --text "Modem optico Fase B" --ecc rs --angle 10 --occlude 0.12
#   (compárese con --ecc none → texto corrupto)

# Foto real de la cámara
python run_receptor.py --source image --path captura.png --debug window
```

Opciones de canal (sim): `--angle --noise --brightness --blur --ambient
--gradient --occlude`. Depuración: `--debug window|save|none`.

### Benchmark cuantitativo — `benchmark_b5.py`
```bash
python benchmark_b5.py        # → tabla por consola + benchmark_b5.png
```
Ablación de BER vs ángulo (0°/15°/30°) × iluminación (uniforme/gradiente) para
las configuraciones *Crudo / +Perspectiva / +Calib.global / +Calib.2D*.
Demuestra el checkpoint: la BER mejora al activar cada corrección.

### Modo vídeo en vivo — `run_camera.py`
```bash
# Stream simulado (sin hardware): secuencia de mensajes
python run_camera.py --source sim --text "Hola|PDS 2026|Fase B|fin" --debug window

# Cámara real apuntando a la pantalla transmisora
python run_camera.py --source camera --cam-id 0 --spc 3
```
`--spc` = cuadros por símbolo (`camera_fps · symbol_ms/1000`). El overlay
muestra el periodo estimado real; ajusta `--spc` si difiere.

---

## API mínima

```python
from receptor import (ModemConfig, DebugViz, ReceiverPipeline, ECCConfig,
                      ROIStage, CalibrationStage, SyncStage, DemapStage)

config = ModemConfig()
pipeline = ReceiverPipeline(config, [
    ROIStage(), CalibrationStage(), SyncStage(),
    DemapStage(ecc=ECCConfig(scheme="rs", nsym=16)),
], viz=DebugViz(mode="window"))

ctx = pipeline.process(frame)     # frame: np.ndarray gris/BGR de la cámara
print(ctx.text)                   # texto recuperado
print(ctx.metrics)                # reproj, calib, sync_peak, ecc, ...
```

---

## Resultados verificados

- **B1**: detección 15/15 hasta condiciones severas (brillo 0.45, gradiente 0.75);
  reproj < 2.6 px a 0°/15°/30°.
- **B4**: `frame_tx.png` (Fase A) → `"Hola Mundo!"` exacto a 0°/15°/30°.
- **B5**: Crudo ~0.43 BER a 15°/30° → +Perspectiva ~0.003 (uniforme); bajo
  gradiente +Calib reduce de ~0.06 a ~0.022.
- **Vídeo**: secuencia de 4 mensajes recuperada exacta con tearing simulado;
  `period_est` correcto.

---

## Limitaciones conocidas / mejoras futuras

- La esquina **BR** (sin finder) se extrapola → más error de muestreo en el
  borde derecho bajo perspectiva. Candidato a refinamiento iterativo del BR.
- B1 muy sensible a gradientes extremos (brillo < 0.4 + gradiente > 0.8).
- `early_late_gate` por celda implementado pero el reloj de símbolo de vídeo usa
  la señal de cambio entre cuadros; falta validar `--source camera` con hardware.
