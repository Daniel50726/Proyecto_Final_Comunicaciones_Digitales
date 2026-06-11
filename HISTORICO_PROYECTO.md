# Histórico del Proyecto — Módem Óptico Espacio-Temporal

## Contexto General

Proyecto final de la asignatura **Procesamiento Digital de Señales (PDS) — 2026-1**.  
Objetivo: transmitir datos entre una **pantalla** (transmisor) y una **cámara web** (receptor)
usando modulación en escala de grises sobre una grilla de macropíxeles.  
Plataforma: Python · Jupyter Notebooks · OpenCV · NumPy · SciPy.

---

## Índice

1. [Arquitectura global](#1-arquitectura-global)
2. [Fase A — Transmisor](#2-fase-a--transmisor)
   - [Celda 1 — Configuración global](#celda-1--configuración-global)
   - [Celda 2 — Mapeo bits → símbolos](#celda-2--mapeo-bits--símbolos)
   - [Celda 3 — Marcadores y pilotos](#celda-3--marcadores-y-pilotos)
   - [Celda 4 — Preámbulo Gold](#celda-4--preámbulo-gold)
   - [Celda 5 — Ensamblador de trama](#celda-5--ensamblador-de-trama)
   - [Celda 6 — Decodificador y loopback](#celda-6--decodificador-y-loopback)
3. [Fase B — Receptor](#3-fase-b--receptor)
   - [Celda 1 — Detección de ROI](#celda-1--detección-de-roi)
4. [Decisiones de diseño transversales](#4-decisiones-de-diseño-transversales)
5. [Bugs encontrados y soluciones](#5-bugs-encontrados-y-soluciones)
6. [Contratos de funciones clave](#6-contratos-de-funciones-clave)
7. [Parámetros de configuración](#7-parámetros-de-configuración)
8. [Pendientes y próximas celdas](#8-pendientes-y-próximas-celdas)

---

## 1. Arquitectura Global

```
[FASE A — TRANSMISOR]
  texto → bits → símbolos → grilla → imagen PNG
         Celda 2    Celda 2   Celda 3,4,5

[FASE B — RECEPTOR]
  imagen capturada → rectificación → calibración → sincronización → bits → texto
       Celda B1           B1              B2             B3          B4
```

### Pipeline completo Fase A

```
text_to_bits()
    │
bits_to_symbols()          ← BPSK Manchester ó 4-ASK (seleccionable)
    │
compute_frame_layout()     ← reserva marcadores / separadores / pilotos
    │
build_preamble()           ← Gold sequence → primeras celdas de datos
    │
assemble_frame()           ← canvas 128 → pilotos → preámbulo → payload → marcadores
    │
save_frame()               ← PNG lossless
```

### Pipeline completo Fase B

```
simulate_capture() / cv2.VideoCapture
    │
detect_roi()               ← contornos → finders → homografía → warpPerspective
    │
calibrate_frame()          ← pilotos → a, b → AGC software            [pendiente B2]
    │
synchronize()              ← rolling shutter + early-late gate + Gold  [pendiente B3]
    │
sample_and_demap()         ← media + mediana por celda → símbolos → bits [pendiente B4]
    │
decode_with_ecc()          ← Hamming / Reed-Solomon                    [pendiente B5]
```

---

## 2. Fase A — Transmisor

### Celda 1 — Configuración global

**Archivo: `FaseA.ipynb`, celda 1**

#### Clase principal: `ModemConfig`

```python
@dataclass
class ModemConfig:
    M: int = 16              # filas de macropíxeles
    N: int = 16              # columnas de macropíxeles
    cell_size: int = 40      # píxeles por lado de cada celda
    scheme: str = "BPSK_Manchester"   # ó "4ASK"
    ask_levels: list = [0, 85, 170, 255]
    marker_cells: int = 3    # tamaño finder pattern en celdas
    pilot_period: int = 4    # 1 piloto cada N celdas libres
    symbol_duration_ms: int = 100
    screen_fps: int = 60
    camera_fps: int = 30
```

#### Constantes globales

```python
ASK4_LEVELS = np.array([0, 85, 170, 255], dtype=np.uint8)
PILOT_VALUE = 128
```

#### Decisión: recomendación de parámetros para canal real

| Uso | M | N | cell_size | marker_cells |
|-----|---|---|-----------|-------------|
| Prueba sintética | 16 | 16 | 40 | 3 |
| Canal real (cámara) | 20 | 20 | 32 | 7 |

**Razón:** `marker_cells=7` produce el patrón finder de 3 niveles jerárquicos (anillo + separador + bloque central) que OpenCV detecta robustamente. Con `marker_cells=3` solo hay 2 niveles, más susceptible a falsos positivos.

---

### Celda 2 — Mapeo bits → símbolos

**Funciones principales:**

| Función | Entrada | Salida |
|---------|---------|--------|
| `text_to_bits(text)` | `str` UTF-8 | `ndarray` uint8 de bits |
| `bits_to_text(bits)` | `ndarray` bits | `str` (trunca al byte completo) |
| `bpsk_manchester_modulate(bits)` | bits | chips `{0,255}` (2× longitud) |
| `bpsk_manchester_demodulate(syms, thr=128)` | chips grises | bits (extrae chip par) |
| `ask4_modulate(bits)` | bits | símbolos `{0,85,170,255}` |
| `ask4_demodulate(symbols)` | símbolos | bits (cuantiza al nivel más cercano) |
| `bits_to_symbols(bits, config)` | bits + config | símbolos (dispatcher) |
| `symbols_to_bits(symbols, config)` | símbolos + config | bits (dispatcher) |

#### Contrato BPSK Manchester

```
Bit 0 → chips [0, 1] → niveles [0, 255]   (transición ↑)
Bit 1 → chips [1, 0] → niveles [255, 0]   (transición ↓)
```

- Garantía DC: `mean(chips) = 128` siempre, independientemente de los datos.
- Eficiencia: **0.5 bit / celda**.
- Demodulación: `chip_par = (symbol >= 128)` → chip par codifica el bit directamente.

#### Contrato 4-ASK con código Gray

```
Dibit  Nivel  Razón del orden Gray
00  →    0    Símbolos adyacentes difieren en 1 solo bit
01  →   85    → minimiza BER cuando el ruido desplaza un nivel
11  →  170
10  →  255
```

- Eficiencia: **2 bits / celda**.
- Requiere buena calibración de brillo para distinguir los 4 niveles.

#### Cambiar de BPSK a 4-ASK

Solo actualizar en Celda 1 y re-ejecutar todas las celdas:
```python
cfg = ModemConfig(scheme="4ASK")
```

---

### Celda 3 — Marcadores y pilotos

**Funciones principales:**

| Función | Descripción |
|---------|-------------|
| `generate_finder_pattern(size)` | Matriz `size×size` binaria (1=oscuro, 0=claro). Anillo exterior + separador + bloque central. |
| `render_finder_to_pixels(pattern, cell_size)` | Upscaling por `np.kron`. 1→0 px, 0→255 px. |
| `compute_frame_layout(config)` | Retorna dict con posiciones por categoría. |
| `generate_pilot_values(n)` | Alterna `[0, 255, 0, 255, …]` para calibración lineal. |
| `draw_markers_on_frame(frame, layout, config)` | Dibuja finders + separadores sobre imagen. |

#### Contrato de `compute_frame_layout`

```python
layout = {
    "marker"   : set  {(r,c)}  # celdas de los 3 finders
    "separator": set  {(r,c)}  # borde blanco 1 celda alrededor de cada finder
    "pilot"    : list [(r,c)]  # 1 cada pilot_period celdas libres (orden raster)
    "data"     : list [(r,c)]  # celdas disponibles para preámbulo + payload
    "corners"  : list [(r0,c0)] # esquina sup-izq de cada finder (TL, TR, BL)
}
```

#### Distribución de 3 finders (estilo QR)

```
TL: (0,       0      )
TR: (0,       N − ms )
BL: (M − ms,  0      )
BR: libre (datos / pilotos / metadatos futuros)
```

#### Pilotos para calibración lineal

El receptor estima `y_rx = a · y_tx + b` resolviendo:
```
y_piloto_recibido_negro  ≈ b
y_piloto_recibido_blanco ≈ 255a + b
```
→ `a = (blanco_rx − negro_rx) / 255`,  `b = negro_rx`

#### Layout es determinista

El receptor puede reconstruir `compute_frame_layout(config)` localmente con solo conocer `M, N, marker_cells, pilot_period`. No requiere transmitir el mapa.

---

### Celda 4 — Preámbulo Gold

**Funciones principales:**

| Función | Descripción |
|---------|-------------|
| `m_sequence(n, rec_taps, seed)` | LFSR Fibonacci, longitud `2^n − 1`. |
| `gold_sequence(n, taps1, taps2, delay)` | `m1 ⊕ T^delay(m2)`. |
| `circular_autocorr(seq)` | Autocorrelación circular BPSK, normalizada. Pico=1 en lag=0. |
| `build_preamble(config, layout, n_lfsr, delay)` | Reserva primeras celdas de `layout["data"]`. |

#### Parámetros LFSR verificados (grado 5)

```python
# p1 = x^5 + x^2 + 1        recurrencia: a_t = a_{t-3} ⊕ a_{t-5}
TAPS_P1 = (2, 4)

# p2 = x^5 + x^3 + x^2 + x + 1   recurrencia: a_t = a_{t-2} ⊕ a_{t-3} ⊕ a_{t-4} ⊕ a_{t-5}
TAPS_P2 = (1, 2, 3, 4)
```

**Derivación de rec_taps desde el polinomio p(x) = x^n + … + 1:**
```
La recurrencia lineal es: a_{t+n} = Σ_{coeficientes c_k≠0, k<n} a_{t+k}
Reescrita para "siguiente bit dado el estado [a_t, …, a_{t-n+1}]":
  rec_taps = índices 0-based del estado cuyos valores se XOR-ean
```

#### Propiedades de la familia Gold (n=5)

```
Longitud L = 31 chips
Famillia: 33 secuencias
Correlación cruzada acotada ∈ {−1/31, −9/31, +7/31}
Sidelobe AC ideal: −1/31 ≈ −0.032
```

#### Rol del preámbulo en la trama espacial

- Los **finders** resuelven DÓNDE está la grilla (corrección geométrica).
- El **preámbulo** resuelve QUÉ celda es cuál dentro del área de datos (permutaciones espaciales residuales post-homografía).

#### Celdas consumidas por el preámbulo

| Esquema | Chips Gold | Celdas consumidas |
|---------|-----------|-------------------|
| BPSK Manchester | 31 bits × 2 chips/bit | **62 celdas** |
| 4-ASK | 30 bits (par) × 0.5 símbolo/bit | **15 celdas** |

---

### Celda 5 — Ensamblador de trama

**Función principal:** `assemble_frame(text, config, layout, preamble_info)`

#### Orden de pintado sobre el canvas (importante)

```
1. np.full(..., 128)        — gris medio, valor neutro
2. Pilotos                  — celdas de calibración
3. Preámbulo                — chips Gold
4. Payload                  — datos de usuario (relleno con 0s al final)
5. Marcadores (ÚLTIMO)      — finders + separadores blancos
                               (sus celdas son RESERVADAS → nunca sobreescriben datos)
```

**Por qué marcadores al final:** aunque sus celdas están reservadas, dibujarlos último garantiza integridad visual ante cualquier cambio futuro del código.

#### Padding del payload

- El payload se rellena con bits `0` hasta ocupar exactamente todas las celdas disponibles.
- El receptor quita el padding con `.rstrip('\x00')` al reconstruir el texto.
- No se transmite campo de longitud explícito.

#### Reutilización de layout y preámbulo

```python
# Para transmitir múltiples tramas: calcular layout/preámbulo una sola vez
layout   = compute_frame_layout(cfg)
preamble = build_preamble(cfg, layout)

fd1 = assemble_frame("Hola", cfg, layout=layout, preamble_info=preamble)
fd2 = assemble_frame("mundo", cfg, layout=layout, preamble_info=preamble)
```

#### Dict devuelto por `assemble_frame`

```python
{
    "frame"        : np.ndarray uint8  # imagen lista para mostrar
    "config"       : ModemConfig
    "layout"       : dict              # resultado de compute_frame_layout
    "preamble"     : dict              # resultado de build_preamble
    "text"         : str               # texto original
    "raw_bits"     : ndarray           # bits sin padding
    "padded_bits"  : ndarray           # bits con padding (lo que realmente se transmite)
    "payload_syms" : ndarray           # símbolos de celda del payload
    "capacity_bits": int               # máximo de bits por trama
    "used_bits"    : int               # bits del mensaje actual
    "efficiency"   : float             # used/capacity
}
```

---

### Celda 6 — Decodificador y loopback

**Pipeline de recepción:**

```
extract_cell_values()   ← media + σ de zona interior de cada celda (con margen)
       │
calibrate_from_pilots() ← regresión LS: y_rx = a·y_tx + b
       │
apply_calibration()     ← y_corr = (y_rx − b) / a, clip [0,255]
       │
verify_preamble()       ← correlación cruzada circular con Gold esperado
       │
symbols_to_bits()       ← demodulación (dispatcher según scheme)
       │
bits_to_text()          ← reconstrucción + rstrip('\x00')
```

#### Contrato de `extract_cell_values`

```python
margin = max(1, cell_size // 8)   # ~12.5% recorte por lado
# → evita ISI espacial entre celdas adyacentes
```

Retorna medias Y desviaciones estándar para cada tipo de celda:
`pilot_mean`, `pilot_std`, `preamble_mean`, `preamble_std`, `payload_mean`, `payload_std`

#### Calibración lineal por mínimos cuadrados

```python
A = np.column_stack([expected.astype(float), np.ones(len(expected))])
coef, *_ = np.linalg.lstsq(A, received.astype(float), rcond=None)
a, b = coef   # ganancia y offset
```

#### Verificación del preámbulo

- Pico de correlación ≥ 0.7 → trama sincronizada.
- En loopback sin canal: pico = 1.0, lag = 0.
- En canal real: pico > 0.7, lag ≈ 0 (si no hay permutación espacial residual).

#### Dos casos de verificación

| Caso | Imagen fuente | BER esperado |
|------|--------------|-------------|
| Loopback directo | `fd["frame"]` en memoria | `0.0` |
| Loopback desde PNG | `cv2.imread(path, IMREAD_GRAYSCALE)` | `0.0` (PNG es lossless) |
| Canal real | Foto de cámara | > 0, depende de condiciones |

---

## 3. Fase B — Receptor

### Arquitectura del receptor (5 celdas)

```
B1: detect_roi()           → rectificación geométrica
B2: calibrate_frame()      → AGC software + umbral adaptativo    [pendiente]
B3: synchronize()          → rolling shutter + reloj símbolo     [pendiente]
B4: sample_and_demap()     → muestreo robusto + ECC              [pendiente]
B5: benchmark_angles()     → BER vs ángulo, con/sin correcciones [pendiente]
```

---

### Celda 1 — Detección de ROI

**Funciones principales:**

| Función | Descripción |
|---------|-------------|
| `simulate_capture(frame, angle_deg, noise_std, brightness, blur_k, bg_color)` | Distorsión perspectiva + ruido + brillo + blur. Para test sin cámara. |
| `preprocess_for_detection(frame, blur_k, block_sz, C_offset)` | Gris → GaussianBlur → adaptiveThreshold(INV) → morfología open/close. |
| `find_finder_candidates(binary, area_min, area_max, min_square, min_solid)` | Contornos RETR_TREE + filtros → lista `FinderCandidate`. |
| `cluster_candidates(candidates, min_dist)` | Suprime detecciones duplicadas del mismo finder. |
| `select_best_finders(candidates, n=3)` | Retorna los 3 mejores agrupados (por área). |
| `get_outer_corner(finder, label)` | Extrae esquina exterior del frame desde `cv2.minAreaRect`. |
| `assign_corners(finders)` | Clasifica TL/TR/BL, calcula esquinas exteriores y BR virtual. |
| `compute_homography(corners, config)` | `findHomography` RANSAC: esquinas exteriores → `(0,0),(W,0),(W,H),(0,H)`. |
| `warp_to_canonical(image, H, config)` | `warpPerspective` al canvas canónico. |
| `detect_roi(frame, config)` | Pipeline completo. |

#### Estrategia de detección de finders (jerarquía de contornos)

```
THRESH_BINARY_INV: negro(0) → blanco(255), blanco → negro
findContours RETR_TREE

Para marker_cells=3:  anillo exterior negro → hijo (hueco blanco)         = 2 niveles
Para marker_cells=7:  anillo → separador → bloque central                 = 3 niveles

Discriminadores vs. celdas de datos:
  • Tiene hijo en jerarquía (hueco) → celdas simples NO
  • Área >> celda individual
  • Aspect ratio ≈ 1 (cuadrado)
  • Solidez > 0.72
```

#### Parámetros adaptativos en `detect_roi`

```python
finder_px = config.marker_cells * config.cell_size
block_sz  = max(21, (finder_px // 3) | 1)   # threshold adaptativo impar
area_min  = int(finder_px**2 * 0.03)
area_max  = int(finder_px**2 * 12)
min_dist  = float(finder_px * 0.6)           # clustering de candidatos
```

#### Asignación de esquinas

```
TL: mínimo  x + y   (más cercano a origen)
TR: mayor   x       entre los dos restantes
BL: mayor   y       entre los dos restantes
BR: TR_outer + BL_outer − TL_outer   (paralelogramo de esquinas exteriores)
```

#### Dict devuelto por `detect_roi`

```python
{
    "success"    : bool
    "warped"     : np.ndarray | None
    "H"          : np.ndarray | None  # homografía 3×3
    "corners"    : dict | None        # TL/TR/BL/BR centros + TL_out/TR_out/BL_out/BR_out
    "reproj_err" : float              # error de reproyección en px
    "n_found"    : int                # finders detectados (ideal = 3)
    "quality"    : float              # score compuesto [0, 1]
    "gray"       : np.ndarray         # escala de grises (debug)
    "binary"     : np.ndarray         # binarización (debug)
    "candidates" : list               # todos los candidatos (debug)
}
```

---

## 4. Decisiones de Diseño Transversales

### Grilla y temporización

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `cell_size = 40 px` | Inicial | Cámara 30 fps, pantalla 60 Hz, cada celda ocupa múltiples píxeles capturados |
| `symbol_duration = 100 ms` | Inicial | Garantiza captura estable a 30 fps (3 frames por símbolo) |
| `pilot_period = 4` | 25% overhead | Balance entre capacidad y resolución de calibración |

### Esquema de modulación: cuándo usar cada uno

| Situación | Esquema recomendado |
|-----------|-------------------|
| Canal con ruido / distancia larga | BPSK Manchester |
| Canal limpio / distancia corta | 4-ASK |
| Prueba inicial sin cámara | BPSK Manchester |

### Canvas background = 128 (gris medio)

- Evita que celdas sin asignar aparezcan como negro (confunde al umbral adaptativo del receptor).
- Facilita la calibración del receptor (punto de referencia neutro).

### Separadores de marcadores = 255 (blanco)

- Da contexto limpio al decodificador para encontrar bordes del finder.
- Facilita `THRESH_BINARY_INV` en el receptor (el separador aparece negro en la imagen binarizada, creando el "hijo" esperado en la jerarquía).

### Layout determinista

Tanto el transmisor como el receptor calculan `compute_frame_layout(config)` localmente. No se transmite el mapa de celdas. Solo se necesita compartir `M, N, marker_cells, pilot_period, scheme`.

---

## 5. Bugs Encontrados y Soluciones

### BUG 1 — BR virtual cae dentro de la grilla de datos (Fase B, Celda 1)

**Síntoma:** El punto `BR*` visualizado en la imagen capturada aparecía muy adentro de la grilla, no en la esquina inferior-derecha del frame.

**Causa raíz:**
La función `assign_corners` calculaba `BR = TR_center + BL_center - TL_center` (regla del paralelogramo sobre los **centros** de los finders). Bajo una transformación proyectiva (perspectiva), los centros NO satisfacen la regla del paralelogramo. Además, los centros están inset `half = marker_cells * cell_size / 2` píxeles desde el borde real del frame, lo que amplificaba el error.

**Solución aplicada:**

1. Nueva función `get_outer_corner(finder, label)`:
   - Usa `cv2.minAreaRect(finder.contour)` → `cv2.boxPoints()` → 4 esquinas del bounding box.
   - Selecciona la esquina en la dirección correcta: TL = mín(x+y), TR = máx(x−y), BL = mín(x−y).
   - Esta esquina **toca el borde real del frame** (verificado contra la Fase A donde los finders se colocan pegados al borde del canvas).

2. `assign_corners` ahora calcula:
   - `TL_out, TR_out, BL_out` → esquinas exteriores reales de cada finder.
   - `BR_out = TR_out + BL_out - TL_out` → paralelogramo aplicado a esquinas exteriores.
   - El error del paralelogramo sobre las esquinas exteriores (separadas `W × H` px) es mucho menor que sobre los centros (separados `(W - 2·half) × (H - 2·half)` px).

3. `compute_homography` ahora mapea:
   - **Fuente:** `[TL_out, TR_out, BR_out, BL_out]` (esquinas exteriores en imagen capturada).
   - **Destino:** `[(0,0), (W,0), (W,H), (0,H)]` (vértices del canvas canónico).
   - **Antes:** usaba centros → destinos `(half,half), (W-half,half), …` — incorrecto bajo perspectiva.

**Tabla comparativa:**

| | Con bug | Corregido |
|---|---------|-----------|
| Puntos fuente | Centros de finders (inset `half` px) | Esquinas exteriores del bounding box |
| Puntos destino | `(half,half), (W-half,half), …` | `(0,0), (W,0), (W,H), (0,H)` |
| Paralelogramo en | Centros → error grande bajo perspectiva | Esquinas exteriores → error pequeño |
| BR cae | Dentro de la grilla | Fuera del frame (correcto) |

---

### BUG 2 — Frame rectificado recorta el borde exterior del frame (Fase B, Celda 1)

**Síntoma:** En el panel "Frame rectificado" de `plot_roi_detection`, los finders de las esquinas aparecían parcialmente cortados (el borde exterior del frame desaparecía en la imagen rectificada).

**Causa raíz:**
`cv2.findContours` con `RETR_TREE` **no traza correctamente los contornos de regiones que tocan el borde de la imagen**. Cuando el frame ocupa toda la imagen capturada (caso 0° sin distorsión), el anillo exterior del finder TL comienza exactamente en el píxel (0,0). OpenCV deja de trazar la parte del contorno que coincide con el borde de la imagen, resultando en que el contorno queda 1-2 píxeles DENTRO de la esquina real.

`get_outer_corner` toma `cv2.minAreaRect` sobre ese contorno → devuelve un punto ligeramente interior a la verdadera esquina del frame.

`compute_homography` mapea ese punto interior → `(0,0)` en el canvas. La verdadera esquina del frame queda en coordenadas negativas → recortada por `warpPerspective`.

**Solución aplicada — padding en `detect_roi`:**

```python
pad = config.cell_size
frame_pad = cv2.copyMakeBorder(frame, pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=128)
```

Se añade un borde de `cell_size` píxeles (gris 128) antes del preprocesamiento. Esto aleja los finders del borde de la imagen, permitiendo que `findContours` trace el contorno completo.

La homografía se calcula en coordenadas del frame con borde (`H_pad`) y luego se convierte al espacio original:

```python
T_pad_mat = np.array([[1., 0., float(pad)],
                      [0., 1., float(pad)],
                      [0., 0., 1.        ]])
H_mat = H_pad @ T_pad_mat   # H_orig = H_pad @ T_pad
```

Esta composición garantiza que `warpPerspective(original_frame, H_mat, ...)` produce exactamente el mismo resultado que usar el frame con borde, sin necesitar cambiar el tamaño del canvas de salida.

Los corners, finders candidatos y contornos se desplazan `-pad` antes de devolverlos, manteniendo compatibilidad con `plot_roi_detection` (que dibuja sobre la imagen capturada original).

**Tabla comparativa:**

| | Con bug | Corregido |
|---|---------|-----------|
| Input de `findContours` | Frame original (finders en borde) | Frame con borde de `cell_size` px |
| Contorno TL | Truncado, no llega a (0,0) | Completo (finder alejado del borde) |
| `get_outer_corner` TL | ≈ (1-2, 1-2) en vez de (0,0) | ≈ (0,0) correcto |
| Canvas resultado | Borde exterior del frame recortado | Frame completo incluyendo finders |
| API de `detect_roi` | — | Sin cambios (H y corners en coords originales) |

---

## 6. Contratos de Funciones Clave

### Fase A

```
text_to_bits(text: str) → ndarray[uint8]
  PRE:  text es UTF-8 válido
  POST: cada byte → 8 bits MSB-first, shape = (8*len(bytes),)

bits_to_symbols(bits, config) → ndarray[uint8]
  PRE:  bits ∈ {0,1}, config.scheme ∈ {"BPSK_Manchester","4ASK"}
  POST: BPSK → len = 2*len(bits), valores ∈ {0,255}
        4ASK → len = len(bits)//2, valores ∈ {0,85,170,255}

compute_frame_layout(config) → dict
  PRE:  config.N > 2*marker_cells+1, config.M > 2*marker_cells+1
  POST: marker ∩ separator = ∅, (marker ∪ sep) ∩ pilot = ∅,
        pilot ∩ data = ∅, layout es reproducible con los mismos params

assemble_frame(text, config, layout, preamble_info) → dict
  PRE:  len(text_to_bits(text)) ≤ capacity_bits
  POST: frame es uint8 numpy array (H,W), todos los sets de celdas
        están correctamente pintados y NO se solapan

decode_frame(image, config, layout, preamble_info) → dict
  PRE:  image uint8 grayscale, layout = compute_frame_layout(config)
  POST: result["text"] == original si BER=0
        result["a"] ≈ 1.0, result["b"] ≈ 0.0 en canal ideal
```

### Fase B

```
detect_roi(frame, config) → dict
  PRE:  frame contiene al menos 3 finders visibles, suficiente contraste
  POST: si success=True: warped tiene shape (config.H, config.W)
        reproj_err < 15.0 px para ángulos ≤ 30°
        H es invertible (det(H) ≠ 0)

get_outer_corner(finder, label) → ndarray shape (2,)
  PRE:  label ∈ {"TL","TR","BL"}
        finder.contour proviene del finder REAL (no de célula de datos)
  POST: el punto devuelto toca o está muy cerca del borde exterior del frame
        (dentro de ~1 cell_size de error para ángulos ≤ 30°)
```

---

## 7. Parámetros de Configuración

### `ModemConfig` — referencia rápida

| Parámetro | Default | Afecta | Cuándo cambiar |
|-----------|---------|--------|----------------|
| `M, N` | 16, 16 | Capacidad, tiempo de transmisión | Aumentar para más payload o finders más grandes |
| `cell_size` | 40 | Robustez vs. resolución | Bajar si pantalla pequeña o cámara lejos |
| `scheme` | BPSK_Manchester | Capacidad y robustez | 4ASK si canal limpio |
| `marker_cells` | 3 | Detección en receptor | Usar 7 para canal real |
| `pilot_period` | 4 | Calibración vs. capacidad | Bajar a 2 con mucho ruido |
| `symbol_duration_ms` | 100 | Sincronismo | Subir si cámara lenta |

### Parámetros de `detect_roi` — ajuste fino

| Parámetro | Default | Subir cuando | Bajar cuando |
|-----------|---------|-------------|-------------|
| `block_sz` (threshold) | `finder_px//3` | Iluminación no uniforme | Imagen muy nítida |
| `C_offset` | 10 | Mucho ruido / reflejos | Contraste bajo |
| `reproj_thresh` | 15 px | Ángulo > 25° | Precisión crítica |
| `min_squareness` | 0.65 | Ángulo grande | Solo frontal |
| `min_solid` | 0.72 | Finders muy ruidosos | Finders perfectos |

---

## 8. Pendientes y Próximas Celdas

### Fase B — Celdas por desarrollar

| Celda | Contenido | Funciones clave |
|-------|-----------|----------------|
| **B2** | Calibración fotométrica y AGC | `calibrate_frame()`, mapa local 2D de ganancia/offset, umbral adaptativo por región |
| **B3** | Sincronización | `estimate_rolling_shutter()`, `early_late_gate()`, correlación Gold del preámbulo |
| **B4** | Muestreo y demapeo robusto | media + mediana por celda, `sample_and_demap()`, ECC (Hamming/Reed-Solomon) |
| **B5** | Verificación cuantitativa | `benchmark_angles()`, BER vs. ángulo (0°,15°,30°), con/sin cada etapa de corrección |

### Mejoras identificadas pero no implementadas

- `marker_cells=7` como default en Fase A (mejor detección en canal real).
- Refinamiento iterativo del BR: aplicar H inicial, buscar features en zona BR del warped, refinar H.
- SIFT/ORB como fallback cuando < 3 finders son detectados por contornos.
- Campo de longitud de 2 bytes en el payload (evita depender de `rstrip('\x00')`).
- Calibración por región (2D) en Fase B para compensar gradientes de iluminación espacial.

---

*Generado automáticamente el 2026-05-29 a partir del historial de la sesión de trabajo.*
