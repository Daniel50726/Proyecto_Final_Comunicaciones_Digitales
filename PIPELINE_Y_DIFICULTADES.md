# Módem óptico espacio-temporal — Pipeline completo y dificultades

> Documento técnico del proyecto final de **Procesamiento Digital de Señales
> (PDS) 2026-1**. Explica de punta a punta cómo funciona el enlace
> **pantalla → cámara** y, sobre todo, **las dificultades reales** que aparecieron
> al pasar de la simulación al hardware y cómo se resolvieron.

---

## 1. Idea general

Se transmiten datos modulando una **grilla de macropíxeles** en escala de grises
que se muestra en una pantalla y se captura con una cámara web. El sistema se
desarrolló en tres fases:

| Fase | Qué hace | Forma |
|------|----------|-------|
| **A** | Transmisor: texto → bits → símbolos → grilla → imagen | Notebook `FaseA.ipynb` |
| **B** | Receptor de cuadro único: foto → texto | Paquete `.py` `receptor/` |
| **C** | Tiempo real multi-cuadro: vídeo → mensaje completo | `tx.py` / `rx.py` |

La configuración compartida (`ModemConfig`): grilla **36×64** celdas de **20 px**
→ imagen canónica **1280×720**, marcadores de **7 celdas**, 1 piloto cada 4
celdas. El layout de la trama es **determinista**: el receptor lo reconstruye sin
transmitirlo.

---

## 2. Cadena de transmisión (TX)

```
texto
  │ codificación de fuente: UTF-8 → bytes
  ▼
[Fase C] protocolo de trama: cabecera (tipo/seq/total/long) + datos
  │
  ▼ Reed-Solomon (GF 2⁸): + nsym bytes de paridad por palabra-código
  ▼ scrambling: XOR con PRBS determinista (blanqueo de energía)
  │
  ▼ bytes → bits → símbolos
  │   BPSK-Manchester: bit→[0,255] ó [255,0]  (0.5 bit/celda, DC balanceado)
  │   4-ASK Gray:      dibit→{0,85,170,255}    (2 bits/celda)
  ▼
ensamblado de la grilla (frame_builder):
  canvas gris 128 → pilotos → preámbulo Gold → payload → marcadores (QR-like)
  │
  ▼
[Fase C] ventana a pantalla completa, fondo negro, ciclo [SYNC, DATA…, EOM]
```

**Piezas clave del TX**

- **Marcadores tipo QR** (3 finders en las esquinas TL/TR/BL) con anillo oscuro +
  separador blanco → permiten localizar y rectificar la pantalla.
- **Preámbulo Gold** (secuencia pseudoaleatoria n=5, L=31) en las primeras celdas
  de datos → sincronización de trama por correlación.
- **Pilotos** de valor conocido (0/255 en BPSK; 0/85/170/255 en 4ASK) → permiten
  calibrar la respuesta del canal.
- **Layout determinista**: TX y RX calculan `compute_frame_layout(config)`; no se
  transmite el mapa de celdas.

---

## 3. Pipeline de recepción (RX) — las 4 etapas de Fase B

El receptor es un pipeline modular desacoplado (`PipelineStage` →
`PipelineContext` → `ReceiverPipeline`), reutilizado tanto en cuadro único como
en tiempo real.

### B1 — Detección de ROI y rectificación (`stages/roi.py`)
1. **Normalización de iluminación**: flat-field (divide por la iluminación
   estimada) + CLAHE → aplana gradientes de luz.
2. **Binarización adaptativa** + búsqueda de contornos por **jerarquía**
   (`RETR_TREE`): los finders son anillos cuadrados, sólidos y con hueco interior.
3. **Homografía de perspectiva** con las 4 esquinas del anillo interior de cada
   finder (12 puntos) + **refinamiento sub-píxel** (`cornerSubPix`).
4. `warpPerspective` → vista cenital canónica 1280×720.

### B2 — Calibración fotométrica + AGC (`stages/calibration.py`)
Invierte el canal `y = a·x + b` usando los pilotos:
- **BPSK**: calibración lineal (global escalar o mapa 2D `a(x,y)`, `b(x,y)` que
  compensa gradientes espaciales / rolling shutter residual).
- **4ASK**: calibración **no lineal por LUT** (ver §5.4) que invierte la gamma.

### B3 — Sincronización (`stages/sync.py`)
- **Sincronización de trama**: correlación circular del preámbulo con la secuencia
  Gold esperada (pico ≥ 0.7 ⇒ grilla bien mapeada; lag = 0 ⇒ sin permutación).
- **Rolling shutter**: se diagnostica como deriva de brillo por fila (la corrige
  el mapa 2D de B2).
- `early_late_gate`: recuperación de reloj de símbolo, para el modo vídeo.

### B4 — Muestreo, demapeo y ECC (`stages/demap.py`)
1. **Muestreo robusto**: media del centro del macropíxel + filtro de mediana
   (suprime reflejos puntuales), **vectorizado** (un `reshape`).
2. **Demapeo**: símbolos → bits (BPSK por umbral; 4ASK por nivel más cercano).
3. **De-scrambling** + **Reed-Solomon**: corrige errores residuales del canal.
4. bytes → texto.

---

## 4. Tiempo real — protocolo multi-cuadro (Fase C)

Para mensajes que no caben en un cuadro (≥ 500 caracteres):

- **Protocolo de trama** (`receptor/protocol.py`): secuencia
  `[SYNC, DATA₀ … DATAₙ₋₁, EOM]`. Cada cuadro lleva, bajo RS, una cabecera con
  `tipo / nº de secuencia / total / longitud`.
- **Aceptación por validez RS**: un cuadro se integra **solo si RS decodifica sin
  fallos**. Los cuadros desgarrados (rolling shutter) o borrosos fallan RS y se
  descartan automáticamente → **no hace falta un reloj de símbolo preciso**.
- **Reensamblado** por número de secuencia hasta tener `[0..total-1]` o ver EOM;
  repetidos se ignoran, perdidos se recuperan en la siguiente vuelta del TX.
- **Cámara sin latencia** (`receptor/camera.py`): un **hilo** mantiene solo el
  último cuadro → el bucle nunca procesa cuadros viejos ni acumula cola.

Métricas al completar (rúbrica): **tiempo de transmisión** (< 10 s) y **BER**
(< 10⁻⁴), medidos contra `mensaje.txt`.

---

## 5. Dificultades encontradas y cómo se resolvieron

Esta es la parte central: cada problema apareció al confrontar la teoría con el
hardware real.

### 5.1 La homografía recortaba el borde del frame (Fase B)
**Síntoma:** los finders pegados al borde de la imagen quedaban cortados tras
rectificar. **Causa:** `findContours` no traza bien contornos que tocan el borde
de la imagen. **Solución:** añadir un borde (`copyMakeBorder`) antes de detectar y
componer la homografía con la traslación del padding.

### 5.2 La calibración detenía el pipeline en fotos reales
**Síntoma:** con fotos reales la calibración "fallaba" y cortaba todo, aunque la
imagen era decodificable. **Causa:** el umbral de éxito (`residual < 40`) estaba
afinado con datos sintéticos limpios; las fotos reales tienen más blur/ruido.
**Solución:** la calibración pasó a ser de **mejor esfuerzo** (solo exige ser
invertible) y emite una advertencia; los jueces reales de calidad son B3/B4.

### 5.3 El procesamiento en tiempo real iba a 0.4 fps
**Síntoma:** el receptor "se quedaba pegado", ~23 cuadros en 60 s. **Causa:** la
normalización de iluminación hacía un `GaussianBlur` de sigma ≈ 140 a 1280×720 →
**2.5 s por cuadro**. **Solución (≈ 50×):**
- Estimar la iluminación en una imagen **reducida** y reescalar (2484 → 9 ms).
- **Muestreo vectorizado** de toda la grilla con un `reshape` (56 → ~1 ms).
- Calibración **global** por defecto en tiempo real (60 → ~5 ms).
→ ciclo completo ~52 ms (**~19 fps**).

### 5.4 La interfaz de cámara en Windows
- **DSHOW** lanzaba excepción / no capturaba por índice en la cámara del usuario,
  y **MSMF** no exponía exposición/foco/WB (`cap.get` = −1). **Solución:**
  `open_camera` con **fallback automático** de backend y lectura blindada
  (`cv2.error` capturado en el hilo).
- **Bug `_step >= minstep` de MSMF:** ocurría al cambiar la resolución *después*
  de un `read()`. **Solución:** fijar la resolución **antes** del primer `read()`,
  y reintentar con resolución nativa si falla.
- No poder desactivar los automáticos **no es bloqueante**: la calibración por
  software compensa las variaciones (la rúbrica lo permite).

### 5.5 Detectaba el ROI pero no decodificaba nada (Fase C)
**Síntoma:** `gold ≈ 1.0` en cuadros limpios (la grilla se mapeaba perfecto) pero
**RS fallaba siempre**. **Diagnóstico** (gracias al modo `--diag` que muestra
`reproj | goldPeak | RSfail`): el `reproj` era 3-6 px en cámara real (vs < 2 en
simulación). El preámbulo (celdas superiores, ancladas a los finders) salía bien,
pero las celdas **lejanas del payload** (hacia la esquina BR, sin finder y por
tanto extrapolada) se muestreaban mal → demasiados errores → RS fallaba.
**Causa raíz:** las esquinas de los finders venían de `approxPolyDP`, imprecisas
con algo de blur. **Solución:** **refinamiento sub-píxel** (`cornerSubPix`) de las
12 esquinas → reproj 1.5 → **0.3 px**; cuadros DATA con RS ok pasaron de 0 a 20/20.

### 5.6 4ASK no funcionaba en el canal real — la gamma
**Síntoma:** 4ASK no decodificaba (BPSK sí). **Causa confirmada:** la respuesta
**no lineal (gamma ≈ 2.2)** del monitor + la cámara. Los niveles 0/85/170/255 son
lineales en código, pero tras la gamma quedan **mal espaciados**: el 85 se capta
en ≈ 23, casi pegado al 0. La calibración lineal (que solo usaba pilotos 0/255)
fija los extremos pero **descoloca los niveles medios**. En simulación lineal
4ASK daba 15/15; con gamma 2.2, **0/15** (BPSK seguía 15/15 porque solo usa los
extremos).
**Solución:** **pilotos de 4 niveles** en 4ASK (0/85/170/255) + **calibración no
lineal por LUT** que, con esos 4 puntos de referencia, construye una tabla que
invierte la gamma y recoloca los niveles equiespaciados. Resultado: 4ASK **0/15 →
15/15** bajo gamma, sin afectar BPSK. Bonus: 4ASK transporta ~233 bytes/cuadro
(3.2× más que BPSK).

### 5.7 4ASK fallaba la detección de finders — el relleno sólido
**Síntoma:** con 4ASK costaba mucho identificar los finders de las esquinas.
**Causa:** en BPSK-Manchester **cada bit produce una transición 0↔255**, así que
el grid nunca tiene regiones sólidas (DC balanceado) y los finders destacan. En
4ASK, un mensaje corto deja cientos de bytes de relleno `0x00` → **~75 % de la
grilla en negro sólido**, que (a) se conecta con los anillos de los finders a
través del separador difuminado y (b) oscurece la imagen, desestabilizando la
auto-exposición. **Solución:** **scrambling / blanqueo de energía** — XOR del flujo
de bytes con una secuencia pseudoaleatoria determinista, integrado dentro de
`rs_encode_payload`/`rs_decode_payload`. Es involutivo y **no mueve los errores
entre bytes** (preserva la corrección de ráfagas de RS). El grid pasa a parecer
ruido moteado: celdas casi-negras en 4ASK **75 % → 28 %**, finders siempre
aislados y brillo medio estable.

---

## 6. Decisiones de diseño transversales

- **Manchester garantiza DC = 128** → robustez y, como efecto colateral, ausencia
  de regiones sólidas (clave para la detección).
- **Reed-Solomon** (orientado a byte) se eligió porque los errores del canal son
  **en ráfaga** (espaciales: manchas, esquina BR), y es el ECC de los códigos QR
  — coherente con los finders. La validez RS por cuadro se usa además como **gate
  de aceptación** en tiempo real.
- **Layout determinista** → cero overhead de señalización.
- **BPSK vs 4ASK:** BPSK es más robusto (solo extremos, inmune a gamma); 4ASK
  triplica la capacidad pero exige calibración de gamma y buena luz/foco.

---

## 7. Resultados

- **Fase B:** decodifica fotos estáticas a 0°/15°/30°; la ablación cuantitativa
  (`benchmark_b5.py`) muestra que la BER mejora al activar perspectiva,
  calibración y compensación espacial.
- **Fase C:** transmisión en tiempo real de **500 caracteres**, reensamblados por
  protocolo, con **tiempo < 10 s** y **BER < 10⁻⁴** verificados contra
  `mensaje.txt`. Funcional en **BPSK y 4ASK**.

---

## 8. Mapa de archivos

```
FaseA.ipynb            Transmisor Fase A (notebook)
mensaje.txt            Texto a transmitir en tiempo real
tx.py / rx.py          Transmisor / receptor en tiempo real (Fase C)
generate_frame.py      Genera una trama PNG (cuadro único)
run_receptor.py        Receptor de cuadro único
run_camera.py          Modo vídeo (cámara o stream simulado)
benchmark_b5.py        Evaluación cuantitativa de BER
receptor/
  config, layout, modulation, preamble, channel_coding, frame_builder
  protocol, camera, video, debug_viz, simulate, pipeline
  stages/  roi(B1) · calibration(B2) · sync(B3) · demap(B4)
```

Documentación de uso y comandos: **`receptor/README.md`** y **`README.md`**.

---

*Generado como cierre técnico del proyecto — PDS 2026-1.*
