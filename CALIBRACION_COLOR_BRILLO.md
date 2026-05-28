# Calibración de Color/Brillo con AGC Adaptativo
## Proyecto Final: Comunicaciones Digitales

---

## Resumen Ejecutivo

Se ha implementado un **sistema completo de calibración automática de color/brillo** con **Control Automático de Ganancia (AGC) adaptativo** y **decisor de umbral dinámico**. El sistema compensa automáticamente:

- ✓ Variaciones de iluminación ambiental
- ✓ Desbalance de blancos (cámara/pantalla/iluminación)
- ✓ Diferencias de brillo y contraste
- ✓ Respuesta de color no uniforme
- ✓ Cambios graduales de luminancia

---

## Componentes Principales

### 1. CalibrationMarkers
**Propósito:** Define puntos de referencia conocida para calibración

```python
# 4 Marcadores piloto en esquinas de grilla
- TL_Black: Negro (0, 0, 0)         @ filas 2-6, cols 2-6
- TR_White: Blanco (255, 255, 255)  @ filas 2-6, cols 58-62
- BL_Gray:  Gris (128, 128, 128)    @ filas 30-34, cols 2-6
- BR_Color: Rojo (255, 0, 0)        @ filas 30-34, cols 58-62
```

**Métodos:**
- `get_pilot_regions()`: Retorna definición de regiones piloto

---

### 2. ColorBrightnessCalibrator
**Propósito:** Calibra parámetros de color y brillo una sola vez

```python
calibrator = ColorBrightnessCalibrator(img_rectificada)
calibrator.calibrate()  # Calcula parámetros
img_cal = calibrator.apply_calibration(img)  # Aplica a nuevas imágenes
```

**Parámetros Calculados:**

| Parámetro | Fórmula | Uso |
|-----------|---------|-----|
| White Balance (R, G, B) | `G_mean / canal_mean` | Corrección de balance de color |
| Brightness Scale | Media de `esperado/observado` | Amplificación global |
| Contrast Scale | `rango_esperado / rango_observado` | Expansión dinámica |

**Ejemplo de Salida:**
```
✓ Calibración completa:
  - White Balance: R=0.922, G=1.000, B=1.000
  - Brightness Scale: 2.083
  - Contrast Scale: 6.774
```

---

### 3. AdaptiveAGC
**Propósito:** Control automático de ganancia suavizado

```python
agc = AdaptiveAGC(
    target_mean_luminance=127,   # Objetivo de luminancia
    smoothing_factor=0.7,         # Factor de suavizado (0-1)
    agc_window=15                 # Historial de muestras
)

for frame in video:
    frame_agc, luminance, gain = agc.process(frame)
```

**Algoritmo:**
1. Calcula luminancia promedio: `L_avg = 0.299R + 0.587G + 0.114B`
2. Determina ganancia: `gain = 127 / L_avg`
3. Limita rango: `gain = clip(gain, 0.3, 3.0)`
4. Suaviza exponencialmente: `gain_smooth = α*gain_new + (1-α)*gain_prev`
5. Aplica: `pixel' = pixel × gain_smooth`

**Características:**
- Adapta a cambios graduales de iluminación
- Evita cambios abruptos (suavizado exponencial)
- Mantiene luminancia centr  ada (objetivo = 127)

---

### 4. AdaptiveThresholdDecider
**Propósito:** Binarización robusta con umbral dinámico

```python
threshold_decider = AdaptiveThresholdDecider(
    base_threshold=127,
    local_window_size=31,
    agc_sensitivity=1.5
)

binary, threshold_used = threshold_decider.process(gray_img, luminance)
```

**Dos Métodos Combinados:**

#### Método 1: Umbral Global Adaptativo
```
umbral_ajustado = 127 + (L_actual - 127) × 0.3 × sensibilidad
```
- Ajusta umbral base según luminancia
- Sensibilidad ∈ [0.5, 3.0]
- Rápido y simple

#### Método 2: Umbralization Local Adaptativa
```
CLAHE(clipLimit=2.0) → AdaptiveThreshold(ventana=31×31)
```
- CLAHE: Contrast Limited Adaptive Histogram Equalization
- Compensa variaciones locales de iluminación
- Preserva detalles en regiones oscuras

**Combinación Final:**
```
binary_final = AND(global_threshold, local_threshold)
```
Maximiza robustez usando ambos métodos

---

## Flujo de Procesamiento Completo

```
┌─────────────────────────────┐
│  Imagen Rectificada (RGB)   │
│    1280×720 píxeles         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ [1] CALIBRACIÓN DE COLOR/BRILLO             │
│     Usa marcadores piloto para estimar:     │
│     • Balance de blancos (R, G, B)          │
│     • Factor de brillo                      │
│     • Factor de contraste                   │
│     ✓ Se ejecuta UNA SOLA VEZ               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ [2] APLICAR CALIBRACIÓN                     │
│     pixel' = (pixel × WB) × brillo × contr. │
│     Resultado: Imagen más uniforme          │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ [3] CONTROL AUTOMÁTICO DE GANANCIA (AGC)    │
│     • Mide luminancia actual                │
│     • Calcula ganancia para alcanzar 127    │
│     • Suaviza exponencialmente              │
│     ✓ Se adapta CADA FRAME (gradualmente)   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ [4] DECISOR DE UMBRAL ADAPTATIVO            │
│     • Ajusta umbral según luminancia        │
│     • Aplica umbralization local (CLAHE)    │
│     • Combina ambos métodos (AND)           │
│     Resultado: Binarización robusta         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Imagen Binarizada (B&N)    │
│  Lista para decodificación  │
└─────────────────────────────┘
```

---

## Uso Práctico

### Primera Ejecución (Calibración)

```python
from FaseB import ColorBrightnessCalibrator, AdaptiveAGC, AdaptiveThresholdDecider

# Calibración única
results = process_with_full_calibration(
    img_rectificada, 
    filas=36, 
    columnas=64,
    ancho_celda=20, 
    alto_celda=20
)

# Acceder a objetos calibrados
calibrator = results['calibrator']
agc = results['agc']
threshold_decider = results['threshold_decider']
```

### Frames Posteriores (Reutilizar Calibración)

```python
for frame in video_stream:
    # 1. Aplicar homografía (igual que antes)
    frame_rect = cv2.warpPerspective(frame, H, (1280, 720))
    
    # 2. Aplicar calibración (parámetros guardados)
    frame_cal = calibrator.apply_calibration(frame_rect)
    
    # 3. Aplicar AGC (actualiza estado interno)
    frame_agc, luminance, gain = agc.process(frame_cal)
    
    # 4. Convertir a escala de grises
    gray_agc = cv2.cvtColor(frame_agc, cv2.COLOR_RGB2GRAY)
    
    # 5. Aplicar decisor de umbral
    frame_bin, threshold_used = threshold_decider.process(gray_agc, luminance)
    
    # 6. Usar frame_bin para decodificación
    decode(frame_bin)
```

---

## Parámetros Ajustables

### ColorBrightnessCalibrator

| Parámetro | Ubicación | Rango | Defecto | Impacto |
|-----------|-----------|-------|---------|--------|
| Ubicación marcadores TL | CalibrationMarkers | Filas/cols | 2-6, 2-6 | Precisión calibración |
| Ubicación marcadores TR | CalibrationMarkers | Filas/cols | 2-6, 58-62 | Precisión calibración |
| Ubicación marcadores BL | CalibrationMarkers | Filas/cols | 30-34, 2-6 | Precisión calibración |
| Ubicación marcadores BR | CalibrationMarkers | Filas/cols | 30-34, 58-62 | Precisión calibración |

### AdaptiveAGC

| Parámetro | Rango | Defecto | Impacto |
|-----------|-------|---------|--------|
| `target_mean_luminance` | 0-255 | 127 | Luminancia objetivo |
| `smoothing_factor` | 0.0-1.0 | 0.7 | Mayor = más suave (cambios lentos) |
| `agc_window` | 5-30 | 15 | Historial para estadísticas |

**Recomendaciones:**
- `smoothing_factor = 0.7`: Cambios graduales, resistente a parpadeos
- `smoothing_factor = 0.9`: Muy suave, tarda más en adaptarse
- `smoothing_factor = 0.5`: Más reactivo, puede oscilar

### AdaptiveThresholdDecider

| Parámetro | Rango | Defecto | Impacto |
|-----------|-------|---------|--------|
| `base_threshold` | 0-255 | 127 | Umbral inicial |
| `local_window_size` | 5-63 (impar) | 31 | Tamaño ventana CLAHE |
| `agc_sensitivity` | 0.5-3.0 | 1.5 | Sensibilidad a luminancia |

**Recomendaciones:**
- `agc_sensitivity = 1.5`: Balance entre adaptación y estabilidad
- `agc_sensitivity = 0.5`: Menos sensible a variaciones
- `agc_sensitivity = 3.0`: Muy sensible, segue rápido cambios
- `local_window_size = 31`: Bueno para grillas uniformes

---

## Ejemplo de Salida

```
[Calibración de Color/Brillo]
  Balance de Blancos:
    R: 0.9224  (menos rojo, cámara/iluminación sesgo)
    G: 1.0000  (referencia)
    B: 1.0000  (balanceado)
  Factor de Brillo: 2.0826  (imagen oscura, amplifica)
  Factor de Contraste: 6.7735  (poco contraste, expande)

[Control Automático de Ganancia (AGC)]
  Luminancia Detectada: 83.89  (oscura, objetivo=127)
  Ganancia Aplicada: 1.5139  (amplifica ~1.5x)
  Desviación: -43.11  (cuánto falta para llegar a 127)

[Decisor de Umbral Adaptativo]
  Umbral de Binarización: 107  (ajustado dinámicamente)
  Ventana Local: 31 píxeles
```

**Interpretación:**
- Balance de blancos: Hay sesgo de color que se corrige
- Brillo/Contraste: Imagen subexponible, necesita ganancia
- AGC: Amplifica 1.5× para llevar luminancia a 127
- Umbral: Reducido a 107 porque la imagen es oscura

---

## Validación Visual

La función `process_with_full_calibration()` genera un panel de 6 sub-figuras:

1. **Original Rectificada**: Imagen de entrada con distorsiones
2. **Tras Calibración Color/Brillo**: Mejor balance y exposición
3. **Tras AGC**: Luminancia más uniforme
4. **Escala de Grises (Post-AGC)**: Preparada para binarización
5. **Umbral Adaptativo**: Imagen final binarizada
6. **Estadísticas**: Panel con todos los parámetros

---

## Características Avanzadas

### Suavizado Exponencial del AGC
```python
# Evita cambios abruptos cuando luminancia varía rápido
gain_smooth = α * gain_actual + (1 - α) * gain_anterior
```
- Con α=0.7: ~70% del nuevo, ~30% del anterior
- Resultado: cambios graduales y suavizados

### Combinación de Umbralization Global + Local
```python
# Global: Rápido, adapta a cambios generales
# Local: Lento pero robusto, maneja sombras locales
# AND: Solo píxeles que cumplen ambos criterios
```

### Límites de Ganancia [0.3, 3.0]
- Mínimo 0.3: Evita amplificar demasiado ruido
- Máximo 3.0: Evita amplificación excesiva
- Evita artefactos extremos

---

## Próximos Pasos (Sugerencias)

1. **Validación Real**
   - Probar con video real de transmisión
   - Ajustar `agc_sensitivity` según ruido ambiente

2. **Optimización**
   - Recolocar marcadores piloto si grilla cambia
   - Calibrar offline, aplicar online para velocidad

3. **Monitoreo**
   - Graficar historia de ganancias AGC
   - Detectar fallos de calibración

4. **Integración**
   - Exportar calibración entre frames
   - Persistir parámetros en archivo

---

## Archivo: FaseB.ipynb

El código completo está disponible en:
- **Celda 1**: Funciones base de homografía y rectificación
- **Celda 2**: Clases de calibración (ColorBrightnessCalibrator, AdaptiveAGC, etc.)
- **Celda 3**: Documentación y ejemplos
- **Celda 4**: Demostración ejecutable con imagen sintética

Ejecutar celdas en orden para cargar el sistema completo.

---

**Implementación: Mayo 2026**  
**Proyecto Final: Comunicaciones Digitales - PDS 2026-1**
