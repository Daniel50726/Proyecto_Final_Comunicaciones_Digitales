# Módem óptico espacio-temporal — PDS 2026-1

Proyecto final de **Procesamiento Digital de Señales**: transmisión de datos
entre una **pantalla** (transmisor) y una **cámara** (receptor) mediante una
grilla de macropíxeles en escala de grises.

```
[FASE A — TRANSMISOR]   texto → bits → símbolos → grilla → imagen PNG
[FASE B — RECEPTOR]     captura → ROI → calibración → sincronización → demapeo → texto
```

## Estructura

| Ruta | Contenido |
|------|-----------|
| `FaseA.ipynb` | **Fase A — transmisor** (notebook). Modulación BPSK-Manchester / 4-ASK, marcadores tipo QR, pilotos, preámbulo Gold, ensamblado de trama. |
| `receptor/` | **Fase B — receptor** (paquete `.py`). Pipeline modular B1–B4 + modo vídeo. Ver [`receptor/README.md`](receptor/README.md). |
| `run_receptor.py` | Receptor de cuadro único (CLI). |
| `benchmark_b5.py` | Evaluación cuantitativa de BER (ablación de correcciones). |
| `run_camera.py` | Modo vídeo en vivo (cámara o stream simulado). |
| `instrucciones_fase_*.md` | Requerimientos técnicos de cada fase. |
| `HISTORICO_PROYECTO.md` | Histórico de diseño y decisiones. |
| `frame_tx.png` | Trama de ejemplo generada por la Fase A. |

## Inicio rápido (receptor)

```bash
python -m pip install numpy opencv-python matplotlib

# decodificar la trama de ejemplo a 15°
python run_receptor.py --source sim --angle 15 --debug window

# end-to-end con texto + Reed-Solomon
python run_receptor.py --source sim --text "Hola PDS 2026" --ecc rs --angle 20
```

La documentación completa de la arquitectura, etapas, ECC y comandos del
receptor está en **[`receptor/README.md`](receptor/README.md)**.

## Configuración

`ModemConfig` (compartida TX/RX): `M=36, N=64, cell_size=20, marker_cells=7,
scheme="BPSK_Manchester"` → imagen canónica 1280×720 px. El layout de la trama
es **determinista**: el receptor lo reconstruye sin overhead de transmisión.
