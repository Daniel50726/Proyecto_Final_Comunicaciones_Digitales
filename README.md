# Módem óptico espacio-temporal — PDS 2026-1

Proyecto final de **Procesamiento Digital de Señales**: transmisión de datos
entre una **pantalla** (transmisor) y una **cámara** (receptor) mediante una
grilla de macropíxeles en escala de grises.

```
[FASE A — TRANSMISOR]   texto → bits → símbolos → grilla → imagen PNG
[FASE B — RECEPTOR]     captura → ROI → calibración → sincronización → demapeo → texto
[FASE C — TIEMPO REAL]  secuencia de cuadros (pantalla→cámara) con protocolo SYNC/DATA/EOM
```

## Estructura

| Ruta | Contenido |
|------|-----------|
| `FaseA.ipynb` | **Fase A — transmisor** (notebook). Modulación BPSK-Manchester / 4-ASK, marcadores tipo QR, pilotos, preámbulo Gold, ensamblado de trama. |
| `receptor/` | **Fase B/C — receptor** (paquete `.py`). Pipeline modular B1–B4, modo vídeo, protocolo y cámara. Ver [`receptor/README.md`](receptor/README.md). |
| `tx.py` / `rx.py` | **Fase C — transmisión multi-cuadro en tiempo real** (pantalla / cámara). |
| `mensaje.txt` | Texto a transmitir en Fase C (lo leen `tx.py` y `rx.py`). |
| `generate_frame.py` | Transmisor de cuadro único: genera una trama PNG (con/sin ECC). |
| `run_receptor.py` | Receptor de cuadro único (CLI). |
| `benchmark_b5.py` | Evaluación cuantitativa de BER (ablación de correcciones). |
| `run_camera.py` | Modo vídeo en vivo (cámara o stream simulado). |
| `instrucciones_fase_*.md` | Requerimientos técnicos de cada fase. |
| `HISTORICO_PROYECTO.md` | Histórico de diseño y decisiones. |
| `frame_tx.png` | Trama de ejemplo generada por la Fase A. |

## Inicio rápido

```bash
python -m pip install numpy opencv-python matplotlib
```

### Fase C — transmisión en tiempo real (lo principal)
```bash
# 1) edita mensaje.txt con el texto a transmitir
# 2) en la máquina-pantalla:
python tx.py
# 3) en la máquina-cámara (apuntando a la pantalla, ≥50 cm):
python rx.py --cam-id 0
```
Al completarse, `rx.py` imprime el **tiempo de transmisión** y el **BER** (medidos
contra `mensaje.txt`). `--scheme`/`--nsym` deben coincidir en ambos (defaults ya
coinciden). Detalles y ajustes (`--hold-ms`, `--diag`, `--calib`, `--backend`) en
[`receptor/README.md`](receptor/README.md).

### Fase B — cuadro único / foto estática
```bash
# decodificar la trama de ejemplo (Fase A, sin ECC) a 15°
python run_receptor.py --source sim --angle 15 --ecc none --debug window

# generar una trama propia con Reed-Solomon para mostrar y fotografiar
python generate_frame.py --text "Hola Mundo! PDS 2026" --ecc rs --nsym 16 --out frame_rs.png
python run_receptor.py --source image --path foto.png --ecc rs --nsym 16
```

La documentación completa de la arquitectura, etapas, ECC y comandos está en
**[`receptor/README.md`](receptor/README.md)**.

## Configuración

`ModemConfig` (compartida TX/RX): `M=36, N=64, cell_size=20, marker_cells=7,
scheme="BPSK_Manchester"` → imagen canónica 1280×720 px. El layout de la trama
es **determinista**: el receptor lo reconstruye sin overhead de transmisión.
