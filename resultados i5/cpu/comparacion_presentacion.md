# Comparación CPU: presentación vs. PC actual — MSSPACEV-1M

| | Presentación | PC actual |
|---|---|---|
| CPU | i5-8300H (4 núcleos / 8 hilos, Coffee Lake) | i5-12450HX (4P + 4E núcleos / 12 hilos, Alder Lake) |
| Hilos usados | 8 | 12 |
| Ejecución | WSL2 + Docker, datos en HDD | WSL2 nativo (`--nodocker`), datos en disco de WSL |
| Código / parámetros | ParlayANN-ppopp24 `45587ba`, `artifact_eval.yaml` | idénticos |
| Corrida | 3–4 nov 2026 | `msspacev-1M_t12_20261111_162450` |

## Tiempo de construcción (s)

| Algoritmo | Presentación | PC actual | Mejora |
|---|---:|---:|---:|
| ParDiskANN | 398.9 | 125.8 | 3.17× |
| ParHCNNG | 377.7 | 142.2 | 2.66× |
| ParHNSW | 242.4 | 118.6 | 2.04× |
| ParPyNNDescent | 392.5 | 174.8 | 2.25× |

## Máximo QPS que cumple el umbral de recall@10

| Algoritmo | ≥0.95 antes | ≥0.95 ahora | Mejora | ≥0.98 antes | ≥0.98 ahora | Mejora |
|---|---:|---:|---:|---:|---:|---:|
| ParDiskANN | 16 617 | 32 079 | 1.93× | 3 235 | 7 920 | 2.45× |
| ParHCNNG | 9 745 | 19 936 | 2.05× | 5 185 | 10 509 | 2.03× |
| ParHNSW | 4 381 | 7 350 | 1.68× | 788 | 1 348 | 1.71× |
| ParPyNNDescent | 20 077 | 44 258 | 2.20× | 7 955 | 18 253 | 2.29× |

## Recall máximo

| Algoritmo | Presentación | PC actual |
|---|---:|---:|
| ParDiskANN | 0.992 | 0.993 |
| ParHCNNG | 0.997 | 0.998 |
| ParHNSW | 0.985 | 0.986 |
| ParPyNNDescent | 0.995 | 0.996 |

## Lectura

- **La calidad no cambia:** el recall máximo coincide (±0.001), así que los grafos son equivalentes.
  Lo que cambia es la velocidad del hardware, no el algoritmo.
- **Búsqueda ~1.7–2.5× más rápida**, consistente con más hilos (12 vs 8) y un núcleo más moderno.
- **Construcción ~2–3.2× más rápida.** Gana algo más que la búsqueda, probablemente porque el build
  time incluye guardar y recargar el índice (antes en HDD y vía Docker).
- **El orden entre algoritmos se mantiene en QPS** (PyNNDescent > DiskANN > HCNNG > HNSW con recall ≥ 0.95;
  HCNNG pasa a DiskANN con recall ≥ 0.98, igual que antes). En build, DiskANN pasa del último al segundo puesto.
- **No es un speedup de un algoritmo:** cambian a la vez CPU, número de hilos y entorno (Docker/HDD).
  Para el informe sirve como *nueva línea base en la misma máquina donde correrá CUDA*.

**Referencia para CUDA (ParDiskANN, 12 hilos):** build 125.8 s; 32 079 QPS con recall ≥ 0.95; 7 920 QPS con recall ≥ 0.98.
