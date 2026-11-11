# Linea base en CPU — ParlayANN sobre MSSPACEV-1M

Repite en **esta PC** el experimento de la presentacion, para luego compararlo
con la version CUDA en la misma maquina.

## Que se usa (todo de los autores, sin modificar los algoritmos)

| Componente | Version |
|---|---|
| Algoritmos C++ | `magdalendobson/ParlayANN-ppopp24`, commit `45587ba` (el mismo de la presentacion) |
| Benchmark | `magdalendobson/big-ann-benchmarks`, tag `ppopp-artifact-final` (commit `80e5bc47`) |
| Parametros | `artifact_eval.yaml` → seccion `msspacev-1M` (18 configuraciones de busqueda por algoritmo) |
| Protocolo | `run.py`, 5 repeticiones por configuracion, se queda con el mejor QPS |

**Unica diferencia con la presentacion:** no se usa Docker. El modulo
`_ParlayANNpy` se compila directamente en WSL con las mismas flags del
Dockerfile (`python/compile.sh`), y se ejecuta `run.py --nodocker`. Igual que en
la presentacion, se reemplaza `time.time()` por `time.perf_counter()` en el runner.

## Requisitos

- Windows con WSL2 + Ubuntu (ya instalado en esta PC).
- `g++`, `make`, `git`, `curl` dentro de Ubuntu (ya estan).
- ~2 GB libres en el disco de WSL. **No hace falta sudo ni Docker.**

## Como ejecutarlo

Abre una terminal de Ubuntu (`wsl` en PowerShell) y:

```bash
cd /mnt/c/Users/Usuario/Desktop/Uni/Paralela/cpu_baseline
```

**1. Instalacion (una sola vez, ~10 min):** clona los repos, crea Python 3.10,
compila el modulo C++ y descarga MSSPACEV-1M.

```bash
bash setup.sh
```

**2. Prueba rapida (~1 min)** para verificar que todo funciona:

```bash
bash run_benchmark.sh random-xs
```

**3. Experimento completo** con todos los hilos de la PC (12):

```bash
bash run_benchmark.sh msspacev-1M
```

**4. (Opcional) Con 8 hilos**, para compararlo con los numeros de la laptop de la presentacion:

```bash
bash run_benchmark.sh msspacev-1M 8
```

**5. (Opcional) Escalabilidad**: construir DiskANN con distintos hilos (trabajo futuro de la presentacion):

```bash
for t in 1 2 4 8 12; do ALGOS=ParDiskANN bash run_benchmark.sh msspacev-1M $t; done
```

## Resultados

Cada corrida crea `results/<dataset>_t<hilos>_<fecha>/` con:

- `resumen.txt` — tabla igual a la de la presentacion: build time, QPS con recall ≥ 0.95 / ≥ 0.98 y recall maximo.
- `all_results.csv` — todas las configuraciones (recall, QPS, comparaciones de distancia, build time, hilos).
- `msspacev-1M.png` — curva recall vs. QPS.
- `hardware.txt` y `log.txt` — equipo, versiones y salida completa.

Los grafos construidos quedan en `~/parlay/big-ann-benchmarks/data/indices/`.
El de DiskANN (`vamana/.../R64_L128_alpha1.2_threads*`) es el que puede cargar
la version CUDA para comparar la busqueda sobre **el mismo grafo**.

## Notas

- El build time incluye guardar y volver a cargar el indice, igual que en la presentacion.
- `plot.py` dibuja todos los resultados guardados del dataset; si corres varios
  numeros de hilos, usa `resumen.txt`, que filtra por hilos.
- WSL ve ~11 GB de RAM por defecto; MSSPACEV-1M usa bastante menos.
