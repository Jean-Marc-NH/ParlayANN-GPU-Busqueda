# ParlayANN-GPU-Busqueda

Aceleracion con CUDA de la busqueda en grafos de **ParlayANN**
(Manohar et al., *ParlayANN: Scalable and Deterministic Parallel Graph-Based
Approximate Nearest Neighbor Search Algorithms*, PPoPP 2024 —
[arXiv:2305.04359](https://arxiv.org/abs/2305.04359)).

ParlayANN construye y recorre grafos de vecinos aproximados **solo en CPU**
(multinucleo, sin locks y deterministico). En este proyecto:

1. Se reproduce la linea base en CPU con el codigo original de los autores.
2. Se implementa en **CUDA la beam search** sobre los mismos grafos, devolviendo
   **exactamente los mismos vecinos** que la CPU.
3. (Pendiente) Se pasa a la GPU la **construccion** de los grafos.

> Los resultados medidos todavia no se publican en este repositorio. Los
> programas los generan localmente (ver "Que deberia pasar").

---

## Contenido

| Carpeta | Que tiene |
|---|---|
| `cpu_baseline/` | Scripts para correr en WSL el artifact original de ParlayANN (linea base en CPU) |
| `cuda/` | Implementacion propia en CUDA: carga de datos, kernel de busqueda, perfilado y microbenchmark |

### De los autores (no se modifica)
- Algoritmos C++: [`magdalendobson/ParlayANN-ppopp24`](https://github.com/magdalendobson/ParlayANN-ppopp24), commit `45587ba`.
- Benchmark: [`magdalendobson/big-ann-benchmarks`](https://github.com/magdalendobson/big-ann-benchmarks), tag `ppopp-artifact-final`.
- Parametros: `artifact_eval.yaml`, seccion `msspacev-1M` (18 configuraciones de busqueda por algoritmo).
- Los grafos que usa la GPU son **los que construye ParlayANN en CPU**.

### Propio
- Scripts de `cpu_baseline/` (instalacion sin Docker ni sudo, ejecucion y resumen).
- Todo `cuda/`: lectores de archivos, kernel de beam search, validacion contra la CPU,
  perfilado por secciones y microbenchmark de distancias.

---

## Requisitos

| | |
|---|---|
| Sistema | Windows 10/11 + **WSL2 con Ubuntu** |
| CPU (linea base) | dentro de Ubuntu: `g++`, `make`, `git`, `curl` (no hace falta sudo ni Docker) |
| GPU | NVIDIA Ampere (`sm_86`, p. ej. RTX 3050) o Blackwell (`sm_120`, p. ej. RTX 5050) |
| Compilador GPU | **CUDA 12.8 o mas nuevo** + **Visual Studio 2022** (herramientas de C++) |
| Disco | ~2 GB en WSL (dataset + grafos) |

Dataset: **MSSPACEV-1M** (1 000 000 vectores int8 de 100 dimensiones, 29 316 consultas, k = 10).
Lo descarga `setup.sh`.

---

## Paso A — Linea base en CPU (WSL)

Detalle completo en [`cpu_baseline/LEEME.md`](cpu_baseline/LEEME.md).

```bash
cd /mnt/c/<ruta-del-repo>/cpu_baseline
bash setup.sh                         # una vez: clona, compila y descarga MSSPACEV-1M
bash run_benchmark.sh random-xs       # prueba rapida
bash run_benchmark.sh msspacev-1M     # experimento completo (todos los hilos)
```

- `setup.sh` clona los repos de los autores en `~/parlay`, crea un entorno Python 3.10
  con `uv`, compila el modulo C++ con las mismas flags del Dockerfile original y
  descarga el dataset.
- `run_benchmark.sh` corre los 4 algoritmos (DiskANN, HCNNG, PyNNDescent, HNSW) con
  `run.py --nodocker`: construye cada grafo y prueba las 18 configuraciones de busqueda
  (mejor de 5 repeticiones).
- Deja en `cpu_baseline/results/<dataset>_t<hilos>_<fecha>/` el `all_results.csv`,
  un `resumen.txt`, la curva recall–QPS y el log.
- **Los grafos quedan en `~/parlay/big-ann-benchmarks/data/indices/`**: la GPU usa esos.

**Importante:** correr enchufado, en modo de maximo rendimiento y sin otros programas
abiertos. Con el modo ahorro de energia los tiempos no sirven.

---

## Paso B — Busqueda en GPU (Windows)

```bat
cd cuda
build.bat
paso1_carga.exe
paso2_busqueda.exe
```

| Programa | Que hace |
|---|---|
| `paso1_carga.exe` | Lee vectores, consultas, ground truth y los 3 grafos; los sube a la GPU y verifica que todo quedo bien |
| `paso2_busqueda.exe` | Corre la busqueda en GPU con las 18 configuraciones sobre DiskANN, HCNNG y PyNNDescent y compara el recall con la CPU |
| `paso2_perfil.exe` | Igual que el anterior pero mide con `clock64()` cuantos ciclos va en cada seccion del kernel |
| `micro_distancias.exe` | Microbenchmark: compara calcular distancias con 1, 4, 8 o 32 hilos por vector |

Argumentos (`-` = valor por defecto):

```
paso2_busqueda.exe [carpeta_data] [csv_cpu] [algoritmo] [Ls]
```

- `carpeta_data`: carpeta `data` de big-ann-benchmarks vista desde Windows.
  Por defecto `\\wsl.localhost\Ubuntu\home\usuario\parlay\big-ann-benchmarks\data\`
  (**cambiar `usuario` / `Ubuntu` segun la maquina**).
- `csv_cpu`: el `all_results.csv` de la corrida de CPU del Paso A.
- `algoritmo` y `Ls`: para correr una sola configuracion (por ejemplo, para perfilar).
  En ese caso los resultados van a `resultados/paso2_parcial.csv`.

Ejemplo, solo DiskANN con Ls = 75:

```bat
paso2_perfil.exe - ..\cpu_baseline\results\msspacev-1M_t12_<fecha>\all_results.csv DiskANN 75
```

---

## Que deberia pasar

**`paso1_carga.exe`**
- `distancias GPU vs CPU: 210000 parejas, 0 diferentes`: la distancia de la GPU es
  **identica** a la de ParlayANN (aritmetica entera, sin redondeo).
- Para cada grafo: `grados distintos: 0` y `filas distintas: 0`.
- Al final: `PASO 1 OK: datos y grafos listos en la GPU`.

**`paso2_busqueda.exe`**
- Una tabla por algoritmo con recall GPU, recall CPU, QPS GPU, QPS CPU y speedup.
- La columna `igual` debe decir **SI en todas las filas** y al final
  `configuraciones con recall identico a la CPU: 54 de 54`.
  Si alguna dice NO, la GPU no esta reproduciendo la busqueda de la CPU.
- `desbordes` debe ser **0**. `cortas` puede ser mayor que 0: son consultas cuya frontera
  termina con menos de 10 elementos, y a la CPU le pasa lo mismo.
- El QPS de este paso es **preliminar**: solo mide el kernel y una corrida (ver Paso 5).

---

## Como funciona la busqueda en GPU (resumen)

**Un warp (32 hilos) resuelve una consulta**, siguiendo paso a paso `beam_search()` de
ParlayANN (`algorithms/utils/beamSearch.h`):

1. Tomar el primer punto de la frontera sin visitar y marcarlo.
2. Leer sus vecinos, descartar los ya vistos (filtro hash) y calcular distancias.
3. Ordenar los candidatos y mezclarlos con la frontera (`set_union`).
4. Recortar a `Ls` y aplicar el *cut* (1.35 × distancia del k-esimo).
5. Buscar el siguiente sin visitar; si no hay, terminar.

**Para que el resultado sea identico a la CPU** se copian tambien detalles del original
(marcados `IGUAL QUE PARLAYANN` en el codigo):
- La busqueda empieza siempre en el punto 0, con `cut = 1.35` y `k = 10`.
- Se descarta el punto de la base cuyo id es igual al **indice de la consulta**
  (asi lo hace el wrapper original).
- El filtro "ya visto" tiene perdidas y usa el mismo hash (`hash64_2`) y el mismo
  tamano; en paralelo se respeta el orden de escritura de la CPU (`__match_any_sync`).
- HCNNG y PyNNDescent tienen **vecinos repetidos** en sus grafos; la frontera se maneja
  como multiconjunto, igual que `std::set_union` / `std::set_difference`.
- Distancias enteras con `__dp4a`: ‖q−p‖² = ‖q‖² + ‖p‖² − 2·q·p.

**Optimizaciones** (justificadas con `paso2_perfil.exe` y `micro_distancias.exe`):
- Cada vector lo leen **8 hilos juntos** (una lectura de 128 bytes) y cada hilo lleva
  4 distancias, asi hay 16 vectores pedidos a memoria a la vez por warp.
- La fila de vecinos y las casillas del filtro se leen **todas de una vez** en vez de
  una ida a memoria por cada trozo de 32 vecinos.
- Se reusan arreglos de memoria compartida para que entren mas warps por SM, y el
  tamano de bloque respeta el limite de registros del kernel.

---

## Usarlo en otra maquina (por ejemplo RTX 5050 + i7 de 13.a generacion)

1. **Repetir el Paso A en esa maquina.** El speedup siempre se compara contra la CPU
   de la misma maquina.
2. `build.bat` ya compila para `sm_86` y `sm_120` (y deja PTX para GPUs mas nuevas).
   Requiere CUDA 12.8+.
3. Si Visual Studio no es la edicion *Community*, ajustar la ruta de `vcvars64.bat`
   en `build.bat`.
4. Pasar la carpeta de datos y el CSV de CPU como argumentos (ver arriba).
5. Anotar el modelo exacto de CPU y GPU, el numero de hilos y el modo de energia.

---

## Pasos futuros

### Comparacion 1 — Busqueda (CPU vs GPU sobre los mismos grafos)

**Paso 3 — HNSW en GPU.**
HNSW guarda su grafo en otro formato y con varios niveles. Falta:
- un lector de su archivo (`HNSW.hpp`: cabecera, nivel y vecinos de cada punto por
  nivel, puntos de entrada);
- la bajada por los niveles superiores (busqueda voraz) antes de la beam search del
  nivel 0 con `ef = Ls`. El nivel 0 reutiliza el kernel actual;
- respetar que en HNSW el parametro `visit` limita **distancias calculadas**
  (`limit_eval`), no nodos visitados.

Criterio: recall identico a la CPU en sus 18 configuraciones.

**Paso 4 — Validacion vecino por vecino.**
El benchmark guarda en HDF5 los vecinos que devolvio la CPU en cada configuracion.
Comparar id por id con los de la GPU (lo esperado es una coincidencia de ~100 %).
Hoy la validacion se hace por recall identico, que es una prueba fuerte pero indirecta.

**Paso 5 — Medicion final.**
Para que la comparacion con la CPU sea justa:
- **mejor de 5 repeticiones**, igual que big-ann-benchmarks;
- tiempo **de punta a punta**: copiar consultas a la GPU + kernel + traer resultados
  (ademas del tiempo solo de kernel);
- mismas condiciones de energia que la corrida de CPU;
- tabla por algoritmo (QPS con recall ≥ 0.95 y ≥ 0.98, speedup) y curva recall–QPS
  CPU vs GPU en el mismo grafico.

**Mejora opcional — filtro sin limpieza.**
El filtro "ya visto" crece con Ls² y hoy se limpia en cada consulta; con Ls grandes eso
satura la cache L2. Se puede guardar en cada casilla el numero de consulta junto con el
id, asi no hace falta limpiar y el resultado sigue siendo exacto.

### Comparacion 2 — Construccion de grafos en GPU

Se mide tiempo de construccion, **calidad** (recall–QPS del grafo construido en GPU,
buscado con el codigo de CPU de los autores y con nuestro kernel) y **determinismo**
(hash del archivo del grafo igual en varias corridas). El grafo se guarda en el mismo
formato de ParlayANN para poder cruzar las validaciones.

Orden propuesto (de menor a mayor riesgo):

1. **DiskANN** (Algoritmo 3 del paper): insercion por lotes con *prefix doubling*
   (lotes 1, 2, 4, … hasta 0.02·n). Por lote: beam search de cada punto nuevo (reusa el
   kernel), *prune* con α = 1.2 y R = 64 (un warp por punto), y aristas inversas con
   `cub::DeviceRadixSort` en vez del semisort, sin atomicos para que sea deterministico.
2. **HNSW**: mismo esquema incremental, repetido por nivel; reutiliza casi todo DiskANN.
3. **HCNNG**: 50 arboles de clustering con hojas de ≤ 1000 puntos; en cada hoja, 10
   vecinos exactos (calculo denso, ideal para GPU) y un MST con grado maximo 3 (un warp
   por hoja, miles de hojas en paralelo). Al final, union de aristas con ordenamiento.
4. **PyNNDescent**: arboles de hiperplanos + rondas de "vecinos de vecinos" hasta
   converger. Es el mas riesgoso: necesita mucha memoria (hay que procesarlo por partes
   en 6–8 GB) y las versiones existentes en GPU usan atomicos (no deterministicas).

### Otros
- **Escalabilidad en CPU**: construir con 1, 2, 4, 8 y 12 hilos
  (`ALGOS=ParDiskANN bash run_benchmark.sh msspacev-1M <hilos>`).
- **Segunda maquina**: repetir todo en RTX 5050 + i7 de 13.a generacion.

---

## Referencias

- M. D. Manohar, Z. Shen, G. E. Blelloch, L. Dhulipala, Y. Gu, H. V. Simhadri, Y. Sun.
  *ParlayANN: Scalable and Deterministic Parallel Graph-Based Approximate Nearest
  Neighbor Search Algorithms*. PPoPP 2024. [arXiv:2305.04359](https://arxiv.org/abs/2305.04359)
- Artifact C++: https://github.com/magdalendobson/ParlayANN-ppopp24
- Framework de evaluacion: https://github.com/magdalendobson/big-ann-benchmarks
