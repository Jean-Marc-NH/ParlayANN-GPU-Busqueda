# ParlayANN-GPU-Busqueda

Aceleracion con CUDA de la busqueda en grafos de **ParlayANN**
(Manohar et al., *ParlayANN: Scalable and Deterministic Parallel Graph-Based
Approximate Nearest Neighbor Search Algorithms*, PPoPP 2024 —
[arXiv:2305.04359](https://arxiv.org/abs/2305.04359)).

ParlayANN construye y recorre grafos de vecinos aproximados **solo en CPU**
(multinucleo, sin locks y deterministico). En este proyecto:

1. Se reproduce la linea base en CPU con el codigo original de los autores.
2. **Comparacion 1 (lista para correr):** se implementa en **CUDA la busqueda** de los
   4 algoritmos (DiskANN, HCNNG, PyNNDescent y HNSW) sobre los mismos grafos,
   devolviendo **los mismos vecinos** que la CPU, y se compara el rendimiento.
3. **Comparacion 2 (pendiente):** pasar a la GPU la **construccion** de los grafos.

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

## Paso B — Comparacion 1: busqueda en GPU (Windows)

### De una sola vez

Con el Paso A ya hecho, desde PowerShell o `cmd`:

```bat
cd cuda
comparacion1.bat
```

Hace todo en orden y se detiene si algo falla:

| # | Que corre | Para que |
|---|---|---|
| 1 | `build.bat` | Compila todos los programas |
| 2 | `paso1_carga.exe` | Verifica que datos y grafos llegan identicos a la GPU |
| 3 | `comparacion1.exe` | Mide los **4 algoritmos x 18 configuraciones** en GPU (mejor de 5): QPS solo kernel y QPS de punta a punta, recall y vecinos de cada configuracion |
| 4 | `reporte_comparacion1.py` (en WSL) | **Paso 4:** validacion vecino por vecino contra los HDF5 de la CPU. **Paso 5:** tablas y graficas CPU vs GPU |

Resultado en `cuda/resultados/comparacion1/`:
`resumen.md` (tablas listas para el informe), `configuraciones.csv`, `validacion.csv`,
`recall_qps.png` y `speedup.png`.

Duracion aproximada: 15–30 minutos (las configuraciones con `Ls` grande son las mas
lentas). Correr enchufado, en maximo rendimiento y sin otros programas abiertos,
**igual que la corrida de CPU**.

Si la carpeta de datos no es la de por defecto:
`comparacion1.bat "\\wsl.localhost\<distro>\home\<usuario>\parlay\big-ann-benchmarks\data"`.
Si la distro de WSL no se llama `Ubuntu`, cambiar `DISTRO` al inicio de `comparacion1.bat`.

### Como se mide

| | CPU | GPU |
|---|---|---|
| Grafo | construido por ParlayANN | **el mismo archivo** |
| Consultas / k | 29 316 / 10 | 29 316 / 10 |
| Configuraciones | 18 por algoritmo (`artifact_eval.yaml`) | las mismas |
| Repeticiones | calentamiento + mejor de 5 | calentamiento + mejor de 5 |
| Tiempo | `algo.query()` en big-ann-benchmarks | **punta a punta**: copiar consultas a la GPU + normas + kernel + traer ids y distancias (reloj de la CPU). Tambien se reporta **solo kernel** (`cudaEvent`) |
| Recall | `compute_recall_with_distance_ties` | la misma formula (`evaluacion.cpp`) |

La carga del grafo y de la base no se cuenta en ninguno de los dos lados.
Los QPS de la CPU se toman de sus HDF5 (`best_search_time`), configuracion por
configuracion.

### Programas sueltos

```bat
cd cuda
build.bat
paso1_carga.exe
paso2_busqueda.exe
paso3_hnsw.exe
```

| Programa | Que hace |
|---|---|
| `paso1_carga.exe` | Lee vectores, consultas, ground truth y los 3 grafos; los sube a la GPU y verifica que todo quedo bien |
| `paso2_busqueda.exe` | Busqueda en GPU con las 18 configuraciones sobre DiskANN, HCNNG y PyNNDescent, comparando el recall con la CPU (1 corrida, solo kernel) |
| `paso2_perfil.exe` | Igual que el anterior pero mide con `clock64()` cuantos ciclos va en cada seccion del kernel |
| `paso3_hnsw.exe` | Lee el grafo HNSW (varios niveles), lo sube a la GPU y corre sus 18 configuraciones comparando el recall con `ParHNSW` de la CPU |
| `comparacion1.exe` | La medicion final de los 4 algoritmos (ver arriba) |
| `micro_distancias.exe` | Microbenchmark: compara calcular distancias con 1, 4, 8 o 32 hilos por vector |

Archivos del codigo:

| Archivo | Contenido |
|---|---|
| `comun.h` | Constantes, rutas por defecto, `CHECK_CUDA` |
| `lectores.*` | Lectura de `.i8bin`, grafos de ParlayANN y ground truth |
| `datos_gpu.*` | Subida de vectores (rellenados a 128 bytes) y grafos a la GPU |
| `distancia.cuh` | Distancia exacta con `__dp4a` |
| `utiles_gpu.cuh` | Hashes de parlaylib (`hash64`, `hash64_2`) y funciones comunes de los kernels |
| `busqueda.*` | Kernel de busqueda para DiskANN, HCNNG y PyNNDescent |
| `hnsw.h`, `hnsw_lector.cpp`, `busqueda_hnsw.cu` | Formato, lectura, subida y kernel de busqueda de HNSW |
| `evaluacion.*` | Recall con empates (igual que big-ann-benchmarks) y lectura del CSV de la CPU |
| `comparacion1.cu` | Medicion final: mejor de N, solo kernel y punta a punta, guarda los vecinos |
| `reporte_comparacion1.py` | Validacion vecino por vecino (paso 4) y tablas/graficas (paso 5) |
| `comparacion1.bat` | Corre toda la comparacion 1 en orden |

Argumentos (`-` = valor por defecto):

```
paso2_busqueda.exe [carpeta_data] [csv_cpu] [algoritmo] [Ls]
paso3_hnsw.exe     [carpeta_data] [csv_cpu] [Ls]
comparacion1.exe   [carpeta_data] [csv_cpu] [algoritmo] [Ls] [corridas]
```

En `comparacion1.exe` el algoritmo va con el nombre de la CPU (`ParDiskANN`,
`ParHCNNG`, `ParPyNNDescent`, `ParHNSW`). Por ejemplo, una prueba rapida de HNSW
con una sola corrida: `comparacion1.exe - - ParHNSW 75 1`.

- `carpeta_data`: carpeta `data` de big-ann-benchmarks vista desde Windows.
  Por defecto `\\wsl.localhost\Ubuntu\home\usuario\parlay\big-ann-benchmarks\data\`
  (**cambiar `usuario` / `Ubuntu` segun la maquina**).
- `csv_cpu`: el `all_results.csv` de la corrida de CPU del Paso A.
- `algoritmo` y `Ls`: para correr una sola configuracion (por ejemplo, para perfilar).
  En ese caso los resultados van a `resultados/paso2_parcial.csv` (o `paso3_parcial.csv`).

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

**`paso3_hnsw.exe`**
- Primero muestra el grafo: `m`, `efc`, grado maximo por nivel, cuantos nodos hay en
  cada nivel y el punto de entrada. Si el archivo no tiene el formato esperado, se
  detiene con un mensaje (por ejemplo `sobran N bytes al final`).
- Despues la misma tabla que el paso 2 para `ParHNSW`. Lo esperado es
  `configuraciones con recall identico a la CPU: 18 de 18` y `desbordes 0`.
- Si alguna configuracion sale `NO`, el programa imprime los recalls de la CPU para
  comparar a mano (ver "Notas sobre HNSW").

**`comparacion1.exe`**
- Una tabla por algoritmo: recall, `=CPU` (SI/NO), QPS solo kernel, QPS de punta a punta,
  QPS de la CPU y los dos speedups.
- Al final: `configuraciones con recall identico a la CPU: 72 de 72` y `desbordes 0`.
- Tambien: `configuraciones deterministas (calentamiento == ultima corrida): 72 de 72`.
  Si alguna sale `NO DETERMINISTA`, hay estado que se arrastra entre lanzamientos del
  kernel (asi se detecto un error con las epocas de las tablas, ya corregido).
- El QPS de punta a punta siempre es menor que el de solo kernel; la diferencia es el
  costo de mover datos entre CPU y GPU, y pesa mas en las configuraciones mas rapidas.

**`reporte_comparacion1.py`**
- Validacion: para cada algoritmo, cuantas configuraciones tienen recall identico y en
  que porcentaje de las consultas la GPU devuelve **los mismos 10 vecinos** que la CPU.
  Lo esperado es ~100 % (en HNSW pueden aparecer diferencias por empates, ver abajo).
- La tabla de "mejor QPS que cumple recall ≥ 0.90 / 0.95 / 0.98" para CPU, GPU de punta
  a punta y GPU solo kernel, con los speedups.
- Las graficas recall–QPS (CPU vs GPU) y speedup por configuracion.

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

### Notas sobre HNSW (`busqueda_hnsw.cu`)

HNSW busca distinto que los otros tres (`HNSW::search` + `beamSearch` en `HNSW.hpp`):

1. Baja por los niveles de arriba con `ef = 1` y sin limite de visitas; el mejor punto
   de cada nivel es la entrada del siguiente.
2. En el nivel 0 hace la busqueda con `ef = Ls`. Ahi `visit` limita los nodos
   **expandidos** (`limit_eval`).
3. Devuelve los `k` mas cercanos del *workset*.

Detalles que se copian para que salga igual (`IGUAL QUE HNSW` en el codigo):
- El filtro de visitados tiene `bits = ceil(log2(ef))*2 - 2` (2 bits si `ef <= 2`).
  El punto de entrada se marca con `hash64` y los vecinos con `hash64_2`.
- `cand` es un `std::set` ordenado **solo por distancia**: si ya hay un candidato con la
  misma distancia, el nuevo no entra.
- `is_inw`: cada nodo entra al *workset* como maximo una vez por nivel.
- Aceptar un vecino cambia el umbral del siguiente, asi que esa decision se toma
  **en orden**. Leer la fila, el filtro, las distancias y correr los arreglos ordenados
  se hace con los 32 hilos.
- El nodo `i` del grafo guarda el punto `datoDe[i]` de la base; si no es la identidad,
  los vectores se suben reordenados por nodo y la salida se traduce a ids de la base.
- Las tablas globales (visitados e `is_inw`) llevan un **numero de epoca** en cada
  casilla, asi no hay que limpiarlas en cada consulta.

Unica diferencia posible con la CPU: el *workset* de la CPU es un heap y, cuando hay
**empates exactos de distancia**, puede quedarse con otro id de igual distancia. Eso no
cambia el recall (las distancias son las mismas), pero en la validacion vecino por
vecino (Paso 4) puede aparecer como diferencia en HNSW.

**Optimizaciones** (justificadas con `paso2_perfil.exe` y `micro_distancias.exe`):
- Cada vector lo leen **8 hilos juntos** (una lectura de 128 bytes) y cada hilo lleva
  4 distancias, asi hay 16 vectores pedidos a memoria a la vez por warp.
- La fila de vecinos y las casillas del filtro se leen **todas de una vez** en vez de
  una ida a memoria por cada trozo de 32 vecinos.
- Se reusan arreglos de memoria compartida para que entren mas warps por SM, y el
  tamano de bloque respeta el limite de registros del kernel.
- **Tablas con epoca:** el filtro "ya visto" crece con Ls² y antes se limpiaba en cada
  consulta, lo que llenaba la cache L2 con Ls grandes. Ahora cada casilla guarda
  `(epoca << 32 | id)` y cada consulta usa una epoca nueva: no se limpia nada y el
  resultado es el mismo (una casilla vieja nunca coincide).

---

## Usarlo en otra maquina (por ejemplo RTX 5050 + i7 de 13.a generacion)

Paso a paso completo, experimentos extra y estructura del paper en
[`GUIA_EXPERIMENTOS.md`](GUIA_EXPERIMENTOS.md). En resumen:

1. **Repetir el Paso A en esa maquina.** El speedup siempre se compara contra la CPU
   de la misma maquina.
2. `build.bat` ya compila para `sm_86` y `sm_120` (y deja PTX para GPUs mas nuevas).
   Requiere CUDA 12.8+.
3. Si Visual Studio no es la edicion *Community*, ajustar la ruta de `vcvars64.bat`
   en `build.bat`.
4. `comparacion1.bat` detecta solo la carpeta de datos dentro de WSL, el CSV de CPU mas
   reciente y el grafo de la corrida con mas hilos (no depende de `threads12` ni del
   nombre de usuario). Para los programas sueltos, pasar la carpeta de datos como
   primer argumento.
5. `comparacion1.bat` guarda CPU, RAM, plan de energia y GPU en
   `resultados/comparacion1_equipo_windows.txt`.

---

## Pasos futuros

### Comparacion 1 — Busqueda (codigo completo, falta correrla)

Los pasos 3 (HNSW), 4 (validacion vecino por vecino) y 5 (medicion final) estan
implementados. Falta ejecutar `comparacion1.bat` y revisar:
- `72 de 72` configuraciones con recall identico y `desbordes 0`;
- validacion vecino por vecino ~100 % (si HNSW difiere, revisar empates de distancia);
- con eso, las tablas y graficas de `resultados/comparacion1/` van al informe.

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
