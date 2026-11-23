# Comparacion 1 - Busqueda CPU vs GPU (MSSPACEV-1M)

Mismos grafos (construidos por ParlayANN en CPU), mismas 29 316 consultas, k = 10,
mismas 18 configuraciones por algoritmo. QPS = mejor de 5 corridas.

- **CPU**: ParlayANN original via big-ann-benchmarks (`run.py`), 24 hilos.
- **GPU total**: copiar consultas a la GPU + kernel + traer ids y distancias.
- **GPU kernel**: solo el kernel (consultas ya en la GPU).

## Equipo

CPU:
```
fecha: 2026-11-23T14:11:40-05:00
dataset: msspacev-1M   hilos: 24   runs: 5
algoritmos: ParDiskANN ParHCNNG ParPyNNDescent ParHNSW
cpu: 13th Gen Intel(R) Core(TM) i7-13700HX
nproc (WSL): 24
ram (WSL): 7.6Gi
ParlayANN commit: 45587ba
big-ann-benchmarks: ppopp-artifact-final
```
GPU:
```
gpu: NVIDIA GeForce RTX 5050 Laptop GPU
sms: 20
compute: 12.0
memoria_mb: 8151
cuda_driver: 13030
cuda_runtime: 12080
corridas: 5
```

## Mejor QPS que cumple cada recall@10 minimo

| Algoritmo | Recall >= | QPS CPU | QPS GPU total | QPS GPU kernel | x total | x kernel |
|---|---|---:|---:|---:|---:|---:|
| ParDiskANN | 0.90 | 107,770 | 1,261,511 | 1,293,649 | 11.71 | 12.00 |
| ParDiskANN | 0.95 | 52,325 | 561,297 | 573,139 | 10.73 | 10.95 |
| ParDiskANN | 0.98 | 12,482 | 62,493 | 62,586 | 5.01 | 5.01 |
| ParHCNNG | 0.90 | 50,912 | 517,290 | 526,418 | 10.16 | 10.34 |
| ParHCNNG | 0.95 | 28,150 | 251,385 | 252,170 | 8.93 | 8.96 |
| ParHCNNG | 0.98 | 16,226 | 108,498 | 108,695 | 6.69 | 6.70 |
| ParPyNNDescent | 0.90 | 100,805 | 1,058,687 | 1,065,669 | 10.50 | 10.57 |
| ParPyNNDescent | 0.95 | 62,352 | 605,957 | 611,876 | 9.72 | 9.81 |
| ParPyNNDescent | 0.98 | 27,658 | 196,575 | 197,109 | 7.11 | 7.13 |
| ParHNSW | 0.90 | 41,249 | 385,865 | 388,448 | 9.35 | 9.42 |
| ParHNSW | 0.95 | 16,530 | 108,921 | 109,073 | 6.59 | 6.60 |
| ParHNSW | 0.98 | 5,291 | 31,841 | 31,853 | 6.02 | 6.02 |

## Validacion (GPU vs CPU)

| Algoritmo | Configs con recall identico | Consultas con los mismos 10 vecinos | Consultas con las mismas 10 distancias | Ids distintos | Consultas cortas | Configs deterministas |
|---|---:|---:|---:|---:|---:|---:|
| ParDiskANN | 18 de 18 | 100.0000 % | 100.0000 % | 0 | 163 | 18 de 18 |
| ParHCNNG | 18 de 18 | 100.0000 % | 100.0000 % | 0 | 0 | 18 de 18 |
| ParPyNNDescent | 18 de 18 | 100.0000 % | 100.0000 % | 0 | 26 | 18 de 18 |
| ParHNSW | 18 de 18 | 99.4372 % | 100.0000 % | 10552 | 4 | 18 de 18 |

*Mismas 10 distancias*: aunque algun id sea distinto, los 10 vecinos de la GPU
estan a exactamente las mismas distancias que los de la CPU (empates).

*Determinista*: la corrida de calentamiento y la ultima corrida devolvieron
exactamente los mismos vecinos.

*Consultas cortas*: la frontera termino con menos de 10 elementos; ahi la CPU
devuelve memoria sin inicializar, por eso no se comparan vecino por vecino.

## Todas las configuraciones

| Algoritmo | Ls | visit | recall CPU | recall GPU | QPS CPU | QPS GPU total | QPS GPU kernel | x total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ParDiskANN | 10 | 15 | 0.69930 | 0.69930 | 271,081 | 3,082,099 | 3,154,603 | 11.37 |
| ParDiskANN | 10 | 12 | 0.59666 | 0.59666 | 340,125 | 3,302,355 | 3,445,232 | 9.71 |
| ParDiskANN | 10 | 8 | 0.33140 | 0.33140 | 579,848 | 4,285,840 | 4,614,100 | 7.39 |
| ParDiskANN | 10 | 5 | 0.07577 | 0.07577 | 1,093,924 | 6,049,900 | 6,864,314 | 5.53 |
| ParDiskANN | 10 | 3 | 0.00644 | 0.00644 | 1,994,241 | 8,694,982 | 9,765,542 | 4.36 |
| ParDiskANN | 15 | - | 0.83848 | 0.83848 | 173,846 | 2,020,386 | 2,084,960 | 11.62 |
| ParDiskANN | 20 | - | 0.86970 | 0.86970 | 135,656 | 1,657,076 | 1,726,892 | 12.22 |
| ParDiskANN | 30 | - | 0.90488 | 0.90488 | 107,770 | 1,261,511 | 1,293,649 | 11.71 |
| ParDiskANN | 40 | - | 0.92431 | 0.92431 | 84,987 | 999,833 | 1,008,707 | 11.76 |
| ParDiskANN | 50 | - | 0.93678 | 0.93678 | 69,803 | 802,218 | 826,201 | 11.49 |
| ParDiskANN | 60 | - | 0.94540 | 0.94540 | 61,531 | 692,244 | 695,636 | 11.25 |
| ParDiskANN | 75 | - | 0.95460 | 0.95460 | 52,325 | 561,297 | 573,139 | 10.73 |
| ParDiskANN | 100 | - | 0.96421 | 0.96421 | 40,037 | 449,911 | 453,724 | 11.24 |
| ParDiskANN | 150 | - | 0.97410 | 0.97410 | 29,727 | 283,732 | 284,986 | 9.54 |
| ParDiskANN | 200 | - | 0.97921 | 0.97921 | 22,278 | 184,839 | 185,801 | 8.30 |
| ParDiskANN | 400 | - | 0.98741 | 0.98741 | 12,482 | 62,493 | 62,586 | 5.01 |
| ParDiskANN | 500 | - | 0.98919 | 0.98919 | 10,475 | 54,608 | 54,681 | 5.21 |
| ParDiskANN | 1000 | - | 0.99292 | 0.99292 | 4,586 | 28,300 | 28,310 | 6.17 |
| ParHCNNG | 10 | 15 | 0.73641 | 0.73641 | 190,121 | 1,116,685 | 1,141,240 | 5.87 |
| ParHCNNG | 10 | 12 | 0.66178 | 0.66178 | 217,703 | 1,310,751 | 1,336,429 | 6.02 |
| ParHCNNG | 10 | 8 | 0.44621 | 0.44621 | 328,249 | 1,746,279 | 1,808,704 | 5.32 |
| ParHCNNG | 10 | 5 | 0.13614 | 0.13614 | 591,198 | 2,386,208 | 2,540,726 | 4.04 |
| ParHCNNG | 10 | 3 | 0.01156 | 0.01156 | 1,026,559 | 3,480,470 | 3,733,419 | 3.39 |
| ParHCNNG | 15 | - | 0.83214 | 0.83214 | 126,950 | 773,109 | 782,836 | 6.09 |
| ParHCNNG | 20 | - | 0.85714 | 0.85714 | 102,632 | 644,839 | 652,544 | 6.28 |
| ParHCNNG | 30 | - | 0.88317 | 0.88317 | 78,009 | 484,962 | 490,110 | 6.22 |
| ParHCNNG | 40 | - | 0.89509 | 0.89509 | 62,214 | 631,329 | 636,841 | 10.15 |
| ParHCNNG | 50 | - | 0.90099 | 0.90099 | 50,912 | 517,290 | 526,418 | 10.16 |
| ParHCNNG | 60 | - | 0.90386 | 0.90386 | 44,541 | 441,414 | 447,946 | 9.91 |
| ParHCNNG | 75 | - | 0.94097 | 0.94097 | 36,651 | 365,100 | 366,592 | 9.96 |
| ParHCNNG | 100 | - | 0.96407 | 0.96407 | 28,150 | 251,385 | 252,170 | 8.93 |
| ParHCNNG | 150 | - | 0.97885 | 0.97885 | 19,973 | 150,890 | 151,273 | 7.55 |
| ParHCNNG | 200 | - | 0.98661 | 0.98661 | 16,226 | 108,498 | 108,695 | 6.69 |
| ParHCNNG | 400 | - | 0.99451 | 0.99451 | 8,879 | 45,903 | 45,948 | 5.17 |
| ParHCNNG | 500 | - | 0.99513 | 0.99513 | 7,438 | 34,658 | 34,687 | 4.66 |
| ParHCNNG | 1000 | - | 0.99723 | 0.99723 | 3,171 | 13,799 | 13,803 | 4.35 |
| ParPyNNDescent | 10 | 15 | 0.66071 | 0.66071 | 317,156 | 3,243,423 | 3,477,969 | 10.23 |
| ParPyNNDescent | 10 | 12 | 0.54509 | 0.54509 | 397,218 | 3,779,540 | 4,082,499 | 9.52 |
| ParPyNNDescent | 10 | 8 | 0.27790 | 0.27790 | 652,636 | 5,032,099 | 5,497,234 | 7.71 |
| ParPyNNDescent | 10 | 5 | 0.05626 | 0.05626 | 1,033,770 | 6,958,957 | 8,141,162 | 6.73 |
| ParPyNNDescent | 10 | 3 | 0.00440 | 0.00440 | 1,853,834 | 10,939,622 | 11,423,575 | 5.90 |
| ParPyNNDescent | 15 | - | 0.82725 | 0.82725 | 209,181 | 2,104,236 | 2,160,825 | 10.06 |
| ParPyNNDescent | 20 | - | 0.86022 | 0.86022 | 170,045 | 1,748,497 | 1,795,436 | 10.28 |
| ParPyNNDescent | 30 | - | 0.89816 | 0.89816 | 130,413 | 1,345,820 | 1,385,390 | 10.32 |
| ParPyNNDescent | 40 | - | 0.92048 | 0.92048 | 100,805 | 1,058,687 | 1,065,669 | 10.50 |
| ParPyNNDescent | 50 | - | 0.93448 | 0.93448 | 86,895 | 867,966 | 880,036 | 9.99 |
| ParPyNNDescent | 60 | - | 0.94491 | 0.94491 | 72,172 | 746,946 | 761,585 | 10.35 |
| ParPyNNDescent | 75 | - | 0.95559 | 0.95559 | 62,352 | 605,957 | 611,876 | 9.72 |
| ParPyNNDescent | 100 | - | 0.96624 | 0.96624 | 48,682 | 480,394 | 485,497 | 9.87 |
| ParPyNNDescent | 150 | - | 0.97774 | 0.97774 | 34,930 | 308,215 | 310,236 | 8.82 |
| ParPyNNDescent | 200 | - | 0.98324 | 0.98324 | 27,658 | 196,575 | 197,109 | 7.11 |
| ParPyNNDescent | 400 | - | 0.99107 | 0.99107 | 14,280 | 73,509 | 73,580 | 5.15 |
| ParPyNNDescent | 500 | - | 0.99255 | 0.99255 | 12,187 | 63,396 | 63,471 | 5.20 |
| ParPyNNDescent | 1000 | - | 0.99550 | 0.99550 | 5,113 | 31,670 | 31,695 | 6.19 |
| ParHNSW | 10 | 15 | 0.71645 | 0.71645 | 139,936 | 1,186,791 | 1,203,975 | 8.48 |
| ParHNSW | 10 | 12 | 0.68671 | 0.68671 | 150,673 | 1,272,031 | 1,300,010 | 8.44 |
| ParHNSW | 10 | 8 | 0.59518 | 0.59518 | 191,818 | 1,577,664 | 1,630,139 | 8.22 |
| ParHNSW | 10 | 5 | 0.46971 | 0.46971 | 280,881 | 1,996,554 | 2,036,299 | 7.11 |
| ParHNSW | 10 | 3 | 0.33283 | 0.33283 | 413,172 | 2,422,490 | 2,523,865 | 5.86 |
| ParHNSW | 15 | - | 0.79128 | 0.79128 | 101,168 | 877,386 | 894,740 | 8.67 |
| ParHNSW | 20 | - | 0.82466 | 0.82466 | 84,170 | 749,670 | 761,494 | 8.91 |
| ParHNSW | 30 | - | 0.86480 | 0.86480 | 61,739 | 553,799 | 559,967 | 8.97 |
| ParHNSW | 40 | - | 0.88840 | 0.88840 | 49,533 | 466,266 | 469,424 | 9.41 |
| ParHNSW | 50 | - | 0.90383 | 0.90383 | 41,249 | 385,865 | 388,448 | 9.35 |
| ParHNSW | 60 | - | 0.91593 | 0.91593 | 35,501 | 329,245 | 331,239 | 9.27 |
| ParHNSW | 75 | - | 0.92729 | 0.92729 | 29,622 | 280,516 | 281,689 | 9.47 |
| ParHNSW | 100 | - | 0.94087 | 0.94087 | 23,466 | 193,070 | 193,867 | 8.23 |
| ParHNSW | 150 | - | 0.95619 | 0.95619 | 16,530 | 108,921 | 109,073 | 6.59 |
| ParHNSW | 200 | - | 0.96472 | 0.96472 | 12,864 | 71,123 | 71,203 | 5.53 |
| ParHNSW | 400 | - | 0.97777 | 0.97777 | 6,423 | 45,001 | 45,024 | 7.01 |
| ParHNSW | 500 | - | 0.98033 | 0.98033 | 5,291 | 31,841 | 31,853 | 6.02 |
| ParHNSW | 1000 | - | 0.98584 | 0.98584 | 2,333 | 6,927 | 6,928 | 2.97 |

Graficas: `recall_qps.png` y `speedup.png`.
