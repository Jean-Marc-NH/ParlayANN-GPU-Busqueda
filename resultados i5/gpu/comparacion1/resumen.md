# Comparacion 1 - Busqueda CPU vs GPU (MSSPACEV-1M)

Mismos grafos (construidos por ParlayANN en CPU), mismas 29 316 consultas, k = 10,
mismas 18 configuraciones por algoritmo. QPS = mejor de 5 corridas.

- **CPU**: ParlayANN original via big-ann-benchmarks (`run.py`), 12 hilos.
- **GPU total**: copiar consultas a la GPU + kernel + traer ids y distancias.
- **GPU kernel**: solo el kernel (consultas ya en la GPU).

## Equipo

CPU:
```
fecha: 2026-11-11T16:24:50-05:00
dataset: msspacev-1M   hilos: 12   runs: 5
algoritmos: ParDiskANN ParHCNNG ParPyNNDescent ParHNSW
cpu: 12th Gen Intel(R) Core(TM) i5-12450HX
nproc (WSL): 12
ram (WSL): 11Gi
ParlayANN commit: 45587ba
big-ann-benchmarks: ppopp-artifact-final
```
GPU:
```
gpu: NVIDIA GeForce RTX 3050 6GB Laptop GPU
sms: 20
compute: 8.6
memoria_mb: 6144
cuda_driver: 13000
cuda_runtime: 12090
corridas: 5
```

## Mejor QPS que cumple cada recall@10 minimo

| Algoritmo | Recall >= | QPS CPU | QPS GPU total | QPS GPU kernel | x total | x kernel |
|---|---|---:|---:|---:|---:|---:|
| ParDiskANN | 0.90 | 74,473 | 344,395 | 347,570 | 4.62 | 4.67 |
| ParDiskANN | 0.95 | 32,079 | 140,531 | 141,139 | 4.38 | 4.40 |
| ParDiskANN | 0.98 | 7,920 | 23,350 | 23,364 | 2.95 | 2.95 |
| ParHCNNG | 0.90 | 35,552 | 167,164 | 168,062 | 4.70 | 4.73 |
| ParHCNNG | 0.95 | 19,936 | 74,808 | 74,992 | 3.75 | 3.76 |
| ParHCNNG | 0.98 | 10,509 | 27,754 | 27,773 | 2.64 | 2.64 |
| ParPyNNDescent | 0.90 | 72,518 | 330,460 | 333,935 | 4.56 | 4.60 |
| ParPyNNDescent | 0.95 | 44,258 | 163,524 | 164,270 | 3.69 | 3.71 |
| ParPyNNDescent | 0.98 | 18,253 | 48,795 | 48,836 | 2.67 | 2.68 |
| ParHNSW | 0.90 | 17,853 | 192,328 | 193,470 | 10.77 | 10.84 |
| ParHNSW | 0.95 | 7,350 | 52,635 | 52,729 | 7.16 | 7.17 |
| ParHNSW | 0.98 | 1,348 | 6,149 | 6,151 | 4.56 | 4.56 |

## Validacion (GPU vs CPU)

| Algoritmo | Configs con recall identico | Consultas con los mismos 10 vecinos | Consultas con las mismas 10 distancias | Ids distintos | Consultas cortas | Configs deterministas |
|---|---:|---:|---:|---:|---:|---:|
| ParDiskANN | 18 de 18 | 100.0000 % | 100.0000 % | 0 | 163 | 18 de 18 |
| ParHCNNG | 18 de 18 | 100.0000 % | 100.0000 % | 0 | 78 | 18 de 18 |
| ParPyNNDescent | 18 de 18 | 100.0000 % | 100.0000 % | 0 | 104 | 18 de 18 |
| ParHNSW | 18 de 18 | 99.4201 % | 100.0000 % | 13717 | 1 | 18 de 18 |

*Mismas 10 distancias*: aunque algun id sea distinto, los 10 vecinos de la GPU
estan a exactamente las mismas distancias que los de la CPU (empates).

*Determinista*: la corrida de calentamiento y la ultima corrida devolvieron
exactamente los mismos vecinos.

*Consultas cortas*: la frontera termino con menos de 10 elementos; ahi la CPU
devuelve memoria sin inicializar, por eso no se comparan vecino por vecino.

## Todas las configuraciones

| Algoritmo | Ls | visit | recall CPU | recall GPU | QPS CPU | QPS GPU total | QPS GPU kernel | x total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ParDiskANN | 10 | 15 | 0.69930 | 0.69930 | 160,539 | 834,313 | 853,575 | 5.20 |
| ParDiskANN | 10 | 12 | 0.59666 | 0.59666 | 182,478 | 1,018,939 | 1,047,680 | 5.58 |
| ParDiskANN | 10 | 8 | 0.33140 | 0.33140 | 250,161 | 1,500,968 | 1,570,541 | 6.00 |
| ParDiskANN | 10 | 5 | 0.07577 | 0.07577 | 429,039 | 2,529,575 | 2,734,914 | 5.90 |
| ParDiskANN | 10 | 3 | 0.00644 | 0.00644 | 760,405 | 4,846,982 | 5,412,435 | 6.37 |
| ParDiskANN | 15 | - | 0.83848 | 0.83848 | 108,796 | 550,792 | 558,950 | 5.06 |
| ParDiskANN | 20 | - | 0.86970 | 0.86970 | 90,430 | 451,967 | 457,585 | 5.00 |
| ParDiskANN | 30 | - | 0.90488 | 0.90488 | 74,473 | 344,395 | 347,570 | 4.62 |
| ParDiskANN | 40 | - | 0.92431 | 0.92431 | 59,602 | 277,036 | 278,983 | 4.65 |
| ParDiskANN | 50 | - | 0.93678 | 0.93678 | 49,390 | 228,399 | 229,726 | 4.62 |
| ParDiskANN | 60 | - | 0.94540 | 0.94540 | 41,575 | 196,147 | 197,252 | 4.72 |
| ParDiskANN | 75 | - | 0.95460 | 0.95460 | 32,079 | 140,531 | 141,139 | 4.38 |
| ParDiskANN | 100 | - | 0.96421 | 0.96421 | 28,461 | 90,875 | 91,148 | 3.19 |
| ParDiskANN | 150 | - | 0.97410 | 0.97410 | 20,029 | 56,490 | 56,588 | 2.82 |
| ParDiskANN | 200 | - | 0.97921 | 0.97921 | 15,823 | 40,742 | 40,783 | 2.57 |
| ParDiskANN | 400 | - | 0.98741 | 0.98741 | 7,920 | 23,350 | 23,364 | 2.95 |
| ParDiskANN | 500 | - | 0.98919 | 0.98919 | 6,553 | 19,030 | 19,039 | 2.90 |
| ParDiskANN | 1000 | - | 0.99292 | 0.99292 | 3,038 | 12,983 | 12,987 | 4.27 |
| ParHCNNG | 10 | 15 | 0.73781 | 0.73781 | 114,576 | 627,471 | 639,139 | 5.48 |
| ParHCNNG | 10 | 12 | 0.66453 | 0.66453 | 125,052 | 756,700 | 771,752 | 6.05 |
| ParHCNNG | 10 | 8 | 0.44851 | 0.44851 | 170,312 | 1,081,975 | 1,108,877 | 6.35 |
| ParHCNNG | 10 | 5 | 0.13746 | 0.13746 | 262,425 | 1,635,098 | 1,703,088 | 6.23 |
| ParHCNNG | 10 | 3 | 0.01185 | 0.01185 | 436,375 | 2,095,871 | 2,216,053 | 4.80 |
| ParHCNNG | 15 | - | 0.83361 | 0.83361 | 87,993 | 431,281 | 437,063 | 4.90 |
| ParHCNNG | 20 | - | 0.85814 | 0.85814 | 73,416 | 356,885 | 360,475 | 4.86 |
| ParHCNNG | 30 | - | 0.88348 | 0.88348 | 54,273 | 266,616 | 268,484 | 4.91 |
| ParHCNNG | 40 | - | 0.89566 | 0.89566 | 43,481 | 203,234 | 204,377 | 4.67 |
| ParHCNNG | 50 | - | 0.90173 | 0.90173 | 35,552 | 167,164 | 168,062 | 4.70 |
| ParHCNNG | 60 | - | 0.90505 | 0.90505 | 29,717 | 142,795 | 143,426 | 4.81 |
| ParHCNNG | 75 | - | 0.94160 | 0.94160 | 25,060 | 110,136 | 110,448 | 4.39 |
| ParHCNNG | 100 | - | 0.96421 | 0.96421 | 19,936 | 74,808 | 74,992 | 3.75 |
| ParHCNNG | 150 | - | 0.97931 | 0.97931 | 13,596 | 39,346 | 39,404 | 2.89 |
| ParHCNNG | 200 | - | 0.98708 | 0.98708 | 10,509 | 27,754 | 27,773 | 2.64 |
| ParHCNNG | 400 | - | 0.99495 | 0.99495 | 5,123 | 15,372 | 15,377 | 3.00 |
| ParHCNNG | 500 | - | 0.99570 | 0.99570 | 4,651 | 13,390 | 13,395 | 2.88 |
| ParHCNNG | 1000 | - | 0.99781 | 0.99781 | 2,115 | 8,733 | 8,734 | 4.13 |
| ParPyNNDescent | 10 | 15 | 0.65202 | 0.65202 | 207,245 | 980,478 | 1,002,623 | 4.73 |
| ParPyNNDescent | 10 | 12 | 0.53313 | 0.53313 | 246,750 | 1,192,672 | 1,235,336 | 4.83 |
| ParPyNNDescent | 10 | 8 | 0.26336 | 0.26336 | 347,192 | 1,776,598 | 1,858,899 | 5.12 |
| ParPyNNDescent | 10 | 5 | 0.05020 | 0.05020 | 544,754 | 2,920,298 | 3,144,306 | 5.36 |
| ParPyNNDescent | 10 | 3 | 0.00382 | 0.00382 | 783,778 | 1,972,163 | 2,087,263 | 2.52 |
| ParPyNNDescent | 15 | - | 0.82517 | 0.82517 | 141,866 | 632,773 | 643,505 | 4.46 |
| ParPyNNDescent | 20 | - | 0.85871 | 0.85871 | 119,694 | 524,278 | 532,454 | 4.38 |
| ParPyNNDescent | 30 | - | 0.89827 | 0.89827 | 92,383 | 408,065 | 412,164 | 4.42 |
| ParPyNNDescent | 40 | - | 0.91999 | 0.91999 | 72,518 | 330,460 | 333,935 | 4.56 |
| ParPyNNDescent | 50 | - | 0.93422 | 0.93422 | 61,800 | 274,436 | 276,822 | 4.44 |
| ParPyNNDescent | 60 | - | 0.94484 | 0.94484 | 52,117 | 236,270 | 237,808 | 4.53 |
| ParPyNNDescent | 75 | - | 0.95495 | 0.95495 | 44,258 | 163,524 | 164,270 | 3.69 |
| ParPyNNDescent | 100 | - | 0.96656 | 0.96656 | 34,364 | 106,929 | 107,313 | 3.11 |
| ParPyNNDescent | 150 | - | 0.97771 | 0.97771 | 24,040 | 67,467 | 67,568 | 2.81 |
| ParPyNNDescent | 200 | - | 0.98306 | 0.98306 | 18,253 | 48,795 | 48,836 | 2.67 |
| ParPyNNDescent | 400 | - | 0.99115 | 0.99115 | 9,131 | 28,314 | 28,334 | 3.10 |
| ParPyNNDescent | 500 | - | 0.99273 | 0.99273 | 7,368 | 22,856 | 22,874 | 3.10 |
| ParPyNNDescent | 1000 | - | 0.99561 | 0.99561 | 3,332 | 15,167 | 15,171 | 4.55 |
| ParHNSW | 10 | 15 | 0.72272 | 0.72272 | 57,717 | 743,186 | 758,227 | 12.88 |
| ParHNSW | 10 | 12 | 0.69393 | 0.69393 | 54,571 | 793,454 | 810,514 | 14.54 |
| ParHNSW | 10 | 8 | 0.60335 | 0.60335 | 71,076 | 1,012,377 | 1,039,803 | 14.24 |
| ParHNSW | 10 | 5 | 0.48038 | 0.48038 | 97,406 | 1,333,964 | 1,382,123 | 13.69 |
| ParHNSW | 10 | 3 | 0.34479 | 0.34479 | 132,515 | 1,470,668 | 742,451 | 11.10 |
| ParHNSW | 15 | - | 0.79486 | 0.79486 | 42,523 | 534,757 | 542,226 | 12.58 |
| ParHNSW | 20 | - | 0.82687 | 0.82687 | 35,418 | 443,809 | 449,104 | 12.53 |
| ParHNSW | 30 | - | 0.86657 | 0.86657 | 26,017 | 328,037 | 331,345 | 12.61 |
| ParHNSW | 40 | - | 0.89072 | 0.89072 | 21,360 | 231,245 | 232,537 | 10.83 |
| ParHNSW | 50 | - | 0.90642 | 0.90642 | 17,853 | 192,328 | 193,470 | 10.77 |
| ParHNSW | 60 | - | 0.91741 | 0.91741 | 15,184 | 164,932 | 165,712 | 10.86 |
| ParHNSW | 75 | - | 0.92980 | 0.92980 | 12,987 | 102,693 | 103,013 | 7.91 |
| ParHNSW | 100 | - | 0.94289 | 0.94289 | 10,226 | 82,374 | 82,584 | 8.06 |
| ParHNSW | 150 | - | 0.95824 | 0.95824 | 7,350 | 52,635 | 52,729 | 7.16 |
| ParHNSW | 200 | - | 0.96584 | 0.96584 | 5,753 | 42,091 | 42,150 | 7.32 |
| ParHNSW | 400 | - | 0.97718 | 0.97718 | 3,125 | 20,531 | 20,548 | 6.57 |
| ParHNSW | 500 | - | 0.97981 | 0.97981 | 2,602 | 17,310 | 17,320 | 6.65 |
| ParHNSW | 1000 | - | 0.98564 | 0.98564 | 1,348 | 6,149 | 6,151 | 4.56 |

Graficas: `recall_qps.png` y `speedup.png`.
