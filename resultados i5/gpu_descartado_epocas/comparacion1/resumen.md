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
| ParDiskANN | 0.90 | 74,473 | 217,678 | 219,112 | 2.92 | 2.94 |
| ParDiskANN | 0.95 | 32,079 | 95,252 | 95,507 | 2.97 | 2.98 |
| ParDiskANN | 0.98 | 7,920 | 19,959 | 19,970 | 2.52 | 2.52 |
| ParHCNNG | 0.90 | 35,552 | 107,189 | 107,577 | 3.01 | 3.03 |
| ParHCNNG | 0.95 | 19,936 | 48,117 | 48,193 | 2.41 | 2.42 |
| ParHCNNG | 0.98 | 10,509 | 24,709 | 24,728 | 2.35 | 2.35 |
| ParPyNNDescent | 0.90 | 72,518 | 193,351 | 194,607 | 2.67 | 2.68 |
| ParPyNNDescent | 0.95 | 44,258 | 107,561 | 107,874 | 2.43 | 2.44 |
| ParPyNNDescent | 0.98 | 18,253 | 42,053 | 42,103 | 2.30 | 2.31 |
| ParHNSW | 0.90 | 17,853 | 186,559 | 187,766 | 10.45 | 10.52 |
| ParHNSW | 0.95 | 7,350 | 50,972 | 51,080 | 6.93 | 6.95 |
| ParHNSW | 0.98 | 1,348 | 6,022 | 6,023 | 4.47 | 4.47 |

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
| ParDiskANN | 10 | 15 | 0.69930 | 0.69930 | 160,539 | 535,196 | 542,956 | 3.33 |
| ParDiskANN | 10 | 12 | 0.59666 | 0.59666 | 182,478 | 655,594 | 667,453 | 3.59 |
| ParDiskANN | 10 | 8 | 0.33140 | 0.33140 | 250,161 | 980,508 | 1,005,515 | 3.92 |
| ParDiskANN | 10 | 5 | 0.07577 | 0.07577 | 429,039 | 1,610,176 | 1,682,865 | 3.75 |
| ParDiskANN | 10 | 3 | 0.00644 | 0.00644 | 760,405 | 2,949,563 | 3,201,264 | 3.88 |
| ParDiskANN | 15 | - | 0.83848 | 0.83848 | 108,796 | 350,874 | 354,240 | 3.23 |
| ParDiskANN | 20 | - | 0.86970 | 0.86970 | 90,430 | 288,850 | 291,359 | 3.19 |
| ParDiskANN | 30 | - | 0.90488 | 0.90488 | 74,473 | 217,678 | 219,112 | 2.92 |
| ParDiskANN | 40 | - | 0.92431 | 0.92431 | 59,602 | 173,976 | 174,885 | 2.92 |
| ParDiskANN | 50 | - | 0.93678 | 0.93678 | 49,390 | 144,238 | 144,846 | 2.92 |
| ParDiskANN | 60 | - | 0.94540 | 0.94540 | 41,575 | 123,906 | 124,377 | 2.98 |
| ParDiskANN | 75 | - | 0.95460 | 0.95460 | 32,079 | 95,252 | 95,507 | 2.97 |
| ParDiskANN | 100 | - | 0.96421 | 0.96421 | 28,461 | 72,827 | 72,949 | 2.56 |
| ParDiskANN | 150 | - | 0.97410 | 0.97410 | 20,029 | 48,661 | 48,717 | 2.43 |
| ParDiskANN | 200 | - | 0.97921 | 0.97921 | 15,823 | 36,807 | 36,843 | 2.33 |
| ParDiskANN | 400 | - | 0.98741 | 0.98741 | 7,920 | 19,959 | 19,970 | 2.52 |
| ParDiskANN | 500 | - | 0.98919 | 0.98919 | 6,553 | 18,333 | 18,342 | 2.80 |
| ParDiskANN | 1000 | - | 0.99292 | 0.99292 | 3,038 | 11,212 | 11,216 | 3.69 |
| ParHCNNG | 10 | 15 | 0.73781 | 0.73781 | 114,576 | 392,965 | 397,112 | 3.43 |
| ParHCNNG | 10 | 12 | 0.66453 | 0.66453 | 125,052 | 477,043 | 483,019 | 3.81 |
| ParHCNNG | 10 | 8 | 0.44851 | 0.44851 | 170,312 | 698,088 | 712,339 | 4.10 |
| ParHCNNG | 10 | 5 | 0.13746 | 0.13746 | 262,425 | 1,119,923 | 1,164,280 | 4.27 |
| ParHCNNG | 10 | 3 | 0.01185 | 0.01185 | 436,375 | 1,910,198 | 2,014,019 | 4.38 |
| ParHCNNG | 15 | - | 0.83361 | 0.83361 | 87,993 | 268,543 | 270,674 | 3.05 |
| ParHCNNG | 20 | - | 0.85814 | 0.85814 | 73,416 | 221,537 | 222,971 | 3.02 |
| ParHCNNG | 30 | - | 0.88348 | 0.88348 | 54,273 | 163,871 | 164,817 | 3.02 |
| ParHCNNG | 40 | - | 0.89566 | 0.89566 | 43,481 | 130,168 | 130,602 | 2.99 |
| ParHCNNG | 50 | - | 0.90173 | 0.90173 | 35,552 | 107,189 | 107,577 | 3.01 |
| ParHCNNG | 60 | - | 0.90505 | 0.90505 | 29,717 | 91,962 | 92,164 | 3.09 |
| ParHCNNG | 75 | - | 0.94160 | 0.94160 | 25,060 | 65,700 | 65,822 | 2.62 |
| ParHCNNG | 100 | - | 0.96421 | 0.96421 | 19,936 | 48,117 | 48,193 | 2.41 |
| ParHCNNG | 150 | - | 0.97931 | 0.97931 | 13,596 | 32,480 | 32,518 | 2.39 |
| ParHCNNG | 200 | - | 0.98708 | 0.98708 | 10,509 | 24,709 | 24,728 | 2.35 |
| ParHCNNG | 400 | - | 0.99495 | 0.99495 | 5,123 | 14,127 | 14,134 | 2.76 |
| ParHCNNG | 500 | - | 0.99570 | 0.99570 | 4,651 | 11,807 | 11,811 | 2.54 |
| ParHCNNG | 1000 | - | 0.99781 | 0.99781 | 2,115 | 7,354 | 7,356 | 3.48 |
| ParPyNNDescent | 10 | 15 | 0.65202 | 0.65202 | 207,245 | 594,742 | 604,874 | 2.87 |
| ParPyNNDescent | 10 | 12 | 0.53313 | 0.53313 | 246,750 | 728,762 | 742,836 | 2.95 |
| ParPyNNDescent | 10 | 8 | 0.26336 | 0.26336 | 347,192 | 1,080,467 | 1,110,590 | 3.11 |
| ParPyNNDescent | 10 | 5 | 0.05020 | 0.05020 | 544,754 | 1,812,023 | 1,900,611 | 3.33 |
| ParPyNNDescent | 10 | 3 | 0.00382 | 0.00382 | 783,778 | 1,453,575 | 1,551,114 | 1.85 |
| ParPyNNDescent | 15 | - | 0.82517 | 0.82517 | 141,866 | 381,274 | 385,908 | 2.69 |
| ParPyNNDescent | 20 | - | 0.85871 | 0.85871 | 119,694 | 317,259 | 319,780 | 2.65 |
| ParPyNNDescent | 30 | - | 0.89827 | 0.89827 | 92,383 | 241,456 | 243,269 | 2.61 |
| ParPyNNDescent | 40 | - | 0.91999 | 0.91999 | 72,518 | 193,351 | 194,607 | 2.67 |
| ParPyNNDescent | 50 | - | 0.93422 | 0.93422 | 61,800 | 161,103 | 161,778 | 2.61 |
| ParPyNNDescent | 60 | - | 0.94484 | 0.94484 | 52,117 | 138,531 | 139,120 | 2.66 |
| ParPyNNDescent | 75 | - | 0.95495 | 0.95495 | 44,258 | 107,561 | 107,874 | 2.43 |
| ParPyNNDescent | 100 | - | 0.96656 | 0.96656 | 34,364 | 82,605 | 82,854 | 2.40 |
| ParPyNNDescent | 150 | - | 0.97771 | 0.97771 | 24,040 | 55,951 | 56,031 | 2.33 |
| ParPyNNDescent | 200 | - | 0.98306 | 0.98306 | 18,253 | 42,053 | 42,103 | 2.30 |
| ParPyNNDescent | 400 | - | 0.99115 | 0.99115 | 9,131 | 22,251 | 22,269 | 2.44 |
| ParPyNNDescent | 500 | - | 0.99273 | 0.99273 | 7,368 | 21,006 | 21,020 | 2.85 |
| ParPyNNDescent | 1000 | - | 0.99561 | 0.99561 | 3,332 | 12,305 | 12,309 | 3.69 |
| ParHNSW | 10 | 15 | 0.72272 | 0.72272 | 57,717 | 719,596 | 735,187 | 12.47 |
| ParHNSW | 10 | 12 | 0.69393 | 0.69393 | 54,571 | 768,344 | 786,339 | 14.08 |
| ParHNSW | 10 | 8 | 0.60335 | 0.60335 | 71,076 | 986,181 | 1,008,073 | 13.88 |
| ParHNSW | 10 | 5 | 0.48038 | 0.48038 | 97,406 | 1,299,791 | 1,344,157 | 13.34 |
| ParHNSW | 10 | 3 | 0.34479 | 0.34479 | 132,515 | 1,461,817 | 1,511,184 | 11.03 |
| ParHNSW | 15 | - | 0.79486 | 0.79486 | 42,523 | 518,958 | 526,074 | 12.20 |
| ParHNSW | 20 | - | 0.82687 | 0.82687 | 35,418 | 430,143 | 435,845 | 12.14 |
| ParHNSW | 30 | - | 0.86657 | 0.86657 | 26,017 | 318,502 | 321,303 | 12.24 |
| ParHNSW | 40 | - | 0.89072 | 0.89072 | 21,360 | 224,580 | 225,510 | 10.51 |
| ParHNSW | 50 | - | 0.90642 | 0.90642 | 17,853 | 186,559 | 187,766 | 10.45 |
| ParHNSW | 60 | - | 0.91741 | 0.91741 | 15,184 | 160,258 | 160,868 | 10.55 |
| ParHNSW | 75 | - | 0.92980 | 0.92980 | 12,987 | 99,549 | 99,839 | 7.67 |
| ParHNSW | 100 | - | 0.94289 | 0.94289 | 10,226 | 79,770 | 79,995 | 7.80 |
| ParHNSW | 150 | - | 0.95824 | 0.95824 | 7,350 | 50,972 | 51,080 | 6.93 |
| ParHNSW | 200 | - | 0.96584 | 0.96584 | 5,753 | 40,762 | 40,816 | 7.08 |
| ParHNSW | 400 | - | 0.97718 | 0.97718 | 3,125 | 19,889 | 19,898 | 6.36 |
| ParHNSW | 500 | - | 0.97981 | 0.97981 | 2,602 | 16,767 | 16,773 | 6.44 |
| ParHNSW | 1000 | - | 0.98564 | 0.98564 | 1,348 | 6,022 | 6,023 | 4.47 |

Graficas: `recall_qps.png` y `speedup.png`.
