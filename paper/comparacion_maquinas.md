# Comparacion 1 entre maquinas

Mejor QPS que cumple cada recall@10 minimo. GPU = de punta a punta.
CPU con todos sus hilos (12 en A, 24 en B).

## Recall >= 0.90

| Algoritmo | CPU A | GPU A | **x A** | CPU B | GPU B | **x B** | CPU B/A | GPU B/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiskANN | 74,473 | 344,395 | **4.6x** | 107,770 | 1,261,511 | **11.7x** | 1.45x | 3.66x |
| HCNNG | 35,552 | 167,164 | **4.7x** | 50,912 | 517,290 | **10.2x** | 1.43x | 3.09x |
| PyNNDescent | 72,518 | 330,460 | **4.6x** | 100,805 | 1,058,687 | **10.5x** | 1.39x | 3.20x |
| HNSW | 17,853 | 192,328 | **10.8x** | 41,249 | 385,865 | **9.4x** | 2.31x | 2.01x |

## Recall >= 0.95

| Algoritmo | CPU A | GPU A | **x A** | CPU B | GPU B | **x B** | CPU B/A | GPU B/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiskANN | 32,079 | 140,531 | **4.4x** | 52,325 | 561,297 | **10.7x** | 1.63x | 3.99x |
| HCNNG | 19,936 | 74,808 | **3.8x** | 28,150 | 251,385 | **8.9x** | 1.41x | 3.36x |
| PyNNDescent | 44,258 | 163,524 | **3.7x** | 62,352 | 605,957 | **9.7x** | 1.41x | 3.71x |
| HNSW | 7,350 | 52,635 | **7.2x** | 16,530 | 108,921 | **6.6x** | 2.25x | 2.07x |

## Recall >= 0.98

| Algoritmo | CPU A | GPU A | **x A** | CPU B | GPU B | **x B** | CPU B/A | GPU B/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiskANN | 7,920 | 23,350 | **2.9x** | 12,482 | 62,493 | **5.0x** | 1.58x | 2.68x |
| HCNNG | 10,509 | 27,754 | **2.6x** | 16,226 | 108,498 | **6.7x** | 1.54x | 3.91x |
| PyNNDescent | 18,253 | 48,795 | **2.7x** | 27,658 | 196,575 | **7.1x** | 1.52x | 4.03x |
| HNSW | 1,348 | 6,149 | **4.6x** | 5,291 | 31,841 | **6.0x** | 3.93x | 5.18x |

## Microbenchmark de distancias (millones por segundo)

| Variante | A (RTX 3050) | B (RTX 5050) | B/A |
|---|---:|---:|---:|
| 1 hilo por distancia | 148 | 568 | 3.84x |
| 1 warp por distancia | 527 | 996 | 1.89x |
| 8 hilos por distancia (la usada) | 555 | 2,590 | 4.67x |
| 4 hilos por distancia | 486 | 2,350 | 4.84x |
