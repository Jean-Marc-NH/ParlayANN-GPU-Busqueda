// comun.h
// constantes, rutas y macros que usan todos los archivos del proyecto cuda
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// MSSPACEV tiene vectores int8 de 100 dimensiones
// en la GPU los rellenamos con ceros hasta 128 bytes:
//   128 bytes = 32 palabras de 4 bytes = 1 palabra por cada hilo del warp
// los ceros no cambian ni la norma ni el producto punto, asi que la distancia es la misma
const int DIM_REAL = 100;
const int DIM_PAD = 128;
const int PALABRAS_POR_VEC = DIM_PAD / 4;   // 32

// relleno en la matriz de vecinos cuando un punto tiene menos vecinos que gradoMax
const uint32_t SIN_VECINO = 0xFFFFFFFFu;

// carpeta "data" de big-ann-benchmarks dentro de WSL, vista desde windows
// se puede cambiar pasando otra ruta como primer argumento al programa
const std::string CARPETA_DATOS_DEFAULT =
    "\\\\wsl.localhost\\Ubuntu\\home\\usuario\\parlay\\big-ann-benchmarks\\data\\";

// archivos del dataset (los mismos que uso la corrida de CPU)
const std::string ARCH_BASE      = "MSSPACEV1B\\spacev1b_base.i8bin.crop_nb_1000000";
const std::string ARCH_CONSULTAS = "MSSPACEV1B\\query.i8bin";
const std::string ARCH_GT        = "MSSPACEV1B\\msspacev-gt-1M";

// grafos que construyo ParlayANN en CPU (corrida msspacev-1M con 12 hilos)
// el de HNSW tiene otro formato, ese se lee en el paso 3
const std::string GRAFO_DISKANN =
    "indices\\vamana\\MSSPACEV1B-1000000\\R64_L128_alpha1.2_threads12\\R64_L128_alpha1.2_threads12";
const std::string GRAFO_HCNNG =
    "indices\\hcnng\\MSSPACEV1B-1000000\\mst3_nc50_cs1000_threads12\\mst3_nc50_cs1000_threads12";
const std::string GRAFO_PYNN =
    "indices\\pynndescent\\MSSPACEV1B-1000000\\mxd60_nc10_cs1000_alpha1.2_delta0.05_threads12\\"
    "mxd60_nc10_cs1000_alpha1.2_delta0.05_threads12";

// revisa cada llamada a cuda, si algo falla muestra donde y termina
#define CHECK_CUDA(llamada)                                                      \
    do {                                                                         \
        cudaError_t errCuda = (llamada);                                         \
        if (errCuda != cudaSuccess) {                                            \
            fprintf(stderr, "ERROR CUDA: %s\n  en %s:%d\n  -> %s\n", #llamada,   \
                    __FILE__, __LINE__, cudaGetErrorString(errCuda));            \
            exit(1);                                                             \
        }                                                                        \
    } while (0)
