// utiles_gpu.cuh
// funciones de GPU que usan los dos kernels de busqueda (grafos simples y HNSW)
#pragma once

#include "comun.h"

#define TODOS 0xffffffffu
#define VACIO 0xffffffffu   // igual a SIN_VECINO (relleno de las matrices de vecinos)

// hash64_2 de parlaylib (utilities.h), copiado tal cual
// IGUAL QUE PARLAYANN: los filtros tienen que caer en las mismas posiciones
__device__ __forceinline__ uint64_t hash64_2(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x = x ^ (x >> 31);
    return x;
}

// hash64 de parlaylib ("from numerical recipes"), copiado tal cual
// HNSW lo usa SOLO para marcar los puntos de entrada en el filtro de visitados
__device__ __forceinline__ uint64_t hash64(uint64_t u) {
    uint64_t v = u * 3935559000370003845ull + 2691343689449507681ull;
    v ^= v >> 21;
    v ^= v << 37;
    v ^= v >> 4;
    v *= 4768777513237032717ull;
    v ^= v << 20;
    v ^= v >> 41;
    v ^= v << 5;
    return v;
}

// orden de ParlayANN: primero distancia, si empatan gana el id menor
__device__ __forceinline__ bool menor(int dA, uint32_t idA, int dB, uint32_t idB) {
    return dA < dB || (dA == dB && idA < idB);
}

__device__ __forceinline__ int minimoWarp(int x) {
    for (int salto = 16; salto > 0; salto >>= 1) x = min(x, __shfl_xor_sync(TODOS, x, salto));
    return x;
}

// EPOCAS: en las tablas globales cada casilla guarda (epoca << 32 | id).
// Al empezar una busqueda nueva se sube la epoca y todas las casillas viejas
// cuentan como vacias, asi no hay que limpiar la tabla (las tablas empiezan en 0
// con cudaMemset y la epoca 0 nunca se usa)
__device__ __forceinline__ unsigned long long marca(uint32_t epoca, uint32_t id) {
    return ((unsigned long long)epoca << 32) | id;
}

// la fila de vecinos se procesa en trozos de 32 -> los arreglos se redondean a multiplo de 32
__host__ __device__ inline int redondear32(int x) { return (x + 31) & ~31; }
