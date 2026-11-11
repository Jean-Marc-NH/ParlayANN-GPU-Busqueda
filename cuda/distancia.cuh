// distancia.cuh
// distancia L2 al cuadrado calculada por un warp completo (32 hilos)
#pragma once

#include "comun.h"

// suma entre los 32 hilos del warp
// al terminar TODOS los hilos tienen el total (no solo el hilo 0)
__device__ __forceinline__ int sumaWarp(int x) {
    for (int salto = 16; salto > 0; salto >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, salto);
    return x;
}

// DISTANCIA EXACTA ENTRE LA CONSULTA Y EL PUNTO p
//
//   ||q - p||^2 = ||q||^2 + ||p||^2 - 2 * (q . p)
//
// - cada hilo guarda en un registro SU palabra de 4 bytes de la consulta (miPalabraQ)
// - cada hilo lee la palabra que le toca del punto p -> los 32 hilos leen 128 bytes
//   seguidos, o sea una sola transaccion de memoria
// - __dp4a multiplica 4 pares de int8 y los suma en una sola instruccion
// - TODO ES ENTERO: el resultado es exactamente el mismo numero que calcula
//   ParlayANN en la CPU, no hay error de redondeo
//
// rango: con 100 dims int8 la distancia maxima es 100*255^2 = 6.5M, cabe en int
__device__ __forceinline__ int distanciaWarp(int miPalabraQ, int normaQ,
                                             const int* __restrict__ palabrasBase,
                                             const int* __restrict__ normasBase,
                                             uint32_t p, int lane) {
    int miPalabraP = __ldg(palabrasBase + (size_t)p * PALABRAS_POR_VEC + lane);
    int prodPunto = sumaWarp(__dp4a(miPalabraQ, miPalabraP, 0));
    return normaQ + __ldg(normasBase + p) - 2 * prodPunto;
}

// MISMA DISTANCIA PERO LA CALCULA UN SOLO HILO
// la usa la busqueda: cada hilo del warp se encarga de un vecino distinto, asi
// 32 vecinos se calculan a la vez sin tener que sumar entre hilos
//
// q4 = la consulta en memoria compartida (32 palabras = 8 int4)
// solo se leen 7 int4 = 112 bytes porque los vectores reales tienen 100 bytes,
// los bytes 100..111 son ceros del relleno y no cambian el resultado
__device__ __forceinline__ int distanciaHilo(const int4* q4, int normaQ,
                                             const int* __restrict__ palabrasBase,
                                             const int* __restrict__ normasBase, uint32_t p) {
    const int4* v4 = reinterpret_cast<const int4*>(palabrasBase + (size_t)p * PALABRAS_POR_VEC);
    int prod = 0;
#pragma unroll
    for (int i = 0; i < 7; i++) {
        int4 a = __ldg(v4 + i);
        int4 b = q4[i];
        prod = __dp4a(a.x, b.x, prod);
        prod = __dp4a(a.y, b.y, prod);
        prod = __dp4a(a.z, b.z, prod);
        prod = __dp4a(a.w, b.w, prod);
    }
    return normaQ + __ldg(normasBase + p) - 2 * prod;
}
