// micro_distancias.cu
// microbenchmark: cual es la forma mas rapida de calcular distancias a puntos
// AL AZAR (como en la busqueda) en esta GPU
//
//  A) 1 hilo por distancia   (lo que hace la version 1 de la busqueda)
//  B) 1 warp por distancia   (32 hilos x 4 bytes = 128 bytes seguidos)
//  C) 8 hilos por distancia  (4 distancias por warp, cada hilo lee un int4)
//  D) 4 hilos por distancia  (8 distancias por warp, cada hilo lee 2 int4)
//
// todas leen los mismos pares (consulta, punto) y se comparan los resultados

#include <chrono>
#include <random>

#include "datos_gpu.h"
#include "distancia.cuh"

// A: un hilo, una distancia
__global__ void kA(const int* base, const int* normas, const int* q, const int* nq,
                   const uint32_t* pares, int numPares, int* salida) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPares) return;
    uint32_t idQ = pares[2 * i], idP = pares[2 * i + 1];
    const int4* q4 = reinterpret_cast<const int4*>(q + (size_t)idQ * PALABRAS_POR_VEC);
    salida[i] = distanciaHilo(q4, nq[idQ], base, normas, idP);
}

// B: un warp, una distancia
__global__ void kB(const int* base, const int* normas, const int* q, const int* nq,
                   const uint32_t* pares, int numPares, int* salida) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (i >= numPares) return;
    uint32_t idQ = pares[2 * i], idP = pares[2 * i + 1];
    int miQ = q[(size_t)idQ * PALABRAS_POR_VEC + lane];
    int d = distanciaWarp(miQ, nq[idQ], base, normas, idP, lane);
    if (lane == 0) salida[i] = d;
}

// C / D: grupos de H hilos por distancia, cada hilo lee 112/H bytes en int4
template <int H>
__global__ void kGrupo(const int* base, const int* normas, const int* q, const int* nq,
                       const uint32_t* pares, int numPares, int* salida) {
    int hilo = blockIdx.x * blockDim.x + threadIdx.x;
    int i = hilo / H;            // distancia que le toca a mi grupo
    int enGrupo = hilo % H;      // mi posicion dentro del grupo
    bool activo = i < numPares;
    uint32_t idQ = activo ? pares[2 * i] : 0, idP = activo ? pares[2 * i + 1] : 0;
    const int4* v4 = reinterpret_cast<const int4*>(base + (size_t)idP * PALABRAS_POR_VEC);
    const int4* q4 = reinterpret_cast<const int4*>(q + (size_t)idQ * PALABRAS_POR_VEC);
    int prod = 0;
    // 7 int4 por vector (112 bytes), repartidos entre los H hilos
    for (int j = enGrupo; j < 7; j += H) {
        int4 a = activo ? __ldg(v4 + j) : make_int4(0, 0, 0, 0);
        int4 b = q4[j];
        prod = __dp4a(a.x, b.x, prod);
        prod = __dp4a(a.y, b.y, prod);
        prod = __dp4a(a.z, b.z, prod);
        prod = __dp4a(a.w, b.w, prod);
    }
    for (int salto = H / 2; salto > 0; salto >>= 1) prod += __shfl_xor_sync(0xffffffffu, prod, salto, H);
    if (activo && enGrupo == 0) salida[i] = nq[idQ] + __ldg(normas + idP) - 2 * prod;
}

int main(int argc, char** argv) {
    std::string carpeta = (argc > 1) ? argv[1] : CARPETA_DATOS_DEFAULT;
    VectoresHost base = leerVectoresI8bin(carpeta + ARCH_BASE);
    VectoresHost consultas = leerVectoresI8bin(carpeta + ARCH_CONSULTAS);
    VectoresGPU baseGPU, qGPU;
    subirVectores(base, baseGPU);
    subirVectores(consultas, qGPU);

    const int numPares = 20000000;
    std::vector<uint32_t> pares(2 * (size_t)numPares);
    std::mt19937 rng(1);
    std::uniform_int_distribution<uint32_t> rq(0, consultas.n - 1), rp(0, base.n - 1);
    for (int i = 0; i < numPares; i++) {
        pares[2 * i] = rq(rng);
        pares[2 * i + 1] = rp(rng);
    }
    uint32_t* paresDev;
    int* salDev;
    CHECK_CUDA(cudaMalloc(&paresDev, pares.size() * 4));
    CHECK_CUDA(cudaMalloc(&salDev, (size_t)numPares * 4));
    CHECK_CUDA(cudaMemcpy(paresDev, pares.data(), pares.size() * 4, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<int> ref(numPares), tmp(numPares);

    auto medir = [&](const char* nombre, auto lanzar, bool esRef) {
        lanzar();
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        for (int r = 0; r < 3; r++) lanzar();
        cudaEventRecord(e1);
        CHECK_CUDA(cudaEventSynchronize(e1));
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= 3;
        CHECK_CUDA(cudaMemcpy(tmp.data(), salDev, (size_t)numPares * 4, cudaMemcpyDeviceToHost));
        if (esRef) ref = tmp;
        bool igual = (tmp == ref);
        double distPorSeg = numPares / (ms / 1000.0);
        printf("  %-28s %7.2f ms  %7.0f M dist/s  ~%5.1f GB/s de vectores  %s\n", nombre, ms,
               distPorSeg / 1e6, distPorSeg * 128 / 1e9, igual ? "OK" : "DISTINTO");
    };

    printf("%d distancias a puntos al azar:\n", numPares);
    medir("A) 1 hilo por distancia", [&] {
        kA<<<(numPares + 255) / 256, 256>>>(baseGPU.palabras, baseGPU.normas, qGPU.palabras,
                                          qGPU.normas, paresDev, numPares, salDev);
    }, true);
    medir("B) 1 warp por distancia", [&] {
        kB<<<(int)(((size_t)numPares * 32 + 255) / 256), 256>>>(baseGPU.palabras, baseGPU.normas,
                                                                qGPU.palabras, qGPU.normas,
                                                                paresDev, numPares, salDev);
    }, false);
    medir("C) 8 hilos por distancia", [&] {
        kGrupo<8><<<(int)(((size_t)numPares * 8 + 255) / 256), 256>>>(
            baseGPU.palabras, baseGPU.normas, qGPU.palabras, qGPU.normas, paresDev, numPares,
            salDev);
    }, false);
    medir("D) 4 hilos por distancia", [&] {
        kGrupo<4><<<(int)(((size_t)numPares * 4 + 255) / 256), 256>>>(
            baseGPU.palabras, baseGPU.normas, qGPU.palabras, qGPU.normas, paresDev, numPares,
            salDev);
    }, false);
    return 0;
}
