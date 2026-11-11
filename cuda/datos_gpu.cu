// datos_gpu.cu
// pasa los vectores y los grafos de la CPU a la GPU con el formato que usan los kernels

#include "datos_gpu.h"
#include "distancia.cuh"

// un warp por vector: ||p||^2 = p . p usando el mismo __dp4a que la distancia
__global__ void kernelNormas(const int* __restrict__ palabras, int* __restrict__ normas,
                             uint32_t n) {
    uint32_t idVec = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    // el warp entero sale junto, porque idVec es igual para sus 32 hilos
    if (idVec >= n) return;

    int miPalabra = palabras[(size_t)idVec * PALABRAS_POR_VEC + lane];
    int norma = sumaWarp(__dp4a(miPalabra, miPalabra, 0));
    if (lane == 0) normas[idVec] = norma;
}

void subirVectores(const VectoresHost& host, VectoresGPU& gpu) {
    gpu.n = host.n;

    // RELLENO: cada vector pasa de 100 a 128 bytes, los 28 bytes extra son ceros
    // si esto queda mal, todas las distancias salen mal
    std::vector<int8_t> tmp((size_t)host.n * DIM_PAD, 0);
    for (uint32_t i = 0; i < host.n; i++)
        memcpy(tmp.data() + (size_t)i * DIM_PAD, host.vec(i), host.dim);

    CHECK_CUDA(cudaMalloc(&gpu.palabras, (size_t)host.n * DIM_PAD));
    CHECK_CUDA(cudaMalloc(&gpu.normas, (size_t)host.n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(gpu.palabras, tmp.data(), tmp.size(), cudaMemcpyHostToDevice));

    // las normas se calculan una sola vez aqui, en la GPU
    int hilosPorBloque = 256;   // 8 warps por bloque
    uint64_t hilosTotales = (uint64_t)host.n * 32;
    int bloques = (int)((hilosTotales + hilosPorBloque - 1) / hilosPorBloque);
    kernelNormas<<<bloques, hilosPorBloque>>>(gpu.palabras, gpu.normas, host.n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void liberarVectores(VectoresGPU& gpu) {
    cudaFree(gpu.palabras);
    cudaFree(gpu.normas);
    gpu = VectoresGPU{};
}

void subirGrafo(const GrafoHost& host, GrafoGPU& gpu) {
    gpu.n = host.n;
    gpu.gradoMax = host.gradoMax;

    // armar la matriz n x gradoMax en la CPU, con SIN_VECINO donde sobra espacio
    // OJO: para HCNNG (gradoMax=150) esto son ~600 MB temporales en RAM
    std::vector<uint32_t> tmp((size_t)host.n * host.gradoMax, SIN_VECINO);
    for (uint32_t i = 0; i < host.n; i++)
        memcpy(tmp.data() + (size_t)i * host.gradoMax, host.vecinos(i),
               host.grados[i] * sizeof(uint32_t));

    CHECK_CUDA(cudaMalloc(&gpu.vecinos, tmp.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&gpu.grados, (size_t)host.n * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(gpu.vecinos, tmp.data(), tmp.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu.grados, host.grados.data(), (size_t)host.n * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
}

void liberarGrafo(GrafoGPU& gpu) {
    cudaFree(gpu.vecinos);
    cudaFree(gpu.grados);
    gpu = GrafoGPU{};
}
