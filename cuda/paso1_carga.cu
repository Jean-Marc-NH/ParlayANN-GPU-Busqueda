// paso1_carga.cu
// PASO 1 de la comparacion 1: cargar todo en la GPU y comprobar que quedo bien
//
//  1. lee vectores base, consultas y ground truth
//  2. los sube a la GPU (rellenados a 128 bytes) y calcula las normas
//  3. prueba que la distancia de la GPU (con __dp4a) es IGUAL a la de la CPU
//  4. carga uno por uno los grafos de DiskANN, HCNNG y PyNNDescent y verifica
//     que la matriz de vecinos en la GPU coincide con el archivo
//
// uso:  paso1_carga.exe [carpeta_data]

#include <chrono>
#include <cmath>
#include <random>

#include "datos_gpu.h"
#include "distancia.cuh"

using Reloj = std::chrono::steady_clock;

static double segundosDesde(Reloj::time_point t0) {
    return std::chrono::duration<double>(Reloj::now() - t0).count();
}

static void mostrarMemoriaGPU(const char* momento) {
    size_t libre, total;
    CHECK_CUDA(cudaMemGetInfo(&libre, &total));
    printf("  [VRAM %s] usada %.0f MB de %.0f MB\n", momento, (total - libre) / 1048576.0,
           total / 1048576.0);
}

// un warp por pareja (consulta, punto): calcula la distancia igual que lo hara la busqueda
__global__ void kernelProbarDistancias(const int* __restrict__ palabrasQ,
                                       const int* __restrict__ normasQ,
                                       const int* __restrict__ palabrasBase,
                                       const int* __restrict__ normasBase,
                                       const uint32_t* __restrict__ parejas, int numParejas,
                                       int* __restrict__ salida) {
    int idPareja = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (idPareja >= numParejas) return;

    uint32_t idQ = parejas[2 * idPareja];
    uint32_t idP = parejas[2 * idPareja + 1];

    int miPalabraQ = palabrasQ[(size_t)idQ * PALABRAS_POR_VEC + lane];
    int distPuntoA = distanciaWarp(miPalabraQ, normasQ[idQ], palabrasBase, normasBase, idP, lane);
    if (lane == 0) salida[idPareja] = distPuntoA;
}

// ---------------------------------------------------------------------------
// VERIFICACION 1: distancias GPU == distancias CPU (tienen que ser identicas)
// ---------------------------------------------------------------------------
static bool verificarDistancias(const VectoresHost& base, const VectoresHost& consultas,
                                const VectoresGPU& baseGPU, const VectoresGPU& consultasGPU,
                                const GroundTruth& gt) {
    // parejas aleatorias + las parejas del ground truth (consulta, sus 10 vecinos reales)
    const int numAleatorias = 200000;
    const int numConsultasGT = 1000;
    const int kGT = 10;

    std::mt19937 rng(42);   // semilla fija -> siempre las mismas parejas
    std::uniform_int_distribution<uint32_t> randQ(0, consultas.n - 1), randP(0, base.n - 1);

    std::vector<uint32_t> parejas;
    for (int i = 0; i < numAleatorias; i++) {
        parejas.push_back(randQ(rng));
        parejas.push_back(randP(rng));
    }
    for (int q = 0; q < numConsultasGT; q++)
        for (int j = 0; j < kGT; j++) {
            parejas.push_back(q);
            parejas.push_back(gt.ids[(size_t)q * gt.k + j]);
        }
    int numParejas = (int)(parejas.size() / 2);

    uint32_t* parejasDev;
    int* salidaDev;
    CHECK_CUDA(cudaMalloc(&parejasDev, parejas.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&salidaDev, numParejas * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(parejasDev, parejas.data(), parejas.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    int hilosPorBloque = 256;
    int bloques = (numParejas * 32 + hilosPorBloque - 1) / hilosPorBloque;
    kernelProbarDistancias<<<bloques, hilosPorBloque>>>(consultasGPU.palabras, consultasGPU.normas,
                                                        baseGPU.palabras, baseGPU.normas,
                                                        parejasDev, numParejas, salidaDev);
    CHECK_CUDA(cudaGetLastError());
    std::vector<int> distGPU(numParejas);
    CHECK_CUDA(cudaMemcpy(distGPU.data(), salidaDev, numParejas * sizeof(int),
                          cudaMemcpyDeviceToHost));
    cudaFree(parejasDev);
    cudaFree(salidaDev);

    // comparar contra la CPU, deben ser EXACTAMENTE iguales
    int errores = 0;
    for (int i = 0; i < numParejas; i++) {
        int distCPU = distanciaCPU(consultas.vec(parejas[2 * i]), base.vec(parejas[2 * i + 1]),
                                   base.dim);
        if (distCPU != distGPU[i]) {
            if (errores < 5)
                printf("  DIFERENTE pareja %d: cpu=%d gpu=%d\n", i, distCPU, distGPU[i]);
            errores++;
        }
    }
    printf("  distancias GPU vs CPU: %d parejas, %d diferentes\n", numParejas, errores);

    // de paso vemos como guarda el GT sus distancias (al cuadrado o con raiz)
    // solo es informativo, el recall se calcula con los ids
    int igualCuadrado = 0, igualRaiz = 0, totalGT = numConsultasGT * kGT;
    for (int i = numAleatorias; i < numParejas; i++) {
        int q = (i - numAleatorias) / kGT, j = (i - numAleatorias) % kGT;
        float distArchivo = gt.dists[(size_t)q * gt.k + j];
        float tmp = (float)distGPU[i];
        if (fabsf(distArchivo - tmp) <= 1e-3f * fmaxf(1.0f, tmp)) igualCuadrado++;
        if (fabsf(distArchivo - sqrtf(tmp)) <= 1e-3f * fmaxf(1.0f, sqrtf(tmp))) igualRaiz++;
    }
    printf("  ground truth: %d/%d coinciden con dist^2, %d/%d con dist (raiz)\n", igualCuadrado,
           totalGT, igualRaiz, totalGT);

    return errores == 0;
}

// ---------------------------------------------------------------------------
// VERIFICACION 2: la matriz de vecinos en la GPU es igual al archivo del grafo
// ---------------------------------------------------------------------------
static bool verificarGrafo(const GrafoHost& g, const GrafoGPU& gGPU) {
    // grados: se comparan todos
    std::vector<uint32_t> gradosDev(g.n);
    CHECK_CUDA(cudaMemcpy(gradosDev.data(), gGPU.grados, g.n * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    int erroresGrado = 0;
    for (uint32_t i = 0; i < g.n; i++)
        if (gradosDev[i] != g.grados[i]) erroresGrado++;

    // filas de vecinos: 2000 puntos al azar + el primero y el ultimo
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint32_t> randP(0, g.n - 1);
    std::vector<uint32_t> revisar = {0, g.n - 1};
    for (int i = 0; i < 2000; i++) revisar.push_back(randP(rng));

    std::vector<uint32_t> fila(g.gradoMax);
    int erroresFila = 0;
    for (uint32_t p : revisar) {
        CHECK_CUDA(cudaMemcpy(fila.data(), gGPU.vecinos + (size_t)p * g.gradoMax,
                              g.gradoMax * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        bool bien = true;
        for (uint32_t j = 0; j < g.gradoMax; j++) {
            uint32_t esperado = (j < g.grados[p]) ? g.vecinos(p)[j] : SIN_VECINO;
            if (fila[j] != esperado) bien = false;
        }
        if (!bien) erroresFila++;
    }
    printf("  grados distintos: %d de %u | filas distintas: %d de %zu\n", erroresGrado, g.n,
           erroresFila, revisar.size());
    return erroresGrado == 0 && erroresFila == 0;
}

int main(int argc, char** argv) {
    std::string carpeta = (argc > 1) ? argv[1] : CARPETA_DATOS_DEFAULT;
    if (carpeta.back() != '\\' && carpeta.back() != '/') carpeta += "\\";

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | %d SMs | compute %d.%d | %.0f MB\n", prop.name, prop.multiProcessorCount,
           prop.major, prop.minor, prop.totalGlobalMem / 1048576.0);
    printf("Carpeta de datos: %s\n\n", carpeta.c_str());
    bool todoBien = true;

    // ---- 1. leer dataset ---------------------------------------------------
    printf("== Leyendo dataset ==\n");
    auto t0 = Reloj::now();
    VectoresHost base = leerVectoresI8bin(carpeta + ARCH_BASE);
    printf("  base:      %u vectores x %u dims (%.1f s)\n", base.n, base.dim, segundosDesde(t0));
    t0 = Reloj::now();
    VectoresHost consultas = leerVectoresI8bin(carpeta + ARCH_CONSULTAS);
    printf("  consultas: %u vectores x %u dims (%.1f s)\n", consultas.n, consultas.dim,
           segundosDesde(t0));
    t0 = Reloj::now();
    GroundTruth gt = leerGroundTruth(carpeta + ARCH_GT);
    printf("  ground truth: %u consultas x %u vecinos (%.1f s)\n", gt.nq, gt.k, segundosDesde(t0));
    if (gt.nq != consultas.n) {
        printf("  ERROR: el ground truth no tiene la misma cantidad de consultas\n");
        return 1;
    }

    // ---- 2. subir vectores a la GPU ------------------------------------------
    printf("\n== Subiendo vectores a la GPU ==\n");
    mostrarMemoriaGPU("antes");
    VectoresGPU baseGPU, consultasGPU;
    t0 = Reloj::now();
    subirVectores(base, baseGPU);
    subirVectores(consultas, consultasGPU);
    printf("  base %.1f MB + consultas %.1f MB subidos en %.2f s\n", baseGPU.bytes() / 1048576.0,
           consultasGPU.bytes() / 1048576.0, segundosDesde(t0));
    mostrarMemoriaGPU("con vectores");

    // ---- 3. verificar distancias ---------------------------------------------
    printf("\n== Verificacion 1: distancia GPU (__dp4a) vs CPU ==\n");
    bool okDist = verificarDistancias(base, consultas, baseGPU, consultasGPU, gt);
    printf("  -> %s\n", okDist ? "OK, identicas" : "FALLO");
    todoBien &= okDist;

    // ---- 4. grafos, uno a la vez ---------------------------------------------
    struct { const char* nombre; std::string archivo; } grafos[] = {
        {"DiskANN", GRAFO_DISKANN},
        {"HCNNG", GRAFO_HCNNG},
        {"PyNNDescent", GRAFO_PYNN},
    };

    for (auto& gr : grafos) {
        printf("\n== Grafo %s ==\n", gr.nombre);
        t0 = Reloj::now();
        GrafoHost g = leerGrafoParlay(carpeta + gr.archivo);
        double tLectura = segundosDesde(t0);
        printf("  n=%u gradoMax=%u gradoProm=%.1f aristas=%zu (lectura %.1f s)\n", g.n, g.gradoMax,
               (double)g.aristas.size() / g.n, g.aristas.size(), tLectura);
        if (g.n != base.n) {
            printf("  ERROR: el grafo tiene %u puntos y la base %u\n", g.n, base.n);
            todoBien = false;
            continue;
        }

        GrafoGPU gGPU;
        t0 = Reloj::now();
        subirGrafo(g, gGPU);
        printf("  subido a la GPU: %.1f MB en %.2f s\n", gGPU.bytes() / 1048576.0,
               segundosDesde(t0));
        mostrarMemoriaGPU("con este grafo");

        bool okGrafo = verificarGrafo(g, gGPU);
        printf("  -> %s\n", okGrafo ? "OK, igual al archivo" : "FALLO");
        todoBien &= okGrafo;

        liberarGrafo(gGPU);   // se libera antes de cargar el siguiente
    }

    liberarVectores(baseGPU);
    liberarVectores(consultasGPU);

    printf("\n==============================\n");
    printf(todoBien ? "PASO 1 OK: datos y grafos listos en la GPU\n"
                    : "PASO 1 CON ERRORES, revisar arriba\n");
    return todoBien ? 0 : 1;
}
