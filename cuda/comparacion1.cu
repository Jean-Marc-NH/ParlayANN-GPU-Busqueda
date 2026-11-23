// comparacion1.cu
// COMPARACION 1 COMPLETA: busqueda CPU vs GPU sobre LOS MISMOS GRAFOS
//
// para DiskANN, HCNNG, PyNNDescent y HNSW, con las 18 configuraciones de
// artifact_eval.yaml (msspacev-1M):
//   - recall@10 igual que big-ann-benchmarks (con empates)
//   - QPS SOLO KERNEL: mejor de N corridas (cudaEvent)
//   - QPS DE PUNTA A PUNTA: mejor de N corridas, con el reloj de la CPU:
//       copiar las consultas a la GPU + normas + kernel + traer ids y distancias
//     (la CPU tambien recibe las consultas y devuelve ids y distancias)
//   - guarda los vecinos de cada configuracion en resultados\vecinos\ para
//     compararlos uno por uno con los de la CPU (reporte_comparacion1.py)
//
// mismo protocolo que big-ann-benchmarks: 1 corrida de calentamiento y despues
// se queda con la MEJOR de N (por defecto 5)
//
// uso: comparacion1.exe [carpeta_data] [csv_cpu] [algoritmo] [Ls] [corridas]
//   "-" = valor por defecto. algoritmo: ParDiskANN, ParHCNNG, ParPyNNDescent, ParHNSW
//   Con algoritmo/Ls se corre solo eso y el csv va a comparacion1_parcial.csv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>

#include "evaluacion.h"
#include "hnsw.h"

using Reloj = std::chrono::steady_clock;

// las 18 configuraciones de artifact_eval.yaml (msspacev-1M); visit -1 = sin limite
static const std::vector<std::pair<int, long>> CONFIGS = {
    {10, 3}, {10, 5}, {10, 8}, {10, 12}, {10, 15}, {15, -1}, {20, -1}, {30, -1}, {40, -1},
    {50, -1}, {60, -1}, {75, -1}, {100, -1}, {150, -1}, {200, -1}, {400, -1}, {500, -1},
    {1000, -1}};

// lo que hay que saber de un buscador ya preparado (sirve para los dos kernels)
struct BuscadorListo {
    std::function<void(cudaStream_t)> lanzar;   // lanza SOLO el kernel
    uint32_t* idsDev;
    int* distsDev;
    int* compDev;
    int* banderasDev;
};

struct Medicion {
    double recall = 0, qpsKernel = 0, qpsTotal = 0, compProm = 0;
    int desbordes = 0, cortas = 0;
    bool determinista = true;   // el calentamiento y la ultima corrida dan lo mismo
    std::vector<uint32_t> ids;
};

// buffers fijados (pinned) para que las copias CPU <-> GPU sean rapidas
struct BuffersHost {
    int8_t* consultas = nullptr;   // consultas rellenadas a 128 bytes
    uint32_t* ids = nullptr;
    int* dists = nullptr;
};

static Medicion medir(const BuscadorListo& b, VectoresGPU& consultasGPU, const BuffersHost& h,
                      const GroundTruth& gt, int k, int corridas, cudaStream_t s) {
    const uint32_t nq = consultasGPU.n;
    const size_t bytesIds = (size_t)nq * k * sizeof(uint32_t);
    const size_t bytesDists = (size_t)nq * k * sizeof(int);

    // UNA CORRIDA COMPLETA, tal como la veria quien usa la busqueda
    auto puntaAPunta = [&]() {
        copiarVectoresAsync(h.consultas, consultasGPU, s);
        b.lanzar(s);
        CHECK_CUDA(cudaMemcpyAsync(h.ids, b.idsDev, bytesIds, cudaMemcpyDeviceToHost, s));
        CHECK_CUDA(cudaMemcpyAsync(h.dists, b.distsDev, bytesDists, cudaMemcpyDeviceToHost, s));
        CHECK_CUDA(cudaStreamSynchronize(s));
    };

    // calentamiento (se guardan sus vecinos para comprobar el determinismo)
    puntaAPunta();
    std::vector<uint32_t> idsCalentamiento(h.ids, h.ids + (size_t)nq * k);

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    double mejorKernel = 1e30, mejorTotal = 1e30;

    // SOLO KERNEL: las consultas ya estan en la GPU
    for (int r = 0; r < corridas; r++) {
        CHECK_CUDA(cudaEventRecord(e0, s));
        b.lanzar(s);
        CHECK_CUDA(cudaEventRecord(e1, s));
        CHECK_CUDA(cudaEventSynchronize(e1));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        mejorKernel = std::min(mejorKernel, ms / 1000.0);
    }
    // DE PUNTA A PUNTA: reloj de la CPU, como time.perf_counter() en big-ann-benchmarks
    for (int r = 0; r < corridas; r++) {
        auto t0 = Reloj::now();
        puntaAPunta();
        double seg = std::chrono::duration<double>(Reloj::now() - t0).count();
        mejorTotal = std::min(mejorTotal, seg);
    }
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));

    Medicion m;
    m.ids.assign(h.ids, h.ids + (size_t)nq * k);
    // DETERMINISMO: varias corridas sobre el mismo grafo tienen que dar exactamente
    // los mismos vecinos (si no, hay estado que se arrastra entre lanzamientos)
    m.determinista = (m.ids == idsCalentamiento);
    std::vector<int> comp(nq), banderas(nq);
    CHECK_CUDA(cudaMemcpy(comp.data(), b.compDev, nq * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(banderas.data(), b.banderasDev, nq * sizeof(int), cudaMemcpyDeviceToHost));
    double suma = 0;
    for (uint32_t i = 0; i < nq; i++) {
        suma += comp[i];
        if (banderas[i] & BANDERA_DESBORDE) m.desbordes++;
        if (banderas[i] & BANDERA_FRONTERA_CORTA) m.cortas++;
    }
    m.compProm = suma / nq;
    m.recall = calcularRecall(gt, m.ids, k);
    m.qpsKernel = nq / mejorKernel;
    m.qpsTotal = nq / mejorTotal;
    return m;
}

// vecinos en binario: [nq u32][k u32][nq*k u32]  (0xFFFFFFFF = sin vecino)
static void guardarVecinos(const std::string& ruta, const std::vector<uint32_t>& ids, uint32_t nq,
                           uint32_t k) {
    FILE* f = fopen(ruta.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "No se pudo escribir %s\n", ruta.c_str());
        exit(1);
    }
    fwrite(&nq, 4, 1, f);
    fwrite(&k, 4, 1, f);
    fwrite(ids.data(), 4, ids.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    auto arg = [&](int i, const std::string& porDefecto) {
        return (argc > i && std::string(argv[i]) != "-") ? std::string(argv[i]) : porDefecto;
    };
    std::string carpeta = arg(1, CARPETA_DATOS_DEFAULT);
    if (carpeta.back() != '\\' && carpeta.back() != '/') carpeta += "\\";
    std::string csvCPU = arg(2, CSV_CPU_DEFAULT);
    std::string soloAlgo = arg(3, "");
    int soloLs = atoi(arg(4, "0").c_str());
    int corridas = std::max(1, atoi(arg(5, "5").c_str()));
    const int K = 10;
    bool parcial = !soloAlgo.empty() || soloLs > 0;

    // ---- equipo ----
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int verDriver = 0, verRuntime = 0;
    cudaDriverGetVersion(&verDriver);
    cudaRuntimeGetVersion(&verRuntime);
    std::filesystem::create_directories("resultados\\vecinos");
    {
        FILE* f = fopen("resultados\\comparacion1_equipo.txt", "w");
        fprintf(f, "gpu: %s\nsms: %d\ncompute: %d.%d\nmemoria_mb: %.0f\n", prop.name,
                prop.multiProcessorCount, prop.major, prop.minor, prop.totalGlobalMem / 1048576.0);
        fprintf(f, "cuda_driver: %d\ncuda_runtime: %d\ncorridas: %d\n", verDriver, verRuntime,
                corridas);
        fclose(f);
    }
    printf("GPU: %s | %d SMs | %d corridas por configuracion (se usa la mejor)\n", prop.name,
           prop.multiProcessorCount, corridas);

    // ---- datos ----
    VectoresHost base = leerVectoresI8bin(carpeta + ARCH_BASE);
    VectoresHost consultas = leerVectoresI8bin(carpeta + ARCH_CONSULTAS);
    GroundTruth gt = leerGroundTruth(carpeta + ARCH_GT);
    VectoresGPU baseGPU, consultasGPU;
    subirVectores(base, baseGPU);
    subirVectores(consultas, consultasGPU);
    const uint32_t nq = consultas.n;
    printf("base %u, consultas %u, gt k=%u\n", base.n, nq, gt.k);

    BuffersHost h;
    CHECK_CUDA(cudaMallocHost(&h.consultas, (size_t)nq * DIM_PAD));
    CHECK_CUDA(cudaMallocHost(&h.ids, (size_t)nq * K * sizeof(uint32_t)));
    CHECK_CUDA(cudaMallocHost(&h.dists, (size_t)nq * K * sizeof(int)));
    rellenarVectores(consultas, h.consultas);   // se hace una vez, como tener el arreglo listo
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));

    auto filasCPU = leerCsvCPU(csvCPU);

    std::string rutaCsv = parcial ? "resultados\\comparacion1_parcial.csv"
                                  : "resultados\\comparacion1_gpu.csv";
    FILE* csv = fopen(rutaCsv.c_str(), "w");
    fprintf(csv, "algoritmo,Ls,visit,recall_gpu,qps_gpu_kernel,qps_gpu_total,comp_prom,"
                 "recall_cpu_csv,qps_cpu_csv,recall_igual,desbordes,fronteras_cortas,determinista\n");

    const char* algoritmos[] = {"ParDiskANN", "ParHCNNG", "ParPyNNDescent", "ParHNSW"};
    int iguales = 0, total = 0, noDeterministas = 0;

    for (const char* algo : algoritmos) {
        if (!soloAlgo.empty() && soloAlgo != algo) continue;
        bool esHnsw = std::string(algo) == "ParHNSW";

        // ---- cargar el grafo de este algoritmo ----
        GrafoGPU grafo;
        HnswGPU hnsw;
        auto t0 = Reloj::now();
        if (esHnsw) {
            HnswHost hh = leerHnsw(rutaGrafo(carpeta, GRAFO_HNSW));
            subirHnsw(hh, base, baseGPU, hnsw);
        } else {
            UbicacionGrafo ubic = (std::string(algo) == "ParDiskANN") ? GRAFO_DISKANN
                                  : (std::string(algo) == "ParHCNNG")  ? GRAFO_HCNNG
                                                                       : GRAFO_PYNN;
            GrafoHost g = leerGrafoParlay(rutaGrafo(carpeta, ubic));
            subirGrafo(g, grafo);
        }
        printf("\n=== %s (grafo cargado en %.1f s) ===\n", algo,
               std::chrono::duration<double>(Reloj::now() - t0).count());
        printf("%6s %6s | %9s %6s | %11s %11s %10s | %7s %7s | %7s\n", "Ls", "visit", "recall",
               "=CPU", "QPS kernel", "QPS total", "QPS CPU", "x kern", "x total", "comp/q");
        auto& cpu = filasCPU[algo];

        for (auto& c : CONFIGS) {
            if (soloLs > 0 && (c.first != soloLs || c.second >= 0)) continue;
            ConfigBusqueda cfg;
            cfg.Ls = c.first;
            cfg.k = K;

            Medicion m;
            if (esHnsw) {
                cfg.limite = c.second;   // -1 = sin limite (el buscador usa n)
                BuscadorHnsw bus;
                bus.preparar(hnsw, nq, cfg);
                BuscadorListo b{[&](cudaStream_t st) { bus.lanzar(consultasGPU, hnsw, st); },
                                bus.idsDev, bus.distsDev, bus.compDev, bus.banderasDev};
                m = medir(b, consultasGPU, h, gt, K, corridas, s);
                bus.liberar();
            } else {
                cfg.limite = (c.second < 0) ? (long)grafo.n : c.second;
                BuscadorGPU bus;
                bus.preparar(grafo, nq, cfg);
                BuscadorListo b{[&](cudaStream_t st) { bus.lanzar(baseGPU, consultasGPU, grafo, st); },
                                bus.idsDev, bus.distsDev, bus.compDev, bus.banderasDev};
                m = medir(b, consultasGPU, h, gt, K, corridas, s);
                bus.liberar();
            }

            // vecinos para la validacion (mismo nombre que los hdf5 de la CPU: _Ls_X o _Ls_X_visit_Y)
            std::string nombre = std::string(algo) + "_Ls_" + std::to_string(c.first);
            if (c.second >= 0) nombre += "_visit_" + std::to_string(c.second);
            guardarVecinos("resultados\\vecinos\\" + nombre + ".bin", m.ids, nq, K);

            // comparacion rapida con el csv de la CPU (el reporte usa los hdf5)
            double recCPU = -1, qpsCPU = -1;
            for (auto& fc : cpu)
                if (fabs(fc.recall - m.recall) < 1e-12) {
                    recCPU = fc.recall;
                    qpsCPU = fc.qps;
                }
            bool igual = recCPU >= 0;
            iguales += igual;
            total++;

            char visitTxt[16], xk[16], xt[16], qc[16];
            snprintf(visitTxt, sizeof(visitTxt), c.second < 0 ? "-" : "%ld", c.second);
            snprintf(qc, sizeof(qc), igual ? "%.0f" : "-", qpsCPU);
            snprintf(xk, sizeof(xk), igual ? "%.2fx" : "-", m.qpsKernel / qpsCPU);
            snprintf(xt, sizeof(xt), igual ? "%.2fx" : "-", m.qpsTotal / qpsCPU);
            printf("%6d %6s | %9.5f %6s | %11.0f %11.0f %10s | %7s %7s | %7.0f", c.first, visitTxt,
                   m.recall, igual ? "SI" : "NO", m.qpsKernel, m.qpsTotal, qc, xk, xt, m.compProm);
            if (m.desbordes || m.cortas) printf("  [desbordes %d, cortas %d]", m.desbordes, m.cortas);
            if (!m.determinista) {
                printf("  [NO DETERMINISTA: el calentamiento dio otros vecinos]");
                noDeterministas++;
            }
            printf("\n");
            fflush(stdout);

            fprintf(csv, "%s,%d,%ld,%.16f,%.2f,%.2f,%.2f,%.16f,%.2f,%d,%d,%d,%d\n", algo, c.first,
                    c.second, m.recall, m.qpsKernel, m.qpsTotal, m.compProm, recCPU, qpsCPU,
                    igual ? 1 : 0, m.desbordes, m.cortas, m.determinista ? 1 : 0);
            fflush(csv);
        }

        if (esHnsw) liberarHnsw(hnsw);
        else liberarGrafo(grafo);
    }
    fclose(csv);

    cudaFreeHost(h.consultas);
    cudaFreeHost(h.ids);
    cudaFreeHost(h.dists);
    cudaStreamDestroy(s);
    liberarVectores(baseGPU);
    liberarVectores(consultasGPU);

    printf("\nconfiguraciones con recall identico a la CPU: %d de %d\n", iguales, total);
    printf("configuraciones deterministas (calentamiento == ultima corrida): %d de %d\n",
           total - noDeterministas, total);
    printf("resultados en %s y resultados\\vecinos\\\n", rutaCsv.c_str());
    printf("siguiente: reporte_comparacion1.py (validacion vecino por vecino, tablas y graficas)\n");
    return 0;
}
