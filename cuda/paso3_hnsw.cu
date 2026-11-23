// paso3_hnsw.cu
// PASO 3 de la comparacion 1: busqueda HNSW en la GPU sobre el grafo de la CPU
//
// corre las 18 configuraciones de artifact_eval.yaml (msspacev-1M, ParHNSW) y:
//   - calcula el recall@10 igual que big-ann-benchmarks (con empates)
//   - lo compara con el recall que obtuvo la CPU (tiene que ser IDENTICO)
//   - mide el tiempo del kernel (1 corrida de calentamiento + 1 medida)
//
// uso: paso3_hnsw.exe [carpeta_data] [csv_cpu] [Ls]
//   "-" = valor por defecto. Con Ls se corre solo esa configuracion.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

#include "evaluacion.h"
#include "hnsw.h"

using Reloj = std::chrono::steady_clock;

int main(int argc, char** argv) {
    auto arg = [&](int i, const std::string& porDefecto) {
        return (argc > i && std::string(argv[i]) != "-") ? std::string(argv[i]) : porDefecto;
    };
    std::string carpeta = arg(1, CARPETA_DATOS_DEFAULT);
    if (carpeta.back() != '\\' && carpeta.back() != '/') carpeta += "\\";
    std::string csvCPU = arg(2, CSV_CPU_DEFAULT);
    int soloLs = (argc > 3) ? atoi(argv[3]) : 0;

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);

    // ---- datos ----
    VectoresHost base = leerVectoresI8bin(carpeta + ARCH_BASE);
    VectoresHost consultas = leerVectoresI8bin(carpeta + ARCH_CONSULTAS);
    GroundTruth gt = leerGroundTruth(carpeta + ARCH_GT);
    VectoresGPU baseGPU, consultasGPU;
    subirVectores(base, baseGPU);
    subirVectores(consultas, consultasGPU);
    printf("base %u, consultas %u, gt k=%u\n", base.n, consultas.n, gt.k);

    // ---- grafo HNSW ----
    auto t0 = Reloj::now();
    HnswHost h = leerHnsw(rutaGrafo(carpeta, GRAFO_HNSW));
    double tLectura = std::chrono::duration<double>(Reloj::now() - t0).count();
    if (h.n != base.n || h.dim != base.dim) {
        printf("ERROR: el grafo tiene n=%u dim=%u y la base n=%u dim=%u\n", h.n, h.dim, base.n,
               base.dim);
        return 1;
    }
    uint32_t nivelMax = *std::max_element(h.nivel.begin(), h.nivel.end());
    std::vector<uint32_t> porNivel(nivelMax + 1, 0);
    for (uint32_t x : h.nivel) porNivel[x]++;
    printf("\n=== HNSW: m=%u efc=%u ml=%.2f alpha=%.2f (lectura %.1f s) ===\n", h.m, h.efc, h.ml,
           h.alpha, tLectura);
    printf("  grado max nivel 0 = %u, niveles de arriba = %u\n", h.grado0Max, h.gradoArribaMax);
    printf("  nodos por nivel mas alto:");
    for (uint32_t l = 0; l <= nivelMax; l++) printf(" [%u]=%u", l, porNivel[l]);
    printf("\n  puntos de entrada: %zu (nivel %u)\n", h.entradas.size(), h.nivel[h.entradas[0]]);

    HnswGPU g;
    subirHnsw(h, base, baseGPU, g);
    printf("  en la GPU: %.1f MB%s\n", g.bytes / 1048576.0,
           g.vecPropios ? " (incluye vectores reordenados por nodo)" : "");

    auto filasCPU = leerCsvCPU(csvCPU);
    auto& cpu = filasCPU["ParHNSW"];

    // ---- las 18 configuraciones de artifact_eval.yaml (msspacev-1M) ----
    std::vector<std::pair<int, long>> configs = {
        {10, 3}, {10, 5}, {10, 8}, {10, 12}, {10, 15}, {15, -1}, {20, -1}, {30, -1}, {40, -1},
        {50, -1}, {60, -1}, {75, -1}, {100, -1}, {150, -1}, {200, -1}, {400, -1}, {500, -1},
        {1000, -1}};

    std::filesystem::create_directories("resultados");
    std::string rutaCsv = (soloLs > 0) ? "resultados\\paso3_parcial.csv" : "resultados\\paso3_hnsw.csv";
    FILE* csv = fopen(rutaCsv.c_str(), "w");
    fprintf(csv, "algoritmo,Ls,visit,recall_gpu,qps_gpu_kernel,comp_prom,recall_cpu,qps_cpu,"
                 "recall_igual,desbordes,fronteras_cortas\n");

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    int iguales = 0, total = 0;

    printf("\n%6s %6s | %9s %9s %6s | %10s %10s %8s | %8s\n", "Ls", "visit", "rec GPU", "rec CPU",
           "igual", "QPS GPU", "QPS CPU", "speedup", "comp/q");
    for (auto& c : configs) {
        if (soloLs > 0 && (c.first != soloLs || c.second >= 0)) continue;
        ConfigBusqueda cfg;
        cfg.Ls = c.first;
        cfg.limite = c.second;   // -1 = sin limite (el buscador usa n)
        cfg.k = 10;

        BuscadorHnsw bus;
        bus.preparar(g, consultas.n, cfg);

        // calentamiento + medicion
        bus.lanzar(consultasGPU, g);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev0));
        bus.lanzar(consultasGPU, g);
        CHECK_CUDA(cudaEventRecord(ev1));
        CHECK_CUDA(cudaEventSynchronize(ev1));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));

        std::vector<uint32_t> ids((size_t)consultas.n * cfg.k);
        std::vector<int> comp(consultas.n), banderas(consultas.n);
        CHECK_CUDA(cudaMemcpy(ids.data(), bus.idsDev, ids.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(comp.data(), bus.compDev, comp.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(banderas.data(), bus.banderasDev, banderas.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
        bus.liberar();

        double recall = calcularRecall(gt, ids, cfg.k);
        double qps = consultas.n / (ms / 1000.0);
        double sumComp = 0;
        int desbordes = 0, cortas = 0;
        for (uint32_t i = 0; i < consultas.n; i++) {
            sumComp += comp[i];
            if (banderas[i] & BANDERA_DESBORDE) desbordes++;
            if (banderas[i] & BANDERA_FRONTERA_CORTA) cortas++;
        }

        // buscar en la CPU una fila con EXACTAMENTE el mismo recall
        double recCPU = -1, qpsCPU = -1;
        for (auto& fc : cpu)
            if (fabs(fc.recall - recall) < 1e-12) {
                recCPU = fc.recall;
                qpsCPU = fc.qps;
            }
        bool igual = recCPU >= 0;
        iguales += igual;
        total++;

        char visitTxt[16];
        if (c.second < 0) snprintf(visitTxt, sizeof(visitTxt), "-");
        else snprintf(visitTxt, sizeof(visitTxt), "%ld", c.second);
        printf("%6d %6s | %9.5f %9s %6s | %10.0f %10s %8s | %8.0f", c.first, visitTxt, recall,
               igual ? "(=)" : "?", igual ? "SI" : "NO", qps,
               igual ? std::to_string((long)qpsCPU).c_str() : "-",
               igual ? (std::to_string(qps / qpsCPU).substr(0, 5) + "x").c_str() : "-",
               sumComp / consultas.n);
        if (desbordes || cortas) printf("  [desbordes %d, cortas %d]", desbordes, cortas);
        printf("\n");

        fprintf(csv, "HNSW,%d,%ld,%.16f,%.2f,%.2f,%.16f,%.2f,%d,%d,%d\n", c.first, c.second, recall,
                qps, sumComp / consultas.n, recCPU, qpsCPU, igual ? 1 : 0, desbordes, cortas);
    }
    fclose(csv);

    if (iguales < total && !cpu.empty()) {
        std::vector<double> rc;
        for (auto& fc : cpu) rc.push_back(fc.recall);
        std::sort(rc.begin(), rc.end());
        printf("  recalls de la CPU:");
        for (double x : rc) printf(" %.5f", x);
        printf("\n");
    }

    liberarHnsw(g);
    liberarVectores(baseGPU);
    liberarVectores(consultasGPU);
    printf("\nconfiguraciones con recall identico a la CPU: %d de %d\n", iguales, total);
    printf("resultados en %s\n", rutaCsv.c_str());
    return 0;
}
