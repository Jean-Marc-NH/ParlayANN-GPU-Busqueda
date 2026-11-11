// paso2_busqueda.cu
// PASO 2 de la comparacion 1: beam search en la GPU sobre los grafos de la CPU
//
// para DiskANN, HCNNG y PyNNDescent corre las 18 configuraciones de
// artifact_eval.yaml y para cada una:
//   - calcula el recall@10 igual que big-ann-benchmarks (con empates)
//   - lo compara con el recall que obtuvo la CPU en su corrida
//     (si los vecinos son los mismos, el recall tiene que ser IDENTICO)
//   - mide el tiempo del kernel (1 corrida de calentamiento + 1 medida)
//
// la medicion seria (mejor de 5, tiempo de punta a punta) va en el paso 5
//
// uso: paso2_busqueda.exe [carpeta_data] [csv_cpu]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "busqueda.cuh"

// resultados de la corrida de CPU en esta misma PC (12 hilos)
const std::string CSV_CPU_DEFAULT =
    "..\\cpu_baseline\\results\\msspacev-1M_t12_20261111_162450\\all_results.csv";

struct FilaCPU {
    double recall;
    double qps;
};

// lee all_results.csv (data_export.py) y agrupa por algoritmo
static std::map<std::string, std::vector<FilaCPU>> leerCsvCPU(const std::string& ruta) {
    std::map<std::string, std::vector<FilaCPU>> res;
    std::ifstream f(ruta);
    if (!f) {
        printf("  (no se encontro el csv de CPU: %s, no se compara)\n", ruta.c_str());
        return res;
    }
    std::string linea;
    std::getline(f, linea);
    std::vector<std::string> cols;
    {
        std::stringstream ss(linea);
        std::string c;
        while (std::getline(ss, c, ',')) cols.push_back(c);
    }
    auto col = [&](const std::string& nombre) {
        for (size_t i = 0; i < cols.size(); i++)
            if (cols[i] == nombre) return (int)i;
        return -1;
    };
    int cAlgo = col("algorithm"), cData = col("dataset"), cHilos = col("threads"),
        cQps = col("qps"), cRec = col("recall/ap");

    while (std::getline(f, linea)) {
        std::vector<std::string> v;
        std::stringstream ss(linea);
        std::string c;
        while (std::getline(ss, c, ',')) v.push_back(c);
        if ((int)v.size() < (int)cols.size()) continue;
        if (v[cData] != "msspacev-1M" || v[cHilos] != "12") continue;
        res[v[cAlgo]].push_back({strtod(v[cRec].c_str(), nullptr), strtod(v[cQps].c_str(), nullptr)});
    }
    return res;
}

// RECALL IGUAL QUE big-ann-benchmarks (benchmark/plotting/metrics.py)
//   compute_recall_with_distance_ties: se cuentan como correctos tambien los
//   puntos del GT que empatan en distancia con el k-esimo
//   recall = promedio de |set(gt[:fin]) & set(resultado)| / k
static double calcularRecall(const GroundTruth& gt, const std::vector<uint32_t>& ids, int k) {
    long long total = 0;
    for (uint32_t q = 0; q < gt.nq; q++) {
        const uint32_t* verdad = &gt.ids[(size_t)q * gt.k];
        const float* dVerdad = &gt.dists[(size_t)q * gt.k];

        int fin = (int)gt.k;
        if ((int)gt.k != k) {
            float ref = dVerdad[k - 1];
            for (int i = k; i < (int)gt.k; i++)
                if (!(fabsf(ref - dVerdad[i]) < 1e-6f)) {
                    fin = i;
                    break;
                }
        }

        // ids distintos devueltos (set): si la GPU repite un id cuenta una vez
        std::vector<uint32_t> mios(ids.begin() + (size_t)q * k, ids.begin() + (size_t)(q + 1) * k);
        std::sort(mios.begin(), mios.end());
        mios.erase(std::unique(mios.begin(), mios.end()), mios.end());
        for (uint32_t x : mios)
            if (std::find(verdad, verdad + fin, x) != verdad + fin) total++;
    }
    return (double)total / gt.nq / k;
}

struct Resultado {
    int Ls;
    long visit;
    double recall;
    double qps;
    double compProm;
    int desbordes, cortas;
    double recallCPU, qpsCPU;   // -1 si no se encontro
};

int main(int argc, char** argv) {
    // "-" en un argumento = usar el valor por defecto
    auto arg = [&](int i, const std::string& porDefecto) {
        return (argc > i && std::string(argv[i]) != "-") ? std::string(argv[i]) : porDefecto;
    };
    std::string carpeta = arg(1, CARPETA_DATOS_DEFAULT);
    if (carpeta.back() != '\\' && carpeta.back() != '/') carpeta += "\\";
    std::string csvCPU = arg(2, CSV_CPU_DEFAULT);
    // opcionales, para perfilar una sola configuracion: algoritmo y Ls
    std::string soloAlgo = (argc > 3) ? argv[3] : "";
    int soloLs = (argc > 4) ? atoi(argv[4]) : 0;

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

    auto filasCPU = leerCsvCPU(csvCPU);

    // ---- las 18 configuraciones de artifact_eval.yaml (msspacev-1M) ----
    // visit = -1 significa sin limite (el wrapper usa G.size())
    std::vector<std::pair<int, long>> configs = {
        {10, 3}, {10, 5}, {10, 8}, {10, 12}, {10, 15}, {15, -1}, {20, -1}, {30, -1}, {40, -1},
        {50, -1}, {60, -1}, {75, -1}, {100, -1}, {150, -1}, {200, -1}, {400, -1}, {500, -1},
        {1000, -1}};

    struct { const char* nombre; const char* nombreCPU; std::string archivo; } grafos[] = {
        {"DiskANN", "ParDiskANN", GRAFO_DISKANN},
        {"HCNNG", "ParHCNNG", GRAFO_HCNNG},
        {"PyNNDescent", "ParPyNNDescent", GRAFO_PYNN},
    };

    std::filesystem::create_directories("resultados");
    // si se filtra por algoritmo/Ls se escribe en otro archivo, para no pisar el completo
    bool parcial = !soloAlgo.empty() || soloLs > 0;
    std::string rutaCsv = parcial ? "resultados\\paso2_parcial.csv" : "resultados\\paso2_busqueda.csv";
    FILE* csv = fopen(rutaCsv.c_str(), "w");
    fprintf(csv, "algoritmo,Ls,visit,recall_gpu,qps_gpu_kernel,comp_prom,recall_cpu,qps_cpu,"
                 "recall_igual,desbordes,fronteras_cortas\n");

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    int totalIguales = 0, totalConfigs = 0;

    for (auto& gr : grafos) {
        if (!soloAlgo.empty() && soloAlgo != gr.nombre) continue;
        GrafoHost g = leerGrafoParlay(carpeta + gr.archivo);
        GrafoGPU gGPU;
        subirGrafo(g, gGPU);
        auto& cpu = filasCPU[gr.nombreCPU];

        printf("\n=== %s (gradoMax %u) ===\n", gr.nombre, g.gradoMax);
        printf("%6s %6s | %9s %9s %6s | %10s %10s %8s | %8s\n", "Ls", "visit", "rec GPU",
               "rec CPU", "igual", "QPS GPU", "QPS CPU", "speedup", "comp/q");

        std::vector<Resultado> res;
        for (auto& c : configs) {
            if (soloLs > 0 && (c.first != soloLs || c.second >= 0)) continue;
            ConfigBusqueda cfg;
            cfg.Ls = c.first;
            cfg.limite = (c.second < 0) ? (long)g.n : c.second;
            cfg.k = 10;
            cfg.cut = 1.35;

            BuscadorGPU bus;
            bus.preparar(gGPU, consultas.n, cfg);

            // calentamiento + medicion
            bus.lanzar(baseGPU, consultasGPU, gGPU);
            CHECK_CUDA(cudaDeviceSynchronize());
#ifdef PERFIL
            reiniciarPerfil();
#endif
            CHECK_CUDA(cudaEventRecord(ev0));
            bus.lanzar(baseGPU, consultasGPU, gGPU);
            CHECK_CUDA(cudaEventRecord(ev1));
            CHECK_CUDA(cudaEventSynchronize(ev1));
            float ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));

#ifdef PERFIL
            {
                // reparto del tiempo de los warps entre las secciones del kernel
                unsigned long long p[NUM_SECCIONES_PERFIL];
                leerPerfil(p);
                const char* nombres[] = {"inicio/salida", "visitar", "filtro/vecinos",
                                         "distancias", "ordenar", "union", "mezcla",
                                         "cut+siguiente"};
                double total = 0;
                for (int s = 0; s < SEC_ITERACIONES; s++) total += (double)p[s];
                double iters = (double)p[SEC_ITERACIONES], nq = (double)p[SEC_CONSULTAS];
                printf("  [perfil] Ls=%d: %.1f vueltas/consulta, %.0f ciclos/vuelta\n", c.first,
                       iters / nq, total / iters);
                for (int s = 0; s < SEC_ITERACIONES; s++)
                    printf("     %-15s %5.1f%%  (%7.0f ciclos/vuelta)\n", nombres[s],
                           100.0 * p[s] / total, p[s] / iters);
            }
#endif

            std::vector<uint32_t> ids((size_t)consultas.n * cfg.k);
            std::vector<int> comp(consultas.n), banderas(consultas.n);
            CHECK_CUDA(cudaMemcpy(ids.data(), bus.idsDev, ids.size() * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(comp.data(), bus.compDev, comp.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(banderas.data(), bus.banderasDev, banderas.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            bus.liberar();

            Resultado r;
            r.Ls = c.first;
            r.visit = c.second;
            r.recall = calcularRecall(gt, ids, cfg.k);
            r.qps = consultas.n / (ms / 1000.0);
            double sumComp = 0;
            r.desbordes = 0;
            r.cortas = 0;
            for (uint32_t i = 0; i < consultas.n; i++) {
                sumComp += comp[i];
                if (banderas[i] & BANDERA_DESBORDE) r.desbordes++;
                if (banderas[i] & BANDERA_FRONTERA_CORTA) r.cortas++;
            }
            r.compProm = sumComp / consultas.n;

            // buscar en la CPU una fila con EXACTAMENTE el mismo recall
            r.recallCPU = -1;
            r.qpsCPU = -1;
            for (auto& fc : cpu)
                if (fabs(fc.recall - r.recall) < 1e-12) {
                    r.recallCPU = fc.recall;
                    r.qpsCPU = fc.qps;
                }
            bool igual = r.recallCPU >= 0;
            totalIguales += igual;
            totalConfigs++;

            char visitTxt[16];
            if (r.visit < 0) snprintf(visitTxt, sizeof(visitTxt), "-");
            else snprintf(visitTxt, sizeof(visitTxt), "%ld", r.visit);
            printf("%6d %6s | %9.5f %9s %6s | %10.0f %10s %8s | %8.0f",
                   r.Ls, visitTxt, r.recall, igual ? "(=)" : "?", igual ? "SI" : "NO", r.qps,
                   igual ? std::to_string((long)r.qpsCPU).c_str() : "-",
                   igual ? (std::to_string(r.qps / r.qpsCPU).substr(0, 5) + "x").c_str() : "-",
                   r.compProm);
            if (r.desbordes || r.cortas) printf("  [desbordes %d, cortas %d]", r.desbordes, r.cortas);
            printf("\n");

            fprintf(csv, "%s,%d,%ld,%.16f,%.2f,%.2f,%.16f,%.2f,%d,%d,%d\n", gr.nombre, r.Ls, r.visit,
                    r.recall, r.qps, r.compProm, r.recallCPU, r.qpsCPU, igual ? 1 : 0, r.desbordes,
                    r.cortas);
            res.push_back(r);
        }

        // si alguna config no coincide, mostrar los recall de la CPU para comparar a mano
        bool todasIguales = std::all_of(res.begin(), res.end(),
                                        [](const Resultado& r) { return r.recallCPU >= 0; });
        if (!todasIguales) {
            std::vector<double> rc;
            for (auto& fc : cpu) rc.push_back(fc.recall);
            std::sort(rc.begin(), rc.end());
            printf("  recalls de la CPU:");
            for (double x : rc) printf(" %.5f", x);
            printf("\n");
        }
        liberarGrafo(gGPU);
    }
    fclose(csv);

    liberarVectores(baseGPU);
    liberarVectores(consultasGPU);
    printf("\nconfiguraciones con recall identico a la CPU: %d de %d\n", totalIguales, totalConfigs);
    printf("resultados en %s\n", rutaCsv.c_str());
    return 0;
}
