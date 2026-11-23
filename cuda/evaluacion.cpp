// evaluacion.cpp

#include "evaluacion.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

std::map<std::string, std::vector<FilaCPU>> leerCsvCPU(const std::string& ruta) {
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

    // el csv puede tener corridas con distinto numero de hilos (escalabilidad):
    // se usa la de MAS hilos, que es la corrida principal de esa maquina
    struct Fila { std::string algo; int hilos; FilaCPU dato; };
    std::vector<Fila> filas;
    int maxHilos = -1;
    while (std::getline(f, linea)) {
        std::vector<std::string> v;
        std::stringstream ss(linea);
        std::string c;
        while (std::getline(ss, c, ',')) v.push_back(c);
        if ((int)v.size() < (int)cols.size()) continue;
        if (v[cData] != "msspacev-1M") continue;
        int hilos = atoi(v[cHilos].c_str());
        maxHilos = std::max(maxHilos, hilos);
        filas.push_back({v[cAlgo], hilos,
                         {strtod(v[cRec].c_str(), nullptr), strtod(v[cQps].c_str(), nullptr)}});
    }
    for (auto& fi : filas)
        if (fi.hilos == maxHilos) res[fi.algo].push_back(fi.dato);
    if (maxHilos > 0) printf("  csv de CPU: corrida con %d hilos\n", maxHilos);
    return res;
}

double calcularRecall(const GroundTruth& gt, const std::vector<uint32_t>& ids, int k) {
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
