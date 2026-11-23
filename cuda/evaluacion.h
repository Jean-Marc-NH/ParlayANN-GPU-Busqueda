// evaluacion.h
// cosas comunes de los programas de prueba: recall y resultados de la CPU
#pragma once

#include <map>

#include "lectores.h"

// resultados de la corrida de CPU (all_results.csv de data_export.py)
// por defecto el de la corrida con 12 hilos; se puede pasar otro por argumento
const std::string CSV_CPU_DEFAULT =
    "..\\cpu_baseline\\results\\msspacev-1M_t12_20261111_162450\\all_results.csv";

struct FilaCPU {
    double recall;
    double qps;
};

// lee el csv de la CPU y agrupa las filas por algoritmo (ParDiskANN, ParHNSW, ...)
std::map<std::string, std::vector<FilaCPU>> leerCsvCPU(const std::string& ruta);

// RECALL IGUAL QUE big-ann-benchmarks (benchmark/plotting/metrics.py)
//   compute_recall_with_distance_ties: tambien cuentan como correctos los
//   puntos del GT que empatan en distancia con el k-esimo
//   recall = promedio de |set(gt[:fin]) & set(resultado)| / k
double calcularRecall(const GroundTruth& gt, const std::vector<uint32_t>& ids, int k);
