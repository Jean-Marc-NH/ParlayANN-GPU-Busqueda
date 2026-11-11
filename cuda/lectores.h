// lectores.h
// lectura de los archivos del dataset y de los grafos de ParlayANN (en la CPU)
#pragma once

#include "comun.h"

// vectores en formato .i8bin de big-ann-benchmarks
//   [n uint32][dim uint32][n*dim int8]
struct VectoresHost {
    uint32_t n = 0;
    uint32_t dim = 0;
    std::vector<int8_t> datos;   // n*dim, tal cual vienen del archivo

    const int8_t* vec(uint32_t i) const { return datos.data() + (size_t)i * dim; }
};

// grafo guardado por ParlayANN (Graph::save en algorithms/utils/graph.h)
//   [n uint32][gradoMax uint32][grados: n uint32][aristas de todos los puntos seguidas]
struct GrafoHost {
    uint32_t n = 0;
    uint32_t gradoMax = 0;
    std::vector<uint32_t> grados;     // cuantos vecinos tiene cada punto
    std::vector<uint64_t> inicio;     // donde empiezan los vecinos del punto i en "aristas"
    std::vector<uint32_t> aristas;    // todos los vecinos uno detras de otro

    const uint32_t* vecinos(uint32_t i) const { return aristas.data() + inicio[i]; }
};

// ground truth de big-ann-benchmarks
//   [nq uint32][k uint32][ids: nq*k uint32][dists: nq*k float]
struct GroundTruth {
    uint32_t nq = 0;
    uint32_t k = 0;
    std::vector<uint32_t> ids;
    std::vector<float> dists;
};

VectoresHost leerVectoresI8bin(const std::string& ruta);
GrafoHost    leerGrafoParlay(const std::string& ruta);
GroundTruth  leerGroundTruth(const std::string& ruta);

// distancia L2 al cuadrado igual que ParlayANN (euclidian_point.h), para comparar
int distanciaCPU(const int8_t* a, const int8_t* b, uint32_t dim);
