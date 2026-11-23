// hnsw.h
// grafo HNSW de ParlayANN (algorithms/HNSW/HNSW.hpp) en la CPU y en la GPU
#pragma once

#include "busqueda.cuh"   // ConfigBusqueda y banderas
#include "datos_gpu.h"

// maximo de puntos de entrada que soportamos (en la practica es 1)
const int MAX_ENTRADAS_HNSW = 8;

// archivo del modelo, version 3 (constructor HNSW(filename, getter) de HNSW.hpp):
//   "HNSW" | version u32 | code_U u64 | size_node u64
//   dim u32 | m_l f32 | m u32 | ef_construction u32 | alpha f32 | n u32
//   n veces:            nivel u32 | id del punto u32
//   por nodo y nivel:   cantidad u64 | ids de los vecinos (u32)
//   entradas:           cantidad u64 | ids (u32)
struct HnswHost {
    uint32_t n = 0, dim = 0, m = 0, efc = 0;
    float ml = 0, alpha = 0;
    std::vector<uint32_t> nivel;          // nivel mas alto de cada nodo
    std::vector<uint32_t> datoDe;         // que punto de la base guarda cada nodo
    std::vector<uint64_t> primeraLista;   // indice (en iniLista) de la lista de nivel 0 del nodo
    std::vector<uint64_t> iniLista;       // donde empieza cada lista en "ids"
    std::vector<uint32_t> tamLista;       // cuantos vecinos tiene cada lista
    std::vector<uint32_t> ids;            // todos los vecinos seguidos
    std::vector<uint32_t> entradas;       // puntos de entrada (nivel mas alto)
    uint32_t grado0Max = 0;               // grado maximo en el nivel 0
    uint32_t gradoArribaMax = 0;          // grado maximo en los niveles 1, 2, ...

    // vecinos del nodo i en el nivel l (l <= nivel[i])
    const uint32_t* vecinos(uint32_t i, uint32_t l) const {
        return ids.data() + iniLista[primeraLista[i] + l];
    }
    uint32_t grado(uint32_t i, uint32_t l) const { return tamLista[primeraLista[i] + l]; }
};

HnswHost leerHnsw(const std::string& ruta);

// el grafo en la GPU
//  - nivel 0: matriz n x G0 rellenada con SIN_VECINO (como los otros grafos)
//  - niveles de arriba: bloques de GA vecinos. El nodo i usa los bloques
//    inicioArriba[i] + (l-1) para l = 1..nivel[i]   (SIN_VECINO si nivel[i] = 0)
struct HnswGPU {
    uint32_t n = 0;
    int G0 = 0, GA = 0;
    int nivelSuperior = 0;               // nivel del primer punto de entrada
    int numEntradas = 0;
    uint32_t* vecinos0 = nullptr;
    uint32_t* vecinosArriba = nullptr;
    uint32_t* inicioArriba = nullptr;
    uint32_t* datoDe = nullptr;
    uint32_t* entradas = nullptr;
    // vectores ordenados por NODO (vector del nodo i en la posicion i).
    // si datoDe es la identidad se usan directo los de la base
    VectoresGPU vecNodos;
    bool vecPropios = false;             // true si vecNodos se reservo aparte
    size_t bytes = 0;
};

void subirHnsw(const HnswHost& h, const VectoresHost& base, const VectoresGPU& baseGPU,
               HnswGPU& gpu);
void liberarHnsw(HnswGPU& gpu);

// memoria auxiliar y lanzamiento del kernel de HNSW
struct BuscadorHnsw {
    unsigned long long* visitados = nullptr;   // filtro de visitados con epoca, por warp
    unsigned long long* inw = nullptr;         // tabla "ya entro al workset" con epoca, por warp

    uint32_t* idsDev = nullptr;
    int* distsDev = nullptr;
    int* compDev = nullptr;
    int* banderasDev = nullptr;

    int bitsVis0 = 0;
    uint32_t capInw = 0;
    uint32_t epocaBase = 0;   // sube en cada lanzamiento para no repetir epocas
    int warpsPorBloque = 0;
    int bloques = 0;
    size_t bytesCompartida = 0;
    uint32_t numConsultas = 0;
    ConfigBusqueda cfg;

    void preparar(const HnswGPU& grafo, uint32_t numConsultas, const ConfigBusqueda& cfg);
    void lanzar(const VectoresGPU& consultas, const HnswGPU& grafo, cudaStream_t stream = 0);
    void liberar();
};
