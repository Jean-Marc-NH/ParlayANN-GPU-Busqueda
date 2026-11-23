
#pragma once

#include "datos_gpu.h"

struct ConfigBusqueda {
    int Ls = 10;         // ancho del beam (beamSize)
    long limite = 0;     // maximo de nodos a visitar ("visit"); sin visit -> n
    int k = 10;          // vecinos que se devuelven
    double cut = 1.35;   // recorte de la frontera
};


const int BANDERA_DESBORDE = 1;      // se lleno la tabla de visitados (no deberia pasar)
const int BANDERA_FRONTERA_CORTA = 2;  // la frontera termino con menos de k elementos

enum SeccionPerfil {
    SEC_INICIO = 0,        // limpiar tablas, punto inicial, escribir salida
    SEC_VISITAR,           // marcar el actual como visitado
    SEC_FILTRO,            // leer vecinos + filtro hash + guardar candidatos
    SEC_DISTANCIAS,        // calcular distancias
    SEC_ORDENAR,           // ordenar candidatos
    SEC_UNION,             // set_union: decidir cuales entran
    SEC_MEZCLA,            // mover la frontera e insertar
    SEC_CUT_SIGUIENTE,     // cut + buscar el siguiente sin visitar
    SEC_ITERACIONES,       // (contador) vueltas del while
    SEC_CONSULTAS,         // (contador) consultas
    NUM_SECCIONES_PERFIL
};
#ifdef PERFIL
void reiniciarPerfil();
void leerPerfil(unsigned long long* salida);
#endif

// se encarga de la memoria auxiliar y de lanzar el kernel
struct BuscadorGPU {
    // memoria global auxiliar, un pedazo por cada warp que corre a la vez
    uint32_t* filtro = nullptr;      // filtro hash de "ya visto" (igual al de ParlayANN)
    uint32_t* visClaves = nullptr;   // tabla de visitados: id
    int* visCuentas = nullptr;       // tabla de visitados: cuantas veces se visito

    // salida
    uint32_t* idsDev = nullptr;      // numConsultas * k
    int* distsDev = nullptr;         // numConsultas * k
    int* compDev = nullptr;          // distancias calculadas por consulta
    int* banderasDev = nullptr;      // banderas por consulta

    int bitsFiltro = 0;
    uint32_t capVis = 0;
    int warpsPorBloque = 0;
    int bloques = 0;
    size_t bytesCompartida = 0;
    uint32_t numConsultas = 0;
    ConfigBusqueda cfg;

    // reserva memoria para esta configuracion y este grafo
    void preparar(const GrafoGPU& grafo, uint32_t numConsultas, const ConfigBusqueda& cfg);
    // lanza el kernel (asincrono), hay que sincronizar despues
    void lanzar(const VectoresGPU& base, const VectoresGPU& consultas, const GrafoGPU& grafo,
                cudaStream_t stream = 0);
    void liberar();
};
