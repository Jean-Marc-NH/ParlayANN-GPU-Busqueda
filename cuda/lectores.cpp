// lectores.cpp
// aqui solo se lee de disco y se valida que los tamanos cuadren, nada de GPU

#include "lectores.h"

#ifdef _MSC_VER
#define fseek64 _fseeki64
#define ftell64 _ftelli64
#else
#define fseek64 fseeko
#define ftell64 ftello
#endif

static FILE* abrir(const std::string& ruta, uint64_t& tamanoArchivo) {
    FILE* f = fopen(ruta.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "No se pudo abrir: %s\n", ruta.c_str());
        exit(1);
    }
    fseek64(f, 0, SEEK_END);
    tamanoArchivo = (uint64_t)ftell64(f);
    fseek64(f, 0, SEEK_SET);
    return f;
}

// lee "cuantos" elementos y se asegura de que realmente se leyeron todos
template <typename T>
static void leerExacto(FILE* f, T* destino, size_t cuantos, const std::string& ruta) {
    size_t leidos = fread(destino, sizeof(T), cuantos, f);
    if (leidos != cuantos) {
        fprintf(stderr, "Archivo incompleto: %s (lei %zu de %zu)\n", ruta.c_str(), leidos, cuantos);
        exit(1);
    }
}

VectoresHost leerVectoresI8bin(const std::string& ruta) {
    uint64_t tamano;
    FILE* f = abrir(ruta, tamano);

    VectoresHost v;
    uint32_t cabecera[2];
    leerExacto(f, cabecera, 2, ruta);
    v.n = cabecera[0];
    v.dim = cabecera[1];

    uint64_t esperado = 8 + (uint64_t)v.n * v.dim;
    if (tamano != esperado) {
        fprintf(stderr, "%s: tamano %llu pero la cabecera dice %llu\n", ruta.c_str(),
                (unsigned long long)tamano, (unsigned long long)esperado);
        exit(1);
    }
    if (v.dim != DIM_REAL) {
        fprintf(stderr, "%s: dimension %u, el codigo espera %d\n", ruta.c_str(), v.dim, DIM_REAL);
        exit(1);
    }

    v.datos.resize((size_t)v.n * v.dim);
    leerExacto(f, v.datos.data(), v.datos.size(), ruta);
    fclose(f);
    return v;
}

GrafoHost leerGrafoParlay(const std::string& ruta) {
    uint64_t tamano;
    FILE* f = abrir(ruta, tamano);

    GrafoHost g;
    uint32_t cabecera[2];
    leerExacto(f, cabecera, 2, ruta);
    g.n = cabecera[0];
    g.gradoMax = cabecera[1];

    g.grados.resize(g.n);
    leerExacto(f, g.grados.data(), g.n, ruta);

    // posicion de inicio de cada lista de vecinos (suma acumulada de grados)
    g.inicio.resize((size_t)g.n + 1);
    g.inicio[0] = 0;
    for (uint32_t i = 0; i < g.n; i++) {
        if (g.grados[i] > g.gradoMax) {
            fprintf(stderr, "%s: el punto %u tiene grado %u > gradoMax %u\n", ruta.c_str(), i,
                    g.grados[i], g.gradoMax);
            exit(1);
        }
        g.inicio[i + 1] = g.inicio[i] + g.grados[i];
    }
    uint64_t totalAristas = g.inicio[g.n];

    // EL TAMANO DEL ARCHIVO TIENE QUE CUADRAR EXACTO, SI NO EL FORMATO NO ES EL QUE CREEMOS
    uint64_t esperado = 8 + 4ull * g.n + 4ull * totalAristas;
    if (tamano != esperado) {
        fprintf(stderr, "%s: tamano %llu pero deberia ser %llu\n", ruta.c_str(),
                (unsigned long long)tamano, (unsigned long long)esperado);
        exit(1);
    }

    g.aristas.resize(totalAristas);
    leerExacto(f, g.aristas.data(), g.aristas.size(), ruta);
    fclose(f);

    // ningun vecino puede apuntar fuera del rango de puntos
    for (uint64_t e = 0; e < totalAristas; e++) {
        if (g.aristas[e] >= g.n) {
            fprintf(stderr, "%s: arista %llu apunta a %u (n=%u)\n", ruta.c_str(),
                    (unsigned long long)e, g.aristas[e], g.n);
            exit(1);
        }
    }
    return g;
}

GroundTruth leerGroundTruth(const std::string& ruta) {
    uint64_t tamano;
    FILE* f = abrir(ruta, tamano);

    GroundTruth gt;
    uint32_t cabecera[2];
    leerExacto(f, cabecera, 2, ruta);
    gt.nq = cabecera[0];
    gt.k = cabecera[1];

    uint64_t total = (uint64_t)gt.nq * gt.k;
    uint64_t esperado = 8 + total * 4 + total * 4;
    if (tamano != esperado) {
        fprintf(stderr, "%s: tamano %llu pero deberia ser %llu\n", ruta.c_str(),
                (unsigned long long)tamano, (unsigned long long)esperado);
        exit(1);
    }

    gt.ids.resize(total);
    gt.dists.resize(total);
    leerExacto(f, gt.ids.data(), total, ruta);
    leerExacto(f, gt.dists.data(), total, ruta);
    fclose(f);
    return gt;
}

int distanciaCPU(const int8_t* a, const int8_t* b, uint32_t dim) {
    int suma = 0;
    for (uint32_t i = 0; i < dim; i++) {
        int dif = (int)a[i] - (int)b[i];
        suma += dif * dif;
    }
    return suma;
}
