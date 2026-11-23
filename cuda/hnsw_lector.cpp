// hnsw_lector.cpp
// lee el modelo HNSW que guarda ParlayANN (solo CPU, nada de GPU aqui)

#include <algorithm>

#include "hnsw.h"

// el archivo se lee entero a memoria y despues se recorre con un cursor
// (leer dato por dato desde \\wsl.localhost seria muy lento)
struct Cursor {
    const std::vector<char>& buf;
    size_t pos = 0;
    std::string ruta;

    template <typename T>
    T leer() {
        if (pos + sizeof(T) > buf.size()) {
            fprintf(stderr, "%s: el archivo se termino antes de tiempo\n", ruta.c_str());
            exit(1);
        }
        T x;
        memcpy(&x, buf.data() + pos, sizeof(T));
        pos += sizeof(T);
        return x;
    }
};

HnswHost leerHnsw(const std::string& ruta) {
    FILE* f = fopen(ruta.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "No se pudo abrir: %s\n", ruta.c_str());
        exit(1);
    }
    std::vector<char> buf;
    {
        char tmp[1 << 16];
        size_t leidos;
        while ((leidos = fread(tmp, 1, sizeof(tmp), f)) > 0) buf.insert(buf.end(), tmp, tmp + leidos);
    }
    fclose(f);

    Cursor c{buf, 0, ruta};
    HnswHost h;

    // ---- cabecera ----
    char magia[4];
    for (int i = 0; i < 4; i++) magia[i] = c.leer<char>();
    if (memcmp(magia, "HNSW", 4) != 0) {
        fprintf(stderr, "%s: no es un modelo HNSW\n", ruta.c_str());
        exit(1);
    }
    uint32_t version = c.leer<uint32_t>();
    if (version != 3) {
        fprintf(stderr, "%s: version %u, se esperaba 3\n", ruta.c_str(), version);
        exit(1);
    }
    c.leer<uint64_t>();   // code_U (no se usa)
    c.leer<uint64_t>();   // size_node (no se usa)
    h.dim = c.leer<uint32_t>();
    h.ml = c.leer<float>();
    h.m = c.leer<uint32_t>();
    h.efc = c.leer<uint32_t>();
    h.alpha = c.leer<float>();
    h.n = c.leer<uint32_t>();

    // ---- nivel e id del punto de cada nodo ----
    h.nivel.resize(h.n);
    h.datoDe.resize(h.n);
    for (uint32_t i = 0; i < h.n; i++) {
        h.nivel[i] = c.leer<uint32_t>();
        h.datoDe[i] = c.leer<uint32_t>();
    }

    // ---- listas de vecinos: para cada nodo, niveles 0..nivel ----
    h.primeraLista.resize(h.n);
    for (uint32_t i = 0; i < h.n; i++) {
        h.primeraLista[i] = h.iniLista.size();
        for (uint32_t l = 0; l <= h.nivel[i]; l++) {
            uint64_t cuantos = c.leer<uint64_t>();
            h.iniLista.push_back(h.ids.size());
            h.tamLista.push_back((uint32_t)cuantos);
            for (uint64_t j = 0; j < cuantos; j++) {
                uint32_t v = c.leer<uint32_t>();
                if (v >= h.n) {
                    fprintf(stderr, "%s: vecino %u fuera de rango (n=%u)\n", ruta.c_str(), v, h.n);
                    exit(1);
                }
                h.ids.push_back(v);
            }
            if (l == 0) h.grado0Max = std::max(h.grado0Max, (uint32_t)cuantos);
            else h.gradoArribaMax = std::max(h.gradoArribaMax, (uint32_t)cuantos);
        }
    }

    // ---- puntos de entrada ----
    uint64_t numEntradas = c.leer<uint64_t>();
    for (uint64_t i = 0; i < numEntradas; i++) h.entradas.push_back(c.leer<uint32_t>());

    // EL ARCHIVO TIENE QUE TERMINAR JUSTO AQUI, si no el formato no es el que creemos
    if (c.pos != buf.size()) {
        fprintf(stderr, "%s: sobran %zu bytes al final, formato inesperado\n", ruta.c_str(),
                buf.size() - c.pos);
        exit(1);
    }
    if (h.entradas.empty() || h.entradas.size() > (size_t)MAX_ENTRADAS_HNSW) {
        fprintf(stderr, "%s: %zu puntos de entrada (se soportan 1..%d)\n", ruta.c_str(),
                h.entradas.size(), MAX_ENTRADAS_HNSW);
        exit(1);
    }
    return h;
}
