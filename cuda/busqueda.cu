// busqueda.cu
// kernel de beam search: UN WARP (32 hilos) RESUELVE UNA CONSULTA
//
// la idea es copiar paso a paso beam_search() de ParlayANN para que la GPU
// devuelva EXACTAMENTE los mismos vecinos que la CPU. Por eso hay cosas que
// parecen raras pero estan a proposito (marcadas con "IGUAL QUE PARLAYANN").
//
// estado de cada consulta:
//   frontera : los mejores Ls puntos encontrados, ordenados por (distancia, id)
//   visitados: cuantas veces se visito cada id (tabla hash exacta)
//   filtro   : filtro hash "ya lo vi" con perdidas, igual al de ParlayANN
//
// en cada vuelta:
//   1. tomar el primer punto de la frontera sin visitar y marcarlo
//   2. leer sus vecinos, descartar los ya vistos (filtro) y calcular distancias
//   3. ordenar los candidatos y mezclarlos con la frontera (set_union)
//   4. recortar a Ls y aplicar el cut (1.35 * distancia del k-esimo)
//   5. buscar el siguiente sin visitar; si no hay, se termina
//
// VERSION 2 (optimizada). Cambios respecto a la version 1, medidos con el perfil:
//   - distancias en grupos de 8 hilos por vector (lectura de 128 bytes de una vez)
//     y 4 distancias por hilo en vuelo -> antes era 1 hilo por distancia
//   - la fila de vecinos y las casillas del filtro se leen TODAS de una vez
//     (antes: 2 viajes a memoria por cada trozo de 32 vecinos)
//   - menos memoria compartida por warp -> mas warps por SM

#include <algorithm>
#include <climits>
#include <cmath>

#include "busqueda.cuh"
#include "distancia.cuh"

#define TODOS 0xffffffffu
#define VACIO 0xffffffffu   // igual a SIN_VECINO (relleno de la matriz del grafo)

// grado maximo soportado = MAX_TROZOS * 32 (HCNNG tiene 150)
#define MAX_TROZOS 8
// distancias que lleva cada hilo por ronda (x4 grupos = 16 por warp)
#define DIST_POR_HILO 4

// ---- PERFILADO (solo si se compila con -DPERFIL) ----------------------------
// cada warp mide con clock64() cuantos ciclos pasa en cada seccion del kernel
// y al final el hilo 0 los suma a g_perfil. Sin -DPERFIL esto no genera codigo.
#ifdef PERFIL
__device__ unsigned long long g_perfil[NUM_SECCIONES_PERFIL];
#define PERFIL_INICIO(t) long long t = clock64()
#define PERFIL_SUMAR(sec, t)               \
    do {                                   \
        long long ahora = clock64();       \
        ciclos[sec] += ahora - t;          \
        t = ahora;                         \
    } while (0)
#define PERFIL_CONTAR(sec) ciclos[sec]++
#else
#define PERFIL_INICIO(t)
#define PERFIL_SUMAR(sec, t)
#define PERFIL_CONTAR(sec)
#endif

// hash64_2 de parlaylib (utilities.h), copiado tal cual
// IGUAL QUE PARLAYANN: el filtro tiene que caer en las mismas posiciones
__device__ __forceinline__ uint64_t hash64_2(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x = x ^ (x >> 31);
    return x;
}

// orden de ParlayANN: primero distancia, si empatan gana el id menor
__device__ __forceinline__ bool menor(int dA, uint32_t idA, int dB, uint32_t idB) {
    return dA < dB || (dA == dB && idA < idB);
}

// cuantos elementos de la lista ordenada son < (d, id)   (std::lower_bound)
__device__ int contarMenores(const int* dist, const uint32_t* ids, int n, int d, uint32_t id) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (menor(dist[mid], ids[mid], d, id)) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// cuantos elementos de la lista ordenada son <= (d, id)  (std::upper_bound)
__device__ int contarMenoresIguales(const int* dist, const uint32_t* ids, int n, int d,
                                    uint32_t id) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (!menor(d, id, dist[mid], ids[mid])) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int minimoWarp(int x) {
    for (int salto = 16; salto > 0; salto >>= 1) x = min(x, __shfl_xor_sync(TODOS, x, salto));
    return x;
}

// ---- tabla de visitados (exacta, direccionamiento abierto) ------------------
__device__ __forceinline__ uint32_t posVis(uint32_t id, uint32_t mascara) {
    return (uint32_t)(hash64_2(id) >> 32) & mascara;
}

// solo la llama el hilo 0
__device__ void sumarVisita(uint32_t* claves, int* cuentas, uint32_t mascara, uint32_t id) {
    uint32_t h = posVis(id, mascara);
    while (true) {
        uint32_t c = claves[h];
        if (c == id) { cuentas[h]++; return; }
        if (c == VACIO) { claves[h] = id; cuentas[h] = 1; return; }
        h = (h + 1) & mascara;
    }
}

__device__ int vecesVisitado(const uint32_t* claves, const int* cuentas, uint32_t mascara,
                             uint32_t id) {
    uint32_t h = posVis(id, mascara);
    while (true) {
        uint32_t c = claves[h];
        if (c == id) return cuentas[h];
        if (c == VACIO) return 0;
        h = (h + 1) & mascara;
    }
}

// la fila de vecinos se procesa en trozos de 32 -> los arreglos de candidatos
// se redondean a multiplo de 32
__host__ __device__ inline int redondear32(int x) { return (x + 31) & ~31; }

// memoria compartida que usa cada warp, redondeada a 16 bytes
__host__ __device__ inline size_t bytesPorWarp(int Ls, int G) {
    int Gp = redondear32(G);
    size_t b = 128;              // consulta (32 palabras)
    b += 4ull * Ls * 2;          // frontera: ids + distancias
    b += 4ull * Gp * 6;          // cId, cDist, sId, sDist, sRep, kPos (ver abajo, se reusan)
    b += Gp;                     // kVis
    b += Ls;                     // fVis
    return (b + 15) & ~(size_t)15;
}

__global__ void kernelBusqueda(const int* __restrict__ palabrasBase,
                               const int* __restrict__ normasBase,
                               const int* __restrict__ palabrasQ, const int* __restrict__ normasQ,
                               uint32_t numConsultas, const uint32_t* __restrict__ vecinos, int G,
                               ConfigBusqueda cfg, int bitsFiltro, uint32_t capVis,
                               uint32_t* filtroGlobal, uint32_t* visClavesGlobal,
                               int* visCuentasGlobal, uint32_t* salidaIds, int* salidaDists,
                               int* salidaComp, int* salidaBanderas) {
    extern __shared__ __align__(16) unsigned char compartida[];

    const int lane = threadIdx.x & 31;
    const unsigned menoresQueYo = (1u << lane) - 1;
    const int warpEnBloque = threadIdx.x >> 5;
    const int warpGlobal = blockIdx.x * (blockDim.x >> 5) + warpEnBloque;
    const int totalWarps = gridDim.x * (blockDim.x >> 5);
    const int Ls = cfg.Ls;
    const int Gp = redondear32(G);
    const int numTrozos = Gp / 32;

    // ---- repartir la memoria compartida de ESTE warp ----
    // OJO: varios arreglos se REUSAN en distintas partes de la vuelta
    //   2a filtro     : cId = casilla de cada vecino, cDist = id de cada vecino
    //   2b pendientes : sId / sDist = vecinos a calcular y su distancia
    //   2c candidatos : cId / cDist
    //   3a ordenados  : sId / sDist / sRep
    //   3b insertar   : kId(=cId) / kDist(=cDist) / kPos / kVis
    unsigned char* ptr = compartida + warpEnBloque * bytesPorWarp(Ls, G);
    int4* q4 = (int4*)ptr;             ptr += 128;
    uint32_t* fId = (uint32_t*)ptr;    ptr += 4 * Ls;
    int* fDist = (int*)ptr;            ptr += 4 * Ls;
    uint32_t* cId = (uint32_t*)ptr;    ptr += 4 * Gp;
    int* cDist = (int*)ptr;            ptr += 4 * Gp;
    uint32_t* sId = (uint32_t*)ptr;    ptr += 4 * Gp;
    int* sDist = (int*)ptr;            ptr += 4 * Gp;
    int* sRep = (int*)ptr;             ptr += 4 * Gp;   // n de copia dentro de su grupo de iguales
    int* kPos = (int*)ptr;             ptr += 4 * Gp;   // posicion final en la frontera
    unsigned char* kVis = ptr;         ptr += Gp;
    unsigned char* fVis = ptr;
    uint32_t* kId = cId;
    int* kDist = cDist;

    // ---- memoria global auxiliar de este warp ----
    const uint32_t tamFiltro = 1u << bitsFiltro;
    const uint32_t mascaraFiltro = tamFiltro - 1;
    uint32_t* filtro = filtroGlobal + (size_t)warpGlobal * tamFiltro;
    uint32_t* visClaves = visClavesGlobal + (size_t)warpGlobal * capVis;
    int* visCuentas = visCuentasGlobal + (size_t)warpGlobal * capVis;
    const uint32_t mascaraVis = capVis - 1;

    const int grupo = lane >> 3;   // 0..3: grupo de 8 hilos (para las distancias)
    const int sub = lane & 7;      // 0..7: que int4 del vector lee este hilo

#ifdef PERFIL
    long long ciclos[NUM_SECCIONES_PERFIL] = {0};
#endif

    // cada warp va tomando consultas: warpGlobal, warpGlobal + totalWarps, ...
    for (uint32_t q = warpGlobal; q < numConsultas; q += totalWarps) {
        PERFIL_INICIO(tp);
        PERFIL_CONTAR(SEC_CONSULTAS);
        ((int*)q4)[lane] = palabrasQ[(size_t)q * PALABRAS_POR_VEC + lane];
        const int normaQ = normasQ[q];

        // LIMPIAR filtro y visitados (en la CPU se crean nuevos en cada consulta)
        for (uint32_t i = lane; i < tamFiltro; i += 32) filtro[i] = VACIO;
        for (uint32_t i = lane; i < capVis; i += 32) visClaves[i] = VACIO;
        __syncwarp();
        // pedazo de la consulta que le toca a este hilo en los grupos de 8 (ver 2b)
        const int4 qMio = q4[sub];

        // frontera inicial = {punto 0}
        // IGUAL QUE PARLAYANN: el wrapper siempre empieza en el punto 0
        if (lane == 0) {
            fId[0] = 0;
            fDist[0] = distanciaHilo(q4, normaQ, palabrasBase, normasBase, 0);
            fVis[0] = 0;
        }
        __syncwarp();

        int fSize = 1;
        int u = 0;              // posicion del primer punto sin visitar (-1 = no hay)
        long numVisitados = 0;
        int comp = 1;           // distancias calculadas (la del punto inicial cuenta)
        int bandera = 0;
        PERFIL_SUMAR(SEC_INICIO, tp);

        while (u >= 0 && numVisitados < cfg.limite) {
            PERFIL_CONTAR(SEC_ITERACIONES);
            const uint32_t actualId = fId[u];
            const int actualDist = fDist[u];

            // ===== 2a (parte 1). PEDIR LA FILA DE VECINOS COMPLETA =====
            // se piden todas las lecturas antes de usarlas, asi viajan juntas
            // la matriz esta rellenada con VACIO, no hace falta leer el grado
            const uint32_t* fila = vecinos + (size_t)actualId * G;
            uint32_t aT[MAX_TROZOS];
#pragma unroll
            for (int t = 0; t < MAX_TROZOS; t++) {
                int j = t * 32 + lane;
                aT[t] = (t < numTrozos && j < G) ? fila[j] : VACIO;
            }

            // ===== 1. VISITAR EL ACTUAL (mientras llega la fila) =====
            if (lane == 0) {
                fVis[u] = 1;
                sumarVisita(visClaves, visCuentas, mascaraVis, actualId);
            }
            __syncwarp();
            numVisitados++;
            if (numVisitados > (long)(capVis / 2)) {   // no deberia pasar nunca
                bandera |= BANDERA_DESBORDE;
                break;
            }
            PERFIL_SUMAR(SEC_VISITAR, tp);

            // si la frontera esta llena, solo sirven candidatos mas cerca que el ultimo
            const int cutoff = (fSize < Ls) ? INT_MAX : fDist[fSize - 1];

            // ===== 2a (parte 2). FILTRO HASH EXACTO =====
            // IGUAL QUE PARLAYANN: "if (a == p.id() || has_been_seen(a)) continue;"
            // p.id() es el INDICE DE LA CONSULTA (no un punto de la base), asi que
            // el punto de la base con ese mismo numero nunca se considera
            //
            // en la CPU los vecinos se revisan EN ORDEN y cada uno puede pisar la
            // casilla de uno anterior. Para que en paralelo salga lo mismo:
            //  - primero se leen de memoria TODAS las casillas (un solo viaje)
            //  - lo que "habia" en mi casilla justo antes de mi es el id del ULTIMO
            //    vecino anterior que cayo en la misma casilla:
            //      * en mi trozo de 32 -> __match_any_sync
            //      * en trozos anteriores -> se busca hacia atras en cId/cDist
            //      * si nadie antes cayo ahi -> el valor leido de memoria
            uint32_t locT[MAX_TROZOS], memT[MAX_TROZOS];
#pragma unroll
            for (int t = 0; t < MAX_TROZOS; t++) {
                bool participa = (aT[t] != VACIO) && (aT[t] != q);
                locT[t] = participa ? (uint32_t)(hash64_2(aT[t]) & mascaraFiltro)
                                    : (0x80000000u | (uint32_t)(t * 32 + lane));  // nunca choca
                memT[t] = participa ? filtro[locT[t]] : VACIO;
            }

            int nPend = 0;   // vecinos a los que hay que calcular distancia
#pragma unroll
            for (int t = 0; t < MAX_TROZOS; t++) {
                if (t < numTrozos) {
                    uint32_t a = aT[t], loc = locT[t];
                    bool participa = (a != VACIO) && (a != q);
                    unsigned mismoLoc = __match_any_sync(TODOS, loc);
                    unsigned antes = mismoLoc & menoresQueYo;
                    int fuente = antes ? (31 - __clz(antes)) : lane;
                    uint32_t idAnterior = __shfl_sync(TODOS, a, fuente);
                    uint32_t enCasilla = memT[t];
                    if (antes) {
                        enCasilla = idAnterior;
                    } else if (participa) {
                        for (int p = t * 32 - 1; p >= 0; p--)
                            if (cId[p] == loc) {
                                enCasilla = (uint32_t)cDist[p];
                                break;
                            }
                    }
                    bool yaVisto = participa && (enCasilla == a);

                    // anotar casilla e id de este trozo para los trozos siguientes
                    cId[t * 32 + lane] = loc;
                    cDist[t * 32 + lane] = (int)a;
                    // a memoria escribe solo el ULTIMO de cada casilla en este trozo
                    unsigned despues = mismoLoc & ~((2u << lane) - 1);
                    if (participa && despues == 0) filtro[loc] = a;

                    // los que pasan el filtro se anotan EN ORDEN en la lista pendiente
                    bool calcular = participa && !yaVisto;
                    unsigned mCalc = __ballot_sync(TODOS, calcular);
                    if (calcular) sId[nPend + __popc(mCalc & menoresQueYo)] = a;
                    nPend += __popc(mCalc);
                    __syncwarp();
                }
            }
            comp += nPend;
            PERFIL_SUMAR(SEC_FILTRO, tp);

            // ===== 2b. DISTANCIAS EN GRUPOS DE 8 HILOS =====
            // CAMBIO CLAVE DE LA OPTIMIZACION:
            // 8 hilos leen JUNTOS un vector de 128 bytes (16 bytes cada uno) en una sola
            // lectura. El warp hace 4 distancias a la vez y cada hilo lleva 4, asi hay
            // 16 vectores pedidos a memoria al mismo tiempo.
            // (ver micro_distancias.cu: compara 1, 4, 8 y 32 hilos por distancia)
            for (int base = 0; base < nPend; base += 4 * DIST_POR_HILO) {
                int4 v[DIST_POR_HILO];
                int normaP[DIST_POR_HILO];
                // primero se piden todas las lecturas...
#pragma unroll
                for (int r = 0; r < DIST_POR_HILO; r++) {
                    int i = base + r * 4 + grupo;
                    v[r] = make_int4(0, 0, 0, 0);
                    normaP[r] = 0;
                    if (i < nPend) {
                        uint32_t p = sId[i];
                        v[r] = __ldg(reinterpret_cast<const int4*>(palabrasBase +
                                                                   (size_t)p * PALABRAS_POR_VEC) + sub);
                        if (sub == 0) normaP[r] = __ldg(normasBase + p);
                    }
                }
                // ...y despues se usan
                int prod[DIST_POR_HILO];
#pragma unroll
                for (int r = 0; r < DIST_POR_HILO; r++) {
                    prod[r] = __dp4a(v[r].x, qMio.x, 0);
                    prod[r] = __dp4a(v[r].y, qMio.y, prod[r]);
                    prod[r] = __dp4a(v[r].z, qMio.z, prod[r]);
                    prod[r] = __dp4a(v[r].w, qMio.w, prod[r]);
                }
                // suma dentro de cada grupo de 8 hilos
#pragma unroll
                for (int salto = 4; salto > 0; salto >>= 1)
#pragma unroll
                    for (int r = 0; r < DIST_POR_HILO; r++)
                        prod[r] += __shfl_xor_sync(TODOS, prod[r], salto, 8);
                if (sub == 0) {
#pragma unroll
                    for (int r = 0; r < DIST_POR_HILO; r++) {
                        int i = base + r * 4 + grupo;
                        if (i < nPend) sDist[i] = normaQ + normaP[r] - 2 * prod[r];
                    }
                }
            }
            __syncwarp();
            PERFIL_SUMAR(SEC_DISTANCIAS, tp);

            // ===== 2c. CUTOFF: pasan a candidatos en el mismo orden de llegada =====
            int nc = 0;
            for (int base = 0; base < nPend; base += 32) {
                int i = base + lane;
                bool esCand = (i < nPend) && sDist[i] < cutoff;
                unsigned mCand = __ballot_sync(TODOS, esCand);
                if (esCand) {
                    int pos = nc + __popc(mCand & menoresQueYo);
                    cId[pos] = sId[i];
                    cDist[pos] = sDist[i];
                }
                nc += __popc(mCand);
            }
            __syncwarp();
            PERFIL_SUMAR(SEC_FILTRO, tp);

            // ===== 3a. ORDENAR CANDIDATOS =====
            // ordenamiento por conteo de rangos: la posicion de cada candidato es
            // cuantos son menores que el. Si hay copias iguales (vecino repetido en
            // el grafo) se desempata por orden de llegada -> "rep" = n de copia
            for (int i = lane; i < nc; i += 32) {
                int di = cDist[i];
                uint32_t idi = cId[i];
                int rango = 0, rep = 0;
                for (int j = 0; j < nc; j++) {
                    int dj = cDist[j];
                    uint32_t idj = cId[j];
                    if (menor(dj, idj, di, idi)) rango++;
                    else if (j < i && dj == di && idj == idi) { rango++; rep++; }
                }
                sId[rango] = idi;
                sDist[rango] = di;
                sRep[rango] = rep;
            }
            __syncwarp();
            PERFIL_SUMAR(SEC_ORDENAR, tp);

            // ===== 3b. SET_UNION CON LA FRONTERA (MULTICONJUNTO) =====
            // IGUAL QUE PARLAYANN (std::set_union): si un id ya esta en la frontera
            // cF veces, de los candidatos con ese id solo entran las copias extra
            // (rep >= cF). Casi siempre cF = 0 o 1 y rep = 0, pero HCNNG y PyNNDescent
            // tienen vecinos repetidos y hay que respetarlo.
            int nK = 0;
            for (int base = 0; base < nc; base += 32) {
                int r = base + lane;
                bool entra = false;
                int hastaIgual = 0;
                unsigned char vis = 0;
                uint32_t id = 0;
                int d = 0;
                if (r < nc) {
                    id = sId[r];
                    d = sDist[r];
                    int rep = sRep[r];
                    int menores = contarMenores(fDist, fId, fSize, d, id);
                    hastaIgual = contarMenoresIguales(fDist, fId, fSize, d, id);
                    int cF = hastaIgual - menores;
                    entra = rep >= cF;
                    // set_difference(frontera, visitados): la copia n "rep" esta
                    // visitada si el id se visito mas de "rep" veces
                    if (entra) vis = (rep < vecesVisitado(visClaves, visCuentas, mascaraVis, id));
                }
                unsigned m = __ballot_sync(TODOS, entra);
                if (entra) {
                    int jK = nK + __popc(m & menoresQueYo);
                    kId[jK] = id;
                    kDist[jK] = d;
                    // en la union las copias de la frontera van antes que las nuevas
                    kPos[jK] = hastaIgual + jK;
                    kVis[jK] = vis;
                }
                nK += __popc(m);
            }
            __syncwarp();
            PERFIL_SUMAR(SEC_UNION, tp);

            // ===== 3c. MEZCLA EN EL MISMO ARREGLO =====
            // los elementos de la frontera solo se mueven a la derecha
            // (nueva pos = pos + candidatos menores que el), por eso se recorre
            // DESDE EL FINAL: primero todos leen, despues todos escriben
            int uNuevo = u;
            if (nK > 0) {
                uNuevo = u + contarMenores(kDist, kId, nK, actualDist, actualId);
                int primeroQueSeMueve = kPos[0];
                for (int alto = fSize - 1; alto >= primeroQueSeMueve; alto -= 32) {
                    int i = alto - lane;
                    bool activo = i >= primeroQueSeMueve;
                    uint32_t id = 0;
                    int d = 0, nuevaPos = 0;
                    unsigned char v = 0;
                    if (activo) {
                        id = fId[i];
                        d = fDist[i];
                        v = fVis[i];
                        nuevaPos = i + contarMenores(kDist, kId, nK, d, id);
                    }
                    __syncwarp();
                    if (activo && nuevaPos < Ls) {   // lo que pasa de Ls se pierde (recorte)
                        fId[nuevaPos] = id;
                        fDist[nuevaPos] = d;
                        fVis[nuevaPos] = v;
                    }
                    __syncwarp();
                }
                for (int j = lane; j < nK; j += 32) {
                    int p = kPos[j];
                    if (p < Ls) {
                        fId[p] = kId[j];
                        fDist[p] = kDist[j];
                        fVis[p] = kVis[j];
                    }
                }
                __syncwarp();
            }
            fSize = min(Ls, fSize + nK);
            PERFIL_SUMAR(SEC_MEZCLA, tp);

            // ===== 4. CUT =====
            // IGUAL QUE PARLAYANN:
            //   upper_bound(frontera, pair{0, cut * frontera[k].second}, less)
            // el umbral se calcula en double y se guarda en float (asi lo hace la CPU)
            if (fSize > cfg.k) {
                int nuevoTam = fSize;
                if (lane == 0) {
                    float umbral = (float)(cfg.cut * (double)fDist[cfg.k]);
                    int lo = 0, hi = fSize;
                    while (lo < hi) {
                        int mid = (lo + hi) >> 1;
                        float dm = (float)fDist[mid];
                        bool vaDespues = (umbral < dm) || (umbral == dm && 0u < fId[mid]);
                        if (!vaDespues) lo = mid + 1;
                        else hi = mid;
                    }
                    nuevoTam = lo;
                }
                fSize = __shfl_sync(TODOS, nuevoTam, 0);
            }

            // ===== 5. SIGUIENTE SIN VISITAR =====
            // antes de uNuevo todo esta visitado salvo, quizas, candidatos nuevos
            int minK = INT_MAX;
            for (int j = lane; j < nK; j += 32)
                if (!kVis[j] && kPos[j] < fSize) minK = min(minK, kPos[j]);
            int desde = min(uNuevo, minimoWarp(minK));

            u = -1;
            for (int base = desde; base < fSize; base += 32) {
                int i = base + lane;
                bool sinVisitar = (i < fSize) && !fVis[i];
                unsigned m = __ballot_sync(TODOS, sinVisitar);
                if (m) {
                    u = base + __ffs(m) - 1;
                    break;
                }
            }
            __syncwarp();
            PERFIL_SUMAR(SEC_CUT_SIGUIENTE, tp);
        }

        // ===== SALIDA: los k primeros de la frontera =====
        if (fSize < cfg.k) bandera |= BANDERA_FRONTERA_CORTA;
        for (int j = lane; j < cfg.k; j += 32) {
            salidaIds[(size_t)q * cfg.k + j] = (j < fSize) ? fId[j] : VACIO;
            salidaDists[(size_t)q * cfg.k + j] = (j < fSize) ? fDist[j] : INT_MAX;
        }
        if (lane == 0) {
            salidaComp[q] = comp;
            salidaBanderas[q] = bandera;
        }
        __syncwarp();
        PERFIL_SUMAR(SEC_INICIO, tp);
    }

#ifdef PERFIL
    if (lane == 0)
        for (int s = 0; s < NUM_SECCIONES_PERFIL; s++)
            atomicAdd(&g_perfil[s], (unsigned long long)ciclos[s]);
#endif
}

#ifdef PERFIL
void reiniciarPerfil() {
    unsigned long long ceros[NUM_SECCIONES_PERFIL] = {0};
    CHECK_CUDA(cudaMemcpyToSymbol(g_perfil, ceros, sizeof(ceros)));
}
void leerPerfil(unsigned long long* salida) {
    CHECK_CUDA(cudaMemcpyFromSymbol(salida, g_perfil,
                                    sizeof(unsigned long long) * NUM_SECCIONES_PERFIL));
}
#endif

static uint32_t siguientePotencia2(uint32_t x) {
    uint32_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

void BuscadorGPU::preparar(const GrafoGPU& grafo, uint32_t nq, const ConfigBusqueda& c) {
    liberar();
    cfg = c;
    numConsultas = nq;

    if (grafo.gradoMax > MAX_TROZOS * 32) {
        fprintf(stderr, "gradoMax %u mayor que el maximo soportado %d\n", grafo.gradoMax,
                MAX_TROZOS * 32);
        exit(1);
    }

    // IGUAL QUE PARLAYANN: tamano del filtro
    //   bits = max(10, ceil(log2(Ls*Ls)) - 2)
    bitsFiltro = std::max(10, (int)std::ceil(std::log2((double)cfg.Ls * cfg.Ls)) - 2);
    // visitados: cabe de sobra (se avisa con BANDERA_DESBORDE si se pasa de la mitad)
    capVis = siguientePotencia2((uint32_t)(8 * cfg.Ls + 128));

    // warps por bloque segun la memoria compartida de cada uno (max 48 KB por bloque)
    size_t porWarp = bytesPorWarp(cfg.Ls, (int)grafo.gradoMax);
    warpsPorBloque = (int)std::min<size_t>(16, std::max<size_t>(1, (48 * 1024) / porWarp));
    // LIMITE POR REGISTROS: si cada hilo usa muchos registros, el bloque no puede
    // tener tantos hilos (si no, el lanzamiento falla con "too many resources")
    cudaFuncAttributes atrib;
    CHECK_CUDA(cudaFuncGetAttributes(&atrib, kernelBusqueda));
    warpsPorBloque = std::min(warpsPorBloque, atrib.maxThreadsPerBlock / 32);
    bytesCompartida = porWarp * warpsPorBloque;

    int bloquesPorSM = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bloquesPorSM, kernelBusqueda,
                                                             warpsPorBloque * 32, bytesCompartida));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    bloques = std::max(1, bloquesPorSM) * prop.multiProcessorCount;
    size_t totalWarps = (size_t)bloques * warpsPorBloque;

    CHECK_CUDA(cudaMalloc(&filtro, totalWarps * ((size_t)1 << bitsFiltro) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&visClaves, totalWarps * capVis * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&visCuentas, totalWarps * capVis * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&idsDev, (size_t)nq * cfg.k * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&distsDev, (size_t)nq * cfg.k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&compDev, (size_t)nq * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&banderasDev, (size_t)nq * sizeof(int)));
}

void BuscadorGPU::lanzar(const VectoresGPU& base, const VectoresGPU& consultas,
                         const GrafoGPU& grafo, cudaStream_t stream) {
    kernelBusqueda<<<bloques, warpsPorBloque * 32, bytesCompartida, stream>>>(
        base.palabras, base.normas, consultas.palabras, consultas.normas, numConsultas,
        grafo.vecinos, (int)grafo.gradoMax, cfg, bitsFiltro, capVis, filtro, visClaves,
        visCuentas, idsDev, distsDev, compDev, banderasDev);
    CHECK_CUDA(cudaGetLastError());
}

void BuscadorGPU::liberar() {
    cudaFree(filtro);
    cudaFree(visClaves);
    cudaFree(visCuentas);
    cudaFree(idsDev);
    cudaFree(distsDev);
    cudaFree(compDev);
    cudaFree(banderasDev);
    filtro = nullptr;
    visClaves = nullptr;
    visCuentas = nullptr;
    idsDev = nullptr;
    distsDev = nullptr;
    compDev = nullptr;
    banderasDev = nullptr;
}
