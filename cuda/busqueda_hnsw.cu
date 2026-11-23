// busqueda_hnsw.cu
// busqueda HNSW en la GPU: UN WARP (32 hilos) RESUELVE UNA CONSULTA
//
// traduccion de HNSW::search() + beamSearch() de ParlayANN
// (algorithms/HNSW/HNSW.hpp, commit 45587ba), para devolver EXACTAMENTE lo
// mismo que la CPU. Lo que parece raro esta a proposito ("IGUAL QUE HNSW").
//
// 1. bajar por los niveles de arriba con ef = 1 (sin limite de visitas):
//    el resultado de cada nivel es el punto de entrada del siguiente
// 2. en el nivel 0, beam search con ef = Ls (y el limite "visit" si hay)
// 3. devolver los k mas cercanos del workset
//
// estructuras de cada busqueda por nivel (en la CPU se crean nuevas cada vez):
//   cand      : std::set ordenado SOLO POR DISTANCIA -> si ya hay un candidato
//               con la misma distancia, el nuevo NO entra
//   workset   : los ef mejores (en la CPU es un heap; aqui un arreglo ordenado)
//   visitados : filtro hash con perdidas (bits = ceil(log2(ef))*2 - 2)
//   is_inw    : ids que ya entraron alguna vez al workset (exacto)
//
// diferencia con el kernel de los otros grafos: aceptar un vecino CAMBIA el
// umbral para el siguiente (d < peor del workset), asi que esa decision se
// recorre EN ORDEN. Todo lo demas (leer, filtrar, distancias, correr arreglos)
// lo hacen los 32 hilos en paralelo.
//
// EPOCAS: en vez de limpiar las tablas globales en cada busqueda, cada casilla
// guarda (epoca << 32 | id). Una casilla de otra epoca cuenta como vacia.

#include <algorithm>
#include <climits>
#include <cmath>

#include "distancia.cuh"
#include "hnsw.h"
#include "utiles_gpu.cuh"

#define MAX_TROZOS 8        // grado maximo soportado = 256
#define DIST_POR_HILO 4     // distancias por hilo en cada ronda (16 por warp)

// ---- is_inw: conjunto exacto con epoca (direccionamiento abierto) ----------
__device__ __forceinline__ uint32_t posInw(uint32_t id, uint32_t mascara) {
    return (uint32_t)(hash64_2(id) >> 32) & mascara;
}

__device__ bool estaEnInw(const unsigned long long* tabla, uint32_t mascara, uint32_t epoca,
                          uint32_t id) {
    uint32_t h = posInw(id, mascara);
    unsigned long long buscado = marca(epoca, id);
    while (true) {
        unsigned long long c = __ldcg(tabla + h);   // sin cache L1 (la escriben atomicos)
        if (c == buscado) return true;
        if ((uint32_t)(c >> 32) != epoca) return false;   // casilla vacia en esta epoca
        h = (h + 1) & mascara;
    }
}

// varios hilos pueden meter a la vez -> atomicCAS
__device__ void meterEnInw(unsigned long long* tabla, uint32_t mascara, uint32_t epoca,
                           uint32_t id) {
    uint32_t h = posInw(id, mascara);
    unsigned long long nuevo = marca(epoca, id);
    while (true) {
        unsigned long long c = __ldcg(tabla + h);
        if (c == nuevo) return;
        if ((uint32_t)(c >> 32) != epoca) {
            unsigned long long viejo = atomicCAS(tabla + h, c, nuevo);
            if (viejo == c || viejo == nuevo) return;
            continue;   // otro hilo la ocupo justo antes: volver a mirar la misma casilla
        }
        h = (h + 1) & mascara;
    }
}

// ---- arreglos ordenados en memoria compartida, entre los 32 hilos ----------
// inserta (d, id) manteniendo el orden ascendente.
//   soloPorDist = true  -> orden solo por distancia y NO inserta si ya hay uno
//                          con la misma distancia (IGUAL QUE std::set de HNSW)
//   soloPorDist = false -> orden por (distancia, id)
// todos los hilos deben llamarla con los mismos valores
__device__ bool insertarOrdenado(int* D, uint32_t* I, int& tam, int d, uint32_t id,
                                 bool soloPorDist, int lane) {
    int lo = 0, hi = tam;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        bool va = soloPorDist ? (D[mid] < d) : menor(D[mid], I[mid], d, id);
        if (va) lo = mid + 1;
        else hi = mid;
    }
    int pos = lo;
    if (soloPorDist && pos < tam && D[pos] == d) return false;

    // correr [pos, tam) un lugar a la derecha, desde el final (todos leen, despues escriben)
    for (int alto = tam - 1; alto >= pos; alto -= 32) {
        int i = alto - lane;
        bool activo = i >= pos;
        int dd = 0;
        uint32_t ii = 0;
        if (activo) {
            dd = D[i];
            ii = I[i];
        }
        __syncwarp();
        if (activo) {
            D[i + 1] = dd;
            I[i + 1] = ii;
        }
        __syncwarp();
    }
    if (lane == 0) {
        D[pos] = d;
        I[pos] = id;
    }
    __syncwarp();
    tam++;
    return true;
}

// quita el primero (el de menor distancia) corriendo todo un lugar a la izquierda
__device__ void quitarPrimero(int* D, uint32_t* I, int& tam, int lane) {
    for (int base = 0; base < tam - 1; base += 32) {
        int i = base + lane;
        bool activo = i < tam - 1;
        int dd = 0;
        uint32_t ii = 0;
        if (activo) {
            dd = D[i + 1];
            ii = I[i + 1];
        }
        __syncwarp();
        if (activo) {
            D[i] = dd;
            I[i] = ii;
        }
        __syncwarp();
    }
    tam--;
}

// todo lo que necesita el kernel (se pasa por valor)
struct ParamsHnsw {
    const uint32_t* vecinos0;
    int G0;
    const uint32_t* vecinosArriba;
    int GA;
    const uint32_t* inicioArriba;
    const uint32_t* datoDe;
    const uint32_t* entradas;
    int numEntradas;
    int nivelSuperior;
    const int* palabras;   // vectores en orden de nodo
    const int* normas;
    const int* palabrasQ;
    const int* normasQ;
    uint32_t numConsultas;
    int Ls;
    uint32_t limite;       // limit_eval del nivel 0
    int k;
    int bitsVis0;
    uint32_t capInw;
    uint32_t epocaBase;    // sube en cada lanzamiento (las tablas no se limpian)
    unsigned long long* visitados;
    unsigned long long* inw;
    uint32_t* salidaIds;
    int* salidaDists;
    int* salidaComp;
    int* salidaBanderas;
};

__host__ __device__ inline int capacidadWorkset(int Ls) { return Ls + MAX_ENTRADAS_HNSW + 2; }

__host__ __device__ inline size_t bytesPorWarpHnsw(int Ls, int Gp) {
    size_t b = 128;                                  // consulta
    b += 4ull * capacidadWorkset(Ls) * 4;            // workset (d, id) + cand (d, id)
    b += 4ull * Gp * 5;                              // pId, pD, locSh, idSh, pPrev
    b += 4ull * MAX_ENTRADAS_HNSW;                   // puntos de entrada
    b += (size_t)Gp * 3;                             // banderas: inwAntes, memb, ins
    return (b + 15) & ~(size_t)15;
}

__global__ void kernelHnsw(ParamsHnsw P) {
    extern __shared__ __align__(16) unsigned char compartida[];

    const int lane = threadIdx.x & 31;
    const unsigned menoresQueYo = (1u << lane) - 1;
    const int warpEnBloque = threadIdx.x >> 5;
    const int warpGlobal = blockIdx.x * (blockDim.x >> 5) + warpEnBloque;
    const int totalWarps = gridDim.x * (blockDim.x >> 5);
    const int Gp = redondear32(max(P.G0, P.GA));
    const int capWs = capacidadWorkset(P.Ls);

    // ---- memoria compartida de ESTE warp ----
    unsigned char* ptr = compartida + warpEnBloque * bytesPorWarpHnsw(P.Ls, Gp);
    int4* q4 = (int4*)ptr;                 ptr += 128;
    int* wsD = (int*)ptr;                  ptr += 4 * capWs;   // workset ordenado
    uint32_t* wsI = (uint32_t*)ptr;        ptr += 4 * capWs;
    int* cD = (int*)ptr;                   ptr += 4 * capWs;   // cand ordenado por distancia
    uint32_t* cI = (uint32_t*)ptr;         ptr += 4 * capWs;
    uint32_t* pId = (uint32_t*)ptr;        ptr += 4 * Gp;      // vecinos que pasan el filtro
    int* pD = (int*)ptr;                   ptr += 4 * Gp;      // y su distancia
    uint32_t* locSh = (uint32_t*)ptr;      ptr += 4 * Gp;      // casilla de cada vecino (filtro)
    uint32_t* idSh = (uint32_t*)ptr;       ptr += 4 * Gp;      // id de cada vecino (filtro)
    int* pPrev = (int*)ptr;                ptr += 4 * Gp;      // copia anterior del mismo id
    uint32_t* epsSh = (uint32_t*)ptr;      ptr += 4 * MAX_ENTRADAS_HNSW;
    unsigned char* flagAntes = ptr;        ptr += Gp;          // ya estaba en is_inw
    unsigned char* flagMemb = ptr;         ptr += Gp;          // estaba en is_inw al llegarle el turno
    unsigned char* flagIns = ptr;                              // entro al workset

    unsigned long long* vis = P.visitados + (size_t)warpGlobal * ((size_t)1 << P.bitsVis0);
    unsigned long long* inw = P.inw + (size_t)warpGlobal * P.capInw;
    const uint32_t mascaraInw = P.capInw - 1;

    const int grupo = lane >> 3;
    const int sub = lane & 7;
    const uint32_t numCapas = (uint32_t)P.nivelSuperior + 1;

    for (uint32_t q = warpGlobal; q < P.numConsultas; q += totalWarps) {
        ((int*)q4)[lane] = P.palabrasQ[(size_t)q * PALABRAS_POR_VEC + lane];
        const int normaQ = P.normasQ[q];
        __syncwarp();
        const int4 qMio = q4[sub];

        int comp = 0, bandera = 0;
        int numEps = P.numEntradas;
        for (int e = lane; e < numEps; e += 32) epsSh[e] = P.entradas[e];
        __syncwarp();
        int wsSize = 0;

        // ===== NIVELES: de arriba hacia abajo =====
        for (int capa = P.nivelSuperior; capa >= 0; capa--) {
            // IGUAL QUE HNSW: arriba ef = 1 y sin limite; abajo ef = Ls y limit_eval
            const int ef = (capa > 0) ? 1 : P.Ls;
            const int bits = (capa > 0) ? 2 : P.bitsVis0;   // ef <= 2 -> 2 bits
            const uint32_t mascaraVis = (1u << bits) - 1;
            const uint32_t limite = (capa > 0) ? 0xffffffffu : P.limite;
            const uint32_t* matriz = (capa > 0) ? P.vecinosArriba : P.vecinos0;
            const int G = (capa > 0) ? P.GA : P.G0;
            const int numTrozos = redondear32(G) / 32;

            // TABLAS "NUEVAS" PARA ESTA BUSQUEDA (una por consulta y por nivel).
            // OJO: la epoca NUNCA se puede repetir mientras la tabla viva, ni siquiera
            // entre lanzamientos del kernel -> depende de la consulta, del nivel y de
            // epocaBase, que el host sube en cada lanzamiento. Nunca vale 0.
            const uint32_t epoca =
                P.epocaBase + q * numCapas + (uint32_t)(P.nivelSuperior - capa) + 1;
            int candSize = 0;
            wsSize = 0;
            uint32_t insertadosCapa = 0;

            // ---- puntos de entrada ----
            // IGUAL QUE HNSW: se marcan con hash64 (los vecinos usan hash64_2)
            for (int e = 0; e < numEps; e++) {
                uint32_t pe = epsSh[e];
                int dpe = 0;
                if (lane == 0) {
                    vis[hash64(pe) & mascaraVis] = marca(epoca, pe);
                    dpe = distanciaHilo(q4, normaQ, P.palabras, P.normas, pe);
                    meterEnInw(inw, mascaraInw, epoca, pe);
                }
                dpe = __shfl_sync(TODOS, dpe, 0);
                __syncwarp();
                insertarOrdenado(cD, cI, candSize, dpe, pe, true, lane);
                insertarOrdenado(wsD, wsI, wsSize, dpe, pe, false, lane);   // sin recortar
                comp++;
                insertadosCapa++;
            }

            uint32_t cntEval = 0;
            while (candSize > 0) {
                // IGUAL QUE HNSW: if(cand.begin()->d > workset[0].d*beta) break;  (beta = 1)
                // workset[0] del heap = el PEOR del workset = el ultimo del arreglo
                if (cD[0] > wsD[wsSize - 1]) break;
                if (++cntEval > limite) break;
                const uint32_t u = cI[0];
                quitarPrimero(cD, cI, candSize, lane);

                // ===== A. LEER LA FILA DE VECINOS COMPLETA =====
                const uint32_t* fila =
                    (capa > 0) ? matriz + ((size_t)P.inicioArriba[u] + (capa - 1)) * G
                               : matriz + (size_t)u * G;
                uint32_t aT[MAX_TROZOS];
#pragma unroll
                for (int t = 0; t < MAX_TROZOS; t++) {
                    int j = t * 32 + lane;
                    aT[t] = (t < numTrozos && j < G) ? fila[j] : VACIO;
                }

                // ===== B. FILTRO DE VISITADOS (con perdidas, exacto como la CPU) =====
                // IGUAL QUE HNSW:
                //   h = hash64_2(pv) & mask; if (visited[h] == pv) continue; visited[h] = pv;
                // mismo truco que en el otro kernel: se leen todas las casillas de una vez
                // y lo que "habia" justo antes de cada vecino se resuelve en paralelo
                uint32_t locT[MAX_TROZOS];
                unsigned long long memT[MAX_TROZOS];
#pragma unroll
                for (int t = 0; t < MAX_TROZOS; t++) {
                    bool hay = aT[t] != VACIO;
                    locT[t] = hay ? (uint32_t)(hash64_2(aT[t]) & mascaraVis)
                                  : (0x80000000u | (uint32_t)(t * 32 + lane));   // nunca choca
                    memT[t] = hay ? vis[locT[t]] : 0ull;
                }

                int nPend = 0;
#pragma unroll
                for (int t = 0; t < MAX_TROZOS; t++) {
                    if (t < numTrozos) {
                        uint32_t a = aT[t], loc = locT[t];
                        bool hay = a != VACIO;
                        unsigned mismoLoc = __match_any_sync(TODOS, loc);
                        unsigned antes = mismoLoc & menoresQueYo;
                        int fuente = antes ? (31 - __clz(antes)) : lane;
                        uint32_t idAnterior = __shfl_sync(TODOS, a, fuente);
                        bool yaVisto;
                        if (antes) {
                            yaVisto = hay && (idAnterior == a);
                        } else {
                            yaVisto = hay && (memT[t] == marca(epoca, a));
                            if (hay) {
                                for (int p = t * 32 - 1; p >= 0; p--)
                                    if (locSh[p] == loc) {
                                        yaVisto = (idSh[p] == a);
                                        break;
                                    }
                            }
                        }
                        locSh[t * 32 + lane] = loc;
                        idSh[t * 32 + lane] = a;
                        unsigned despues = mismoLoc & ~((2u << lane) - 1);
                        if (hay && despues == 0) vis[loc] = marca(epoca, a);

                        bool pasa = hay && !yaVisto;
                        unsigned mPasa = __ballot_sync(TODOS, pasa);
                        if (pasa) pId[nPend + __popc(mPasa & menoresQueYo)] = a;
                        nPend += __popc(mPasa);
                        __syncwarp();
                    }
                }
                comp += nPend;   // cnt_visited

                // ===== C. DISTANCIAS EN GRUPOS DE 8 HILOS =====
                for (int base = 0; base < nPend; base += 4 * DIST_POR_HILO) {
                    int4 v[DIST_POR_HILO];
                    int normaP[DIST_POR_HILO];
#pragma unroll
                    for (int r = 0; r < DIST_POR_HILO; r++) {
                        int i = base + r * 4 + grupo;
                        v[r] = make_int4(0, 0, 0, 0);
                        normaP[r] = 0;
                        if (i < nPend) {
                            uint32_t p = pId[i];
                            v[r] = __ldg(reinterpret_cast<const int4*>(P.palabras +
                                                                       (size_t)p * PALABRAS_POR_VEC) + sub);
                            if (sub == 0) normaP[r] = __ldg(P.normas + p);
                        }
                    }
                    int prod[DIST_POR_HILO];
#pragma unroll
                    for (int r = 0; r < DIST_POR_HILO; r++) {
                        prod[r] = __dp4a(v[r].x, qMio.x, 0);
                        prod[r] = __dp4a(v[r].y, qMio.y, prod[r]);
                        prod[r] = __dp4a(v[r].z, qMio.z, prod[r]);
                        prod[r] = __dp4a(v[r].w, qMio.w, prod[r]);
                    }
#pragma unroll
                    for (int salto = 4; salto > 0; salto >>= 1)
#pragma unroll
                        for (int r = 0; r < DIST_POR_HILO; r++)
                            prod[r] += __shfl_xor_sync(TODOS, prod[r], salto, 8);
                    if (sub == 0) {
#pragma unroll
                        for (int r = 0; r < DIST_POR_HILO; r++) {
                            int i = base + r * 4 + grupo;
                            if (i < nPend) pD[i] = normaQ + normaP[r] - 2 * prod[r];
                        }
                    }
                }
                __syncwarp();

                // ===== D. is_inw ANTES DE EMPEZAR + COPIAS REPETIDAS =====
                // si el mismo id aparece dos veces en la lista, la segunda depende de lo
                // que paso con la primera -> se guarda donde esta la copia anterior
                for (int j = lane; j < nPend; j += 32) {
                    uint32_t pv = pId[j];
                    int prev = -1;
                    for (int t = j - 1; t >= 0; t--)
                        if (pId[t] == pv) {
                            prev = t;
                            break;
                        }
                    pPrev[j] = prev;
                    flagAntes[j] = (prev < 0) ? estaEnInw(inw, mascaraInw, epoca, pv) : 0;
                }
                __syncwarp();

                // ===== E. DECIDIR EN ORDEN QUIEN ENTRA (SECUENCIAL) =====
                // IGUAL QUE HNSW, para cada vecino en orden:
                //   if (!(workset.size() < ef || d < workset[0].d)) continue;
                //   if (!is_inw.insert(pv).second) continue;
                //   cand.insert({d,pv}); workset.push(...); si workset > ef sacar el peor;
                //   si cand > ef sacar el peor
                // todos los hilos ven los mismos datos -> mismas decisiones
                for (int j = 0; j < nPend; j++) {
                    const int d = pD[j];
                    const uint32_t pv = pId[j];
                    const int prev = pPrev[j];
                    bool memb = (prev >= 0) ? (flagMemb[prev] || flagIns[prev]) : flagAntes[j];
                    bool cond = (wsSize < ef) || (d < wsD[wsSize - 1]);
                    bool entra = cond && !memb;
                    if (lane == 0) {
                        flagMemb[j] = memb;
                        flagIns[j] = entra;
                    }
                    if (entra) {
                        insertarOrdenado(cD, cI, candSize, d, pv, true, lane);
                        insertarOrdenado(wsD, wsI, wsSize, d, pv, false, lane);
                        if (wsSize > ef) wsSize--;
                        if (candSize > ef) candSize--;
                    }
                    __syncwarp();
                }

                // ===== F. ANOTAR EN is_inw LOS QUE ENTRARON =====
                for (int j = lane; j < nPend; j += 32)
                    if (flagIns[j]) meterEnInw(inw, mascaraInw, epoca, pId[j]);
                for (int base = 0; base < nPend; base += 32) {
                    int j = base + lane;
                    insertadosCapa += __popc(__ballot_sync(TODOS, j < nPend && flagIns[j]));
                }
                __syncwarp();
                if (insertadosCapa > P.capInw / 2) {   // no deberia pasar
                    bandera |= BANDERA_DESBORDE;
                    break;
                }
            }

            // IGUAL QUE HNSW: el punto de entrada del nivel siguiente es W[0] del heap,
            // o sea el de MAYOR distancia del workset (con ef = 1 hay uno solo)
            if (capa > 0) {
                if (lane == 0) epsSh[0] = wsI[wsSize - 1];
                numEps = 1;
                __syncwarp();
            }
        }

        // ===== SALIDA: los k primeros del workset del nivel 0 =====
        // la CPU ordena el workset por distancia y se queda con los k primeros;
        // aqui ya esta ordenado. Los ids se traducen de NODO a PUNTO de la base.
        if (wsSize < P.k) bandera |= BANDERA_FRONTERA_CORTA;
        for (int j = lane; j < P.k; j += 32) {
            P.salidaIds[(size_t)q * P.k + j] = (j < wsSize) ? P.datoDe[wsI[j]] : VACIO;
            P.salidaDists[(size_t)q * P.k + j] = (j < wsSize) ? wsD[j] : INT_MAX;
        }
        if (lane == 0) {
            P.salidaComp[q] = comp;
            P.salidaBanderas[q] = bandera;
        }
        __syncwarp();
    }
}

// ============================================================================
// subir el grafo a la GPU
// ============================================================================
void subirHnsw(const HnswHost& h, const VectoresHost& base, const VectoresGPU& baseGPU,
               HnswGPU& gpu) {
    gpu.n = h.n;
    gpu.G0 = (int)h.grado0Max;
    gpu.GA = (int)std::max<uint32_t>(1, h.gradoArribaMax);
    gpu.numEntradas = (int)h.entradas.size();
    gpu.nivelSuperior = (int)h.nivel[h.entradas[0]];
    if (gpu.G0 > MAX_TROZOS * 32 || gpu.GA > MAX_TROZOS * 32) {
        fprintf(stderr, "grado %d/%d mayor que el maximo soportado %d\n", gpu.G0, gpu.GA,
                MAX_TROZOS * 32);
        exit(1);
    }

    // nivel 0: matriz n x G0
    std::vector<uint32_t> tmp((size_t)h.n * gpu.G0, SIN_VECINO);
    for (uint32_t i = 0; i < h.n; i++)
        memcpy(tmp.data() + (size_t)i * gpu.G0, h.vecinos(i, 0), h.grado(i, 0) * sizeof(uint32_t));
    CHECK_CUDA(cudaMalloc(&gpu.vecinos0, tmp.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(gpu.vecinos0, tmp.data(), tmp.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    gpu.bytes = tmp.size() * sizeof(uint32_t);

    // niveles de arriba: un bloque de GA por cada (nodo, nivel >= 1)
    std::vector<uint32_t> inicio(h.n, SIN_VECINO);
    uint64_t bloques = 0;
    for (uint32_t i = 0; i < h.n; i++)
        if (h.nivel[i] >= 1) {
            inicio[i] = (uint32_t)bloques;
            bloques += h.nivel[i];
        }
    std::vector<uint32_t> arriba(std::max<uint64_t>(1, bloques) * gpu.GA, SIN_VECINO);
    for (uint32_t i = 0; i < h.n; i++)
        for (uint32_t l = 1; l <= h.nivel[i]; l++)
            memcpy(arriba.data() + ((size_t)inicio[i] + (l - 1)) * gpu.GA, h.vecinos(i, l),
                   h.grado(i, l) * sizeof(uint32_t));
    CHECK_CUDA(cudaMalloc(&gpu.vecinosArriba, arriba.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(gpu.vecinosArriba, arriba.data(), arriba.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&gpu.inicioArriba, h.n * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(gpu.inicioArriba, inicio.data(), h.n * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&gpu.datoDe, h.n * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(gpu.datoDe, h.datoDe.data(), h.n * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&gpu.entradas, h.entradas.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(gpu.entradas, h.entradas.data(), h.entradas.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    gpu.bytes += arriba.size() * 4 + 2ull * h.n * 4;

    // VECTORES EN ORDEN DE NODO: el nodo i guarda el punto datoDe[i]
    // si es la identidad se usan los de la base tal cual (no se copia nada)
    bool identidad = true;
    for (uint32_t i = 0; i < h.n && identidad; i++)
        if (h.datoDe[i] != i) identidad = false;
    if (identidad) {
        gpu.vecNodos = baseGPU;
        gpu.vecPropios = false;
    } else {
        VectoresHost porNodo;
        porNodo.n = h.n;
        porNodo.dim = base.dim;
        porNodo.datos.resize((size_t)h.n * base.dim);
        for (uint32_t i = 0; i < h.n; i++)
            memcpy(porNodo.datos.data() + (size_t)i * base.dim, base.vec(h.datoDe[i]), base.dim);
        subirVectores(porNodo, gpu.vecNodos);
        gpu.vecPropios = true;
        gpu.bytes += gpu.vecNodos.bytes();
    }
}

void liberarHnsw(HnswGPU& gpu) {
    cudaFree(gpu.vecinos0);
    cudaFree(gpu.vecinosArriba);
    cudaFree(gpu.inicioArriba);
    cudaFree(gpu.datoDe);
    cudaFree(gpu.entradas);
    if (gpu.vecPropios) liberarVectores(gpu.vecNodos);
    gpu = HnswGPU{};
}

// ============================================================================
// buscador
// ============================================================================
static uint32_t siguientePotencia2(uint32_t x) {
    uint32_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

void BuscadorHnsw::preparar(const HnswGPU& grafo, uint32_t nq, const ConfigBusqueda& c) {
    liberar();
    cfg = c;
    numConsultas = nq;

    // IGUAL QUE HNSW: bits = ef > 2 ? ceil(log2(ef))*2 - 2 : 2   (con ef = Ls en el nivel 0)
    bitsVis0 = (cfg.Ls > 2) ? (int)std::ceil(std::log2((double)cfg.Ls)) * 2 - 2 : 2;
    // is_inw: de sobra (se avisa con BANDERA_DESBORDE si se pasa de la mitad)
    capInw = siguientePotencia2((uint32_t)(32 * cfg.Ls + 2048));

    int Gp = redondear32(std::max(grafo.G0, grafo.GA));
    size_t porWarp = bytesPorWarpHnsw(cfg.Ls, Gp);
    warpsPorBloque = (int)std::min<size_t>(16, std::max<size_t>(1, (48 * 1024) / porWarp));
    cudaFuncAttributes atrib;
    CHECK_CUDA(cudaFuncGetAttributes(&atrib, kernelHnsw));
    warpsPorBloque = std::min(warpsPorBloque, atrib.maxThreadsPerBlock / 32);
    bytesCompartida = porWarp * warpsPorBloque;

    int bloquesPorSM = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bloquesPorSM, kernelHnsw,
                                                             warpsPorBloque * 32, bytesCompartida));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    bloques = std::max(1, bloquesPorSM) * prop.multiProcessorCount;
    size_t totalWarps = (size_t)bloques * warpsPorBloque;

    // las tablas con epoca empiezan en 0 (la epoca 0 no se usa nunca)
    size_t bytesVis = totalWarps * ((size_t)1 << bitsVis0) * sizeof(unsigned long long);
    size_t bytesInw = totalWarps * (size_t)capInw * sizeof(unsigned long long);
    CHECK_CUDA(cudaMalloc(&visitados, bytesVis));
    CHECK_CUDA(cudaMalloc(&inw, bytesInw));
    CHECK_CUDA(cudaMemset(visitados, 0, bytesVis));
    CHECK_CUDA(cudaMemset(inw, 0, bytesInw));
    epocaBase = 0;

    CHECK_CUDA(cudaMalloc(&idsDev, (size_t)nq * cfg.k * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&distsDev, (size_t)nq * cfg.k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&compDev, (size_t)nq * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&banderasDev, (size_t)nq * sizeof(int)));
}

void BuscadorHnsw::lanzar(const VectoresGPU& consultas, const HnswGPU& g, cudaStream_t stream) {
    ParamsHnsw P;
    P.vecinos0 = g.vecinos0;
    P.G0 = g.G0;
    P.vecinosArriba = g.vecinosArriba;
    P.GA = g.GA;
    P.inicioArriba = g.inicioArriba;
    P.datoDe = g.datoDe;
    P.entradas = g.entradas;
    P.numEntradas = g.numEntradas;
    P.nivelSuperior = g.nivelSuperior;
    P.palabras = g.vecNodos.palabras;
    P.normas = g.vecNodos.normas;
    P.palabrasQ = consultas.palabras;
    P.normasQ = consultas.normas;
    P.numConsultas = numConsultas;
    P.Ls = cfg.Ls;
    // IGUAL QUE EL WRAPPER: sin "visit" no se pone limit_eval (en HNSW vale n)
    P.limite = (cfg.limite > 0 && cfg.limite < (long)g.n) ? (uint32_t)cfg.limite : g.n;
    P.k = cfg.k;
    P.bitsVis0 = bitsVis0;
    P.capInw = capInw;
    // CADA LANZAMIENTO USA EPOCAS NUEVAS (una por consulta y nivel)
    const uint64_t epocasPorLanzamiento = (uint64_t)numConsultas * (g.nivelSuperior + 1) + 1;
    if ((uint64_t)epocaBase + epocasPorLanzamiento >= 0xffffffffull) {
        fprintf(stderr, "se acabaron las epocas: volver a llamar a preparar()\n");
        exit(1);
    }
    P.epocaBase = epocaBase;
    epocaBase += (uint32_t)epocasPorLanzamiento;
    P.visitados = visitados;
    P.inw = inw;
    P.salidaIds = idsDev;
    P.salidaDists = distsDev;
    P.salidaComp = compDev;
    P.salidaBanderas = banderasDev;
    kernelHnsw<<<bloques, warpsPorBloque * 32, bytesCompartida, stream>>>(P);
    CHECK_CUDA(cudaGetLastError());
}

void BuscadorHnsw::liberar() {
    cudaFree(visitados);
    cudaFree(inw);
    cudaFree(idsDev);
    cudaFree(distsDev);
    cudaFree(compDev);
    cudaFree(banderasDev);
    visitados = nullptr;
    inw = nullptr;
    idsDev = nullptr;
    distsDev = nullptr;
    compDev = nullptr;
    banderasDev = nullptr;
}
