// datos_gpu.h
// estructuras que viven en la memoria de la GPU y funciones para subirlas / liberarlas
#pragma once

#include "lectores.h"

// vectores ya rellenados a 128 bytes + su norma al cuadrado
struct VectoresGPU {
    uint32_t n = 0;
    int* palabras = nullptr;   // n * 32 palabras (cada palabra = 4 valores int8)
    int* normas = nullptr;     // n normas ||p||^2

    size_t bytes() const { return (size_t)n * DIM_PAD + (size_t)n * sizeof(int); }
};

// grafo como matriz fija n x gradoMax
// la fila i tiene los vecinos del punto i y despues SIN_VECINO hasta completar
// asi el vecino j del punto i esta en vecinos[i*gradoMax + j], sin indirecciones
struct GrafoGPU {
    uint32_t n = 0;
    uint32_t gradoMax = 0;
    uint32_t* vecinos = nullptr;   // n * gradoMax
    uint32_t* grados = nullptr;    // n

    size_t bytes() const {
        return (size_t)n * gradoMax * sizeof(uint32_t) + (size_t)n * sizeof(uint32_t);
    }
};

void subirVectores(const VectoresHost& host, VectoresGPU& gpu);
void liberarVectores(VectoresGPU& gpu);

void subirGrafo(const GrafoHost& host, GrafoGPU& gpu);
void liberarGrafo(GrafoGPU& gpu);
