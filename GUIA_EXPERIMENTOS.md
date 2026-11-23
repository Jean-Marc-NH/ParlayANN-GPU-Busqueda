# Guia de experimentos (para el short paper)

Objetivo: comparar la busqueda de ParlayANN en **CPU** contra nuestra version en **GPU**
sobre los **mismos grafos**, en **dos maquinas** de distinta generacion:

| | Maquina A | Maquina B |
|---|---|---|
| CPU | i5-12450HX (8 nucleos / 12 hilos) | i7 de 13.a generacion (anotar modelo exacto) |
| GPU | RTX 3050 6 GB Laptop (Ampere, `sm_86`) | RTX 5050 (Blackwell, `sm_120`) |

Con dos maquinas el paper puede responder no solo "cuanto acelera la GPU" sino tambien
"como cambia la ventaja de la GPU cuando ambos lados son mas nuevos".

---

## 0. Condiciones de medicion (en TODAS las corridas, CPU y GPU)

- [ ] Laptop **enchufada**.
- [ ] Plan de energia de Windows en **maximo rendimiento** y la app del fabricante
      (Armoury Crate, Vantage, OMEN, etc.) en modo **Performance / Turbo**.
- [ ] "Al estar enchufado, suspender despues de: **Nunca**". No cerrar la tapa.
- [ ] Sin otros programas abiertos (navegador, Teams, juegos, antivirus escaneando).
- [ ] Dejar enfriar unos minutos entre la corrida de CPU y la de GPU.
- [ ] Anotar si algo raro paso durante la corrida.

`comparacion1.bat` guarda el plan de energia activo en
`cuda/resultados/comparacion1_equipo_windows.txt`; revisarlo despues de cada corrida.

---

## 1. Instalacion en la maquina B (una sola vez)

En Windows:
1. **Visual Studio 2022** con el workload "Desarrollo para el escritorio con C++".
   Si no es la edicion *Community*, cambiar la ruta de `vcvars64.bat` en `cuda/build.bat`.
2. **CUDA Toolkit 12.8 o mas nuevo** (la RTX 5050 necesita `sm_120`) y el driver NVIDIA al dia.
3. **WSL2 con Ubuntu**: `wsl --install -d Ubuntu` (reiniciar si lo pide).
4. **Git** y clonar el repo:
   `git clone https://github.com/Jean-Marc-NH/ParlayANN-GPU-Busqueda.git`

Dentro de Ubuntu (WSL):
```bash
sudo apt update && sudo apt install -y build-essential git curl   # solo si falta algo
cd /mnt/c/<ruta-del-repo>/cpu_baseline
bash setup.sh                    # ~10 min: repos de los autores, Python 3.10, dataset
bash run_benchmark.sh random-xs  # prueba rapida (~1 min)
```

---

## 2. Linea base en CPU (en cada maquina)

```bash
cd /mnt/c/<ruta-del-repo>/cpu_baseline
bash run_benchmark.sh msspacev-1M      # todos los hilos de la maquina, 30-60 min
```

Usa todos los hilos que ve WSL (`nproc`). Los programas de GPU y el reporte detectan
solos esa corrida (la de mas hilos) y su grafo.

---

## 3. Comparacion 1 en GPU (en cada maquina)

**Pruebas rapidas primero** (segundos). En PowerShell, dentro de `cuda`:

```powershell
.\build.bat
$datos = wsl -d Ubuntu -- bash -c "wslpath -w ~/parlay/big-ann-benchmarks/data"
.\comparacion1.exe $datos - ParHNSW 75 1
.\comparacion1.exe $datos - ParDiskANN 200 1
```

En ambas: `=CPU` debe decir **SI** y al final `deterministas ... 1 de 1`.
Si alguna dice NO, no seguir: avisar con la salida completa.

**Corrida completa** (15-30 min):

```powershell
.\comparacion1.bat
```

Esperado: `72 de 72` con recall identico, `72 de 72` deterministas, `desbordes 0`.

---

## 4. Experimentos extra para el paper (en cada maquina)

| # | Que | Comando (en `cuda`, PowerShell) | Para que seccion |
|---|---|---|---|
| E1 | Techo de memoria de la GPU | `.\micro_distancias.exe $datos > resultados\micro_distancias.txt` | Discusion: por que el speedup tiene un techo |
| E2 | Donde se va el tiempo del kernel | `.\paso2_perfil.exe $datos - DiskANN 75 > resultados\perfil_diskann_75.txt` y lo mismo con `500` | Diseno / Discusion |
| E3 | Repetibilidad | correr `comparacion1.bat` **dos veces** y comparar los QPS | Metodologia (variacion entre corridas) |
| E4 | Escalabilidad de la CPU (opcional, largo) | en WSL: `for t in 1 2 4 8; do ALGOS=ParDiskANN bash run_benchmark.sh msspacev-1M $t; done` | Resultados: con cuantos hilos de CPU se iguala a la GPU |

Notas:
- E1 y E2 reciben la carpeta de datos como primer argumento (`$datos`, ver paso 3).
- E4 reconstruye el grafo con cada cantidad de hilos y tarda bastante (con 1 hilo puede
  pasar de 20 minutos). Despues de E4, **la corrida principal sigue siendo la de mas
  hilos**, asi que la comparacion de GPU no cambia.
- Antes de E3, copiar `resultados\comparacion1\` a otra carpeta para no pisarla.

**Solo en la maquina A** (ya existe, no esta en git): `cuda/resultados/paso2_log_v1.txt`
(version 1 del kernel) y `paso2_log.txt` (version optimizada). Sirven para la tabla de
optimizaciones del paper (v1 → v2 → v3). **Guardarlos**.

---

## 5. Que guardar de cada maquina

Crear una carpeta por maquina, por ejemplo `datos_paper/maquinaB/`, y copiar:

| Archivo | Que tiene |
|---|---|
| `cpu_baseline/results/msspacev-1M_t<N>_<fecha>/` | CPU: `resumen.txt`, `all_results.csv`, `hardware.txt`, `log.txt` |
| `cuda/resultados/comparacion1/` | `resumen.md`, `configuraciones.csv`, `validacion.csv`, `recall_qps.png`, `speedup.png` |
| `cuda/resultados/comparacion1_gpu.csv` | Todas las mediciones de GPU |
| `cuda/resultados/comparacion1_equipo.txt` y `..._equipo_windows.txt` | GPU, driver, CUDA, CPU, RAM, plan de energia |
| `cuda/resultados/micro_distancias.txt`, `perfil_*.txt` | Experimentos E1 y E2 |

---

## 6. Estructura sugerida del short paper (4-6 paginas)

Preguntas que el paper responde:
- **P1 (exactitud):** ¿la busqueda en GPU devuelve lo mismo que ParlayANN? → validacion.
- **P2 (rendimiento):** ¿cuanto mas rapida es, a igual recall, en cada maquina? → QPS@recall.
- **P3 (limites):** ¿que limita a la GPU? → microbenchmark + perfil.
- **P4 (hardware):** ¿cambia la ventaja entre generaciones? → maquina A vs B.

| Seccion | Contenido | De donde sale |
|---|---|---|
| Resumen | Problema, que se hizo, resultado principal (speedup a recall 0.95 en ambas maquinas) | `resumen.md` |
| 1. Introduccion | ANNS en grafos, ParlayANN es solo CPU, contribucion | paper original |
| 2. Antecedentes | Beam search, DiskANN/HCNNG/PyNNDescent/HNSW, ParlayANN | paper original, README |
| 3. Diseno en GPU | Un warp por consulta; distancias exactas con `__dp4a`; como se garantiza el mismo resultado (filtro con perdidas, multiconjuntos, HNSW secuencial); optimizaciones (8 hilos por vector, lecturas juntas, epocas) | README "Como funciona", codigo |
| 4. Metodologia | Dataset, 18 configs, mismos grafos, mejor de 5, punta a punta vs kernel, maquinas, condiciones | esta guia, `*_equipo*.txt` |
| 5. Resultados | Tabla QPS@0.90/0.95/0.98 (CPU, GPU) por maquina; curvas recall-QPS; validacion y determinismo; tabla de optimizaciones v1→v3 | `resumen.md`, `.png`, logs de la maquina A |
| 6. Discusion | Limite de memoria (E1), desglose (E2), costo de copiar datos (kernel vs punta a punta), A vs B | E1, E2, `resumen.md` |
| 7. Limitaciones | 1M puntos; laptops; solo busqueda (la construccion sigue en CPU); hilos distintos en cada maquina | — |
| 8. Conclusion y trabajo futuro | Construccion en GPU (comparacion 2) | README "Pasos futuros" |

Figuras recomendadas (maximo 3-4 en un short paper):
1. Recall vs QPS, CPU vs GPU, DiskANN y HNSW, ambas maquinas.
2. Speedup a recall ≥ 0.95 por algoritmo y maquina (barras).
3. Tabla de optimizaciones del kernel (v1 → v3) con el perfil por secciones.

Amenazas a la validez (mencionarlas en el paper):
- Las dos CPUs usan distinto numero de hilos (cada una usa todos los suyos).
- Laptops: la temperatura y el modo de energia afectan; por eso E3 y la seccion 0.
- WSL2 para la CPU y Windows nativo para la GPU.
- Un solo dataset (MSSPACEV-1M) y un solo tamano.
