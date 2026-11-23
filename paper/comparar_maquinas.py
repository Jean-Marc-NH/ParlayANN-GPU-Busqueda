"""comparar_maquinas.py
Junta los resultados de la comparacion 1 de las dos maquinas y arma las tablas y la
figura para el paper.

    source ~/parlay/venv/bin/activate
    python paper/comparar_maquinas.py

Entrada : resultados i5/gpu/comparacion1/configuraciones.csv
          resultados i7/gpu_t24/comparacion1/configuraciones.csv
          resultados i5|i7/.../micro_distancias.txt
Salida  : paper/comparacion_maquinas.md
          paper/figuras/speedup_maquinas.png
          paper/figuras/recall_qps_maquinas.png
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAQUINAS = {
    "A: i5-12450HX + RTX 3050": dict(
        csv="resultados i5/gpu/comparacion1/configuraciones.csv",
        micro="resultados i5/gpu/micro_distancias.txt", hilos=12),
    "B: i7-13700HX + RTX 5050": dict(
        csv="resultados i7/gpu_t24/comparacion1/configuraciones.csv",
        micro="resultados i7/gpu_t24/micro_distancias.txt", hilos=24),
}
ALGOS = ["ParDiskANN", "ParHCNNG", "ParPyNNDescent", "ParHNSW"]
NOMBRE = {"ParDiskANN": "DiskANN", "ParHCNNG": "HCNNG", "ParPyNNDescent": "PyNNDescent",
          "ParHNSW": "HNSW"}
UMBRALES = [0.90, 0.95, 0.98]


def mejor(df, col_rec, col_qps, u):
    s = df[df[col_rec] >= u]
    return float(s[col_qps].max()) if len(s) else np.nan


def leer_micro(ruta):
    """millones de distancias por segundo de cada variante (A, B, C, D)"""
    b = open(os.path.join(RAIZ, ruta), "rb").read()
    txt = b.decode("utf-16") if b[:2] == b"\xff\xfe" else b.decode("utf-8", "ignore")
    return {m.group(1): float(m.group(2))
            for m in re.finditer(r"([ABCD])\)[^\n]*?([\d.]+) M dist/s", txt)}


def main():
    filas = []
    tablas = {}
    for maq, info in MAQUINAS.items():
        t = pd.read_csv(os.path.join(RAIZ, info["csv"]))
        tablas[maq] = t
        for a in ALGOS:
            ta = t[t.algoritmo == a]
            for u in UMBRALES:
                cpu = mejor(ta, "recall_cpu", "qps_cpu", u)
                gpu = mejor(ta, "recall_gpu", "qps_gpu_total", u)
                filas.append(dict(maquina=maq, algoritmo=NOMBRE[a], recall=u, qps_cpu=cpu,
                                  qps_gpu=gpu, speedup=gpu / cpu))
    r = pd.DataFrame(filas)
    maqA, maqB = list(MAQUINAS)

    L = ["# Comparacion 1 entre maquinas\n",
         "Mejor QPS que cumple cada recall@10 minimo. GPU = de punta a punta.",
         f"CPU con todos sus hilos ({MAQUINAS[maqA]['hilos']} en A, "
         f"{MAQUINAS[maqB]['hilos']} en B).\n"]
    for u in UMBRALES:
        L.append(f"## Recall >= {u:.2f}\n")
        L.append("| Algoritmo | CPU A | GPU A | **x A** | CPU B | GPU B | **x B** | "
                 "CPU B/A | GPU B/A |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a in ALGOS:
            n = NOMBRE[a]
            fa = r[(r.maquina == maqA) & (r.algoritmo == n) & (r.recall == u)].iloc[0]
            fb = r[(r.maquina == maqB) & (r.algoritmo == n) & (r.recall == u)].iloc[0]
            L.append(f"| {n} | {fa.qps_cpu:,.0f} | {fa.qps_gpu:,.0f} | **{fa.speedup:.1f}x** | "
                     f"{fb.qps_cpu:,.0f} | {fb.qps_gpu:,.0f} | **{fb.speedup:.1f}x** | "
                     f"{fb.qps_cpu / fa.qps_cpu:.2f}x | {fb.qps_gpu / fa.qps_gpu:.2f}x |")
        L.append("")

    mA, mB = leer_micro(MAQUINAS[maqA]["micro"]), leer_micro(MAQUINAS[maqB]["micro"])
    L.append("## Microbenchmark de distancias (millones por segundo)\n")
    L.append("| Variante | A (RTX 3050) | B (RTX 5050) | B/A |")
    L.append("|---|---:|---:|---:|")
    nombres = {"A": "1 hilo por distancia", "B": "1 warp por distancia",
               "C": "8 hilos por distancia (la usada)", "D": "4 hilos por distancia"}
    for k in "ABCD":
        if k in mA and k in mB:
            L.append(f"| {nombres[k]} | {mA[k]:,.0f} | {mB[k]:,.0f} | {mB[k] / mA[k]:.2f}x |")
    with open(os.path.join(RAIZ, "paper", "comparacion_maquinas.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    r.to_csv(os.path.join(RAIZ, "paper", "comparacion_maquinas.csv"), index=False)

    # ---- figura 1: speedup a recall >= 0.95 por algoritmo y maquina ----
    os.makedirs(os.path.join(RAIZ, "paper", "figuras"), exist_ok=True)
    fig, eje = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(ALGOS))
    ancho = 0.38
    for i, maq in enumerate(MAQUINAS):
        v = [r[(r.maquina == maq) & (r.algoritmo == NOMBRE[a]) & (r.recall == 0.95)].speedup.iloc[0]
             for a in ALGOS]
        barras = eje.bar(x + (i - 0.5) * ancho, v, ancho, label=maq)
        for b, val in zip(barras, v):
            eje.text(b.get_x() + b.get_width() / 2, val + 0.15, f"{val:.1f}x", ha="center",
                     fontsize=8)
    eje.set_xticks(x)
    eje.set_xticklabels([NOMBRE[a] for a in ALGOS])
    eje.set_ylabel("speedup GPU / CPU")
    eje.set_title("Busqueda con recall@10 >= 0.95 (MSSPACEV-1M)")
    eje.axhline(1, color="gray", lw=0.8)
    eje.legend(fontsize=8)
    eje.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RAIZ, "paper", "figuras", "speedup_maquinas.png"), dpi=200)
    plt.close(fig)

    # ---- figura 2: recall vs QPS, DiskANN y HNSW, ambas maquinas ----
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    estilos = {maqA: ("tab:blue", "A"), maqB: ("tab:orange", "B")}
    for eje, a in zip(ejes, ["ParDiskANN", "ParHNSW"]):
        for maq, t in tablas.items():
            ta = t[t.algoritmo == a]
            color, letra = estilos[maq]
            c = ta.sort_values("recall_cpu")
            g = ta.sort_values("recall_gpu")
            eje.plot(c.recall_cpu.to_numpy(), c.qps_cpu.to_numpy(), "o--", color=color, ms=3,
                     label=f"CPU {letra}")
            eje.plot(g.recall_gpu.to_numpy(), g.qps_gpu_total.to_numpy(), "s-", color=color, ms=3,
                     label=f"GPU {letra}")
        eje.set_yscale("log")
        eje.set_xlim(0.80, 1.0)
        eje.set_title(NOMBRE[a])
        eje.set_xlabel("recall@10")
        eje.set_ylabel("consultas por segundo")
        eje.grid(True, which="both", alpha=0.3)
        eje.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RAIZ, "paper", "figuras", "recall_qps_maquinas.png"), dpi=200)
    plt.close(fig)

    print("\n".join(L))


if __name__ == "__main__":
    main()
