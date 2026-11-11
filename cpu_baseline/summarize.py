"""Resume all_results.csv (data_export.py) en la misma tabla que la presentacion.

Uso: python summarize.py all_results.csv [DATASET] [HILOS]

Para cada algoritmo reporta:
  - tiempo de construccion (s)
  - maximo QPS entre configuraciones con recall@10 >= 0.95 y >= 0.98
  - recall maximo observado
"""
import sys

import pandas as pd

csv = sys.argv[1]
dataset = sys.argv[2] if len(sys.argv) > 2 else None
threads = int(sys.argv[3]) if len(sys.argv) > 3 else None

df = pd.read_csv(csv)
if dataset is not None and "dataset" in df:
    df = df[df["dataset"] == dataset]
if threads is not None and "threads" in df:
    df = df[df["threads"] == threads]
if df.empty:
    sys.exit(f"No hay filas para dataset={dataset} hilos={threads} en {csv}")

rec = "recall/ap"
order = ["ParDiskANN", "ParHCNNG", "ParHNSW", "ParPyNNDescent"]
algos = [a for a in order if a in set(df["algorithm"])] + \
        sorted(set(df["algorithm"]) - set(order))


def best_qps(g, thr):
    sel = g[g[rec] >= thr]
    return f"{sel['qps'].max():>10,.0f}" if len(sel) else f"{'—':>10}"


print(f"\nDataset: {dataset}   Hilos: {threads}   Filas: {len(df)}")
print(f"{'Algoritmo':<16}{'Build (s)':>11}{'QPS@>=0.95':>12}{'QPS@>=0.98':>12}"
      f"{'Recall max':>12}{'Configs':>9}")
print("-" * 72)
for a in algos:
    g = df[df["algorithm"] == a]
    build = g["build"].max() if "build" in g else float("nan")
    print(f"{a:<16}{build:>11.1f} {best_qps(g, 0.95)}  {best_qps(g, 0.98)}"
          f"{g[rec].max():>12.3f}{len(g):>9}")
print("\nQPS = mejor de RUNS repeticiones (protocolo big-ann-benchmarks); "
      "cada celda es el mejor punto del barrido que supera el umbral.")
