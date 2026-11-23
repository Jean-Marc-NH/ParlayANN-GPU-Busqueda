"""reporte_comparacion1.py
REPORTE DE LA COMPARACION 1 (busqueda CPU vs GPU sobre los mismos grafos)

Se corre en WSL con el entorno de cpu_baseline (ahi estan h5py, pandas y matplotlib):

    source ~/parlay/venv/bin/activate
    python reporte_comparacion1.py

Hace dos cosas:
  PASO 4 - VALIDACION VECINO POR VECINO
    compara, configuracion por configuracion, los vecinos que guardo
    comparacion1.exe (resultados/vecinos/*.bin) con los que guardo la CPU en los
    hdf5 de big-ann-benchmarks
  PASO 5 - TABLAS Y GRAFICAS
    QPS de la CPU (mejor de 5, sacado de los hdf5) contra QPS de la GPU solo kernel
    y de punta a punta (resultados/comparacion1_gpu.csv)

Salida en resultados/comparacion1/:
    validacion.csv, configuraciones.csv, resumen.md, recall_qps.png, speedup.png
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
ALGOS = ["ParDiskANN", "ParHCNNG", "ParPyNNDescent", "ParHNSW"]
UMBRALES = [0.90, 0.95, 0.98]
SIN_VECINO = 0xFFFFFFFF


def argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--hdf5", default=os.path.expanduser(
        "~/parlay/big-ann-benchmarks/results/msspacev-1M/10"),
        help="carpeta con los hdf5 de la CPU (una subcarpeta por algoritmo)")
    p.add_argument("--gpu", default=os.path.join(AQUI, "resultados", "comparacion1_gpu.csv"))
    p.add_argument("--vecinos", default=os.path.join(AQUI, "resultados", "vecinos"))
    p.add_argument("--datos", default=os.path.expanduser(
        "~/parlay/big-ann-benchmarks/data/MSSPACEV1B"),
        help="vectores base y consultas, para comparar DISTANCIAS cuando los ids difieren")
    p.add_argument("--hilos", type=int, default=None,
                   help="hilos de la corrida de CPU a usar (por defecto: la de mas hilos)")
    p.add_argument("--salida", default=os.path.join(AQUI, "resultados", "comparacion1"))
    return p.parse_args()


def config_de_nombre(nombre):
    """..._Ls_75.hdf5 -> (75, -1)    ..._Ls_10_visit_3.bin -> (10, 3)"""
    m = re.search(r"_Ls_(\d+)(?:_visit_(\d+))?\.(?:hdf5|bin)$", nombre)
    if not m:
        return None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else -1)


# ---------------------------------------------------------------------------
# CPU: hdf5 de big-ann-benchmarks
# ---------------------------------------------------------------------------
def hilos_disponibles(dir_hdf5):
    """numeros de hilos de las corridas de CPU guardadas en los hdf5"""
    hilos = set()
    for ruta in glob.glob(os.path.join(dir_hdf5, "*", "*.hdf5")):
        with h5py.File(ruta, "r") as f:
            hilos.add(int(f.attrs.get("threads", -1)))
    return sorted(h for h in hilos if h > 0)


def leer_cpu(dir_hdf5, hilos):
    filas, vecinos = [], {}
    for algo in ALGOS:
        for ruta in sorted(glob.glob(os.path.join(dir_hdf5, algo, "*.hdf5"))):
            cfg = config_de_nombre(os.path.basename(ruta))
            if cfg is None:
                continue
            with h5py.File(ruta, "r") as f:
                if int(f.attrs.get("threads", -1)) != hilos:
                    continue
                vec = np.array(f["neighbors"])
                nq = vec.shape[0]
                k = int(f.attrs["count"])
                # recall por consulta que guardo data_export.py (conteo, hay que dividir por k)
                if "metrics/knn/recalls" in f:
                    recall = float(np.mean(f["metrics/knn/recalls"])) / k
                else:
                    recall = np.nan
                filas.append(dict(algoritmo=algo, Ls=cfg[0], visit=cfg[1], recall_cpu=recall,
                                  qps_cpu=nq / float(f.attrs["best_search_time"]),
                                  build_cpu=float(f.attrs["build_time"])))
                vecinos[(algo, cfg[0], cfg[1])] = vec
    return pd.DataFrame(filas), vecinos


def leer_bin(ruta):
    with open(ruta, "rb") as fh:
        nq, k = np.frombuffer(fh.read(8), dtype=np.uint32)
        ids = np.frombuffer(fh.read(), dtype=np.uint32).reshape(int(nq), int(k))
    return ids


def cargar_vectores(dir_datos):
    """base y consultas int8 (formato .i8bin). None si no estan (se salta esa metrica)"""
    rb = os.path.join(dir_datos, "spacev1b_base.i8bin.crop_nb_1000000")
    rq = os.path.join(dir_datos, "query.i8bin")
    if not (os.path.exists(rb) and os.path.exists(rq)):
        print(f"  (no encuentro los vectores en {dir_datos}: no se comparan distancias)")
        return None, None
    base = np.fromfile(rb, dtype=np.int8, offset=8).reshape(-1, 100)
    q = np.fromfile(rq, dtype=np.int8, offset=8).reshape(-1, 100)
    return base, q


def mismas_distancias(filas, vc, vg, base, Q):
    """para las consultas en 'filas': los 10 vecinos de CPU y GPU, aunque tengan ids
    distintos, ¿estan a las MISMAS distancias? (empates). Devuelve cuantas si."""
    total = 0
    for ini in range(0, len(filas), 4000):
        f = filas[ini:ini + 4000]
        qq = Q[f].astype(np.int32)[:, None, :]
        dc = ((base[vc[f]].astype(np.int32) - qq) ** 2).sum(-1)
        dg = ((base[vg[f]].astype(np.int32) - qq) ** 2).sum(-1)
        total += int(np.all(np.sort(dc, 1) == np.sort(dg, 1), axis=1).sum())
    return total


# ---------------------------------------------------------------------------
# PASO 4: validacion vecino por vecino
# ---------------------------------------------------------------------------
def validar(vecinos_cpu, dir_vecinos, base=None, Q=None):
    filas = []
    for (algo, Ls, visit), vc in sorted(vecinos_cpu.items()):
        nombre = f"{algo}_Ls_{Ls}" + (f"_visit_{visit}" if visit >= 0 else "") + ".bin"
        ruta = os.path.join(dir_vecinos, nombre)
        if not os.path.exists(ruta):
            continue
        vg = leer_bin(ruta).astype(np.int64)
        vc = vc.astype(np.int64) & 0xFFFFFFFF   # int32 del hdf5 -> mismos bits que uint32
        if vg.shape != vc.shape:
            print(f"  {nombre}: tamanos distintos GPU {vg.shape} CPU {vc.shape}", file=sys.stderr)
            continue

        # consultas donde la GPU no llego a k vecinos ("cortas"): en esas la CPU lee
        # fuera de la frontera (basura), asi que no se comparan
        cortas = np.any(vg == SIN_VECINO, axis=1)
        validas = ~cortas
        igual_orden = np.all(vg == vc, axis=1) & validas
        igual_conjunto = np.all(np.sort(vg, axis=1) == np.sort(vc, axis=1), axis=1) & validas
        # ids de la CPU que no aparecen en la respuesta de la GPU
        esta = (vc[:, :, None] == vg[:, None, :]).any(axis=2)
        faltan = int((~esta)[validas].sum())

        n = int(validas.sum())
        # cuando los ids difieren: ¿es solo un empate de distancia?
        distintas = np.where(validas & ~igual_conjunto)[0]
        if base is not None and len(distintas) > 0:
            iguales_dist = int(igual_conjunto.sum()) + mismas_distancias(distintas, vc, vg, base, Q)
        elif base is not None:
            iguales_dist = int(igual_conjunto.sum())
        else:
            iguales_dist = np.nan
        filas.append(dict(algoritmo=algo, Ls=Ls, visit=visit, consultas=n, cortas=int(cortas.sum()),
                          iguales_orden=int(igual_orden.sum()),
                          iguales_conjunto=int(igual_conjunto.sum()),
                          pct_iguales_conjunto=100.0 * igual_conjunto.sum() / max(n, 1),
                          iguales_distancias=iguales_dist,
                          ids_distintos=faltan))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# PASO 5: tablas
# ---------------------------------------------------------------------------
def mejor_qps(df, col_recall, col_qps, umbral):
    sel = df[df[col_recall] >= umbral]
    return float(sel[col_qps].max()) if len(sel) else np.nan


def resumen_por_algoritmo(tabla):
    filas = []
    for algo in ALGOS:
        t = tabla[tabla.algoritmo == algo]
        if t.empty:
            continue
        for u in UMBRALES:
            cpu = mejor_qps(t, "recall_cpu", "qps_cpu", u)
            gk = mejor_qps(t, "recall_gpu", "qps_gpu_kernel", u)
            gt = mejor_qps(t, "recall_gpu", "qps_gpu_total", u)
            filas.append(dict(algoritmo=algo, recall_min=u, qps_cpu=cpu, qps_gpu_total=gt,
                              qps_gpu_kernel=gk, x_total=gt / cpu, x_kernel=gk / cpu))
    return pd.DataFrame(filas)


def leer_texto(ruta):
    try:
        with open(ruta) as f:
            return f.read().strip()
    except OSError:
        return "(no encontrado)"


def fmt(x, dec=0):
    return "-" if pd.isna(x) else f"{x:,.{dec}f}"


def escribir_resumen(ruta, tabla, resumen, valid, hilos):
    cpus = sorted(glob.glob(os.path.join(RAIZ, "cpu_baseline", "results",
                                         f"msspacev-1M_t{hilos}_*", "hardware.txt")))
    equipo_cpu = leer_texto(cpus[-1]) if cpus else "(no encontrado)"
    equipo_gpu = leer_texto(os.path.join(AQUI, "resultados", "comparacion1_equipo.txt"))

    L = []
    L.append("# Comparacion 1 - Busqueda CPU vs GPU (MSSPACEV-1M)\n")
    L.append("Mismos grafos (construidos por ParlayANN en CPU), mismas 29 316 consultas, k = 10,")
    L.append("mismas 18 configuraciones por algoritmo. QPS = mejor de 5 corridas.\n")
    L.append("- **CPU**: ParlayANN original via big-ann-benchmarks (`run.py`), "
             f"{hilos} hilos.")
    L.append("- **GPU total**: copiar consultas a la GPU + kernel + traer ids y distancias.")
    L.append("- **GPU kernel**: solo el kernel (consultas ya en la GPU).\n")
    L.append("## Equipo\n")
    L.append("CPU:\n```\n" + equipo_cpu + "\n```")
    L.append("GPU:\n```\n" + equipo_gpu + "\n```\n")

    L.append("## Mejor QPS que cumple cada recall@10 minimo\n")
    L.append("| Algoritmo | Recall >= | QPS CPU | QPS GPU total | QPS GPU kernel | x total | x kernel |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in resumen.iterrows():
        L.append(f"| {r.algoritmo} | {r.recall_min:.2f} | {fmt(r.qps_cpu)} | {fmt(r.qps_gpu_total)} | "
                 f"{fmt(r.qps_gpu_kernel)} | {fmt(r.x_total, 2)} | {fmt(r.x_kernel, 2)} |")

    L.append("\n## Validacion (GPU vs CPU)\n")
    L.append("| Algoritmo | Configs con recall identico | Consultas con los mismos 10 vecinos | "
             "Consultas con las mismas 10 distancias | Ids distintos | Consultas cortas | "
             "Configs deterministas |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for algo in ALGOS:
        t = tabla[tabla.algoritmo == algo]
        v = valid[valid.algoritmo == algo] if not valid.empty else valid
        if t.empty:
            continue
        iguales = int(t.recall_igual.sum())
        det = "-" if t.determinista.isna().all() else f"{int(t.determinista.sum())} de {len(t)}"
        if not v.empty:
            pct = 100.0 * v.iguales_conjunto.sum() / max(v.consultas.sum(), 1)
            pct_d = ("-" if v.iguales_distancias.isna().all() else
                     f"{100.0 * v.iguales_distancias.sum() / max(v.consultas.sum(), 1):.4f} %")
            L.append(f"| {algo} | {iguales} de {len(t)} | {pct:.4f} % | {pct_d} | "
                     f"{int(v.ids_distintos.sum())} | {int(v.cortas.sum())} | {det} |")
        else:
            L.append(f"| {algo} | {iguales} de {len(t)} | - | - | - | - | {det} |")
    L.append("\n*Mismas 10 distancias*: aunque algun id sea distinto, los 10 vecinos de la GPU")
    L.append("estan a exactamente las mismas distancias que los de la CPU (empates).")
    L.append("\n*Determinista*: la corrida de calentamiento y la ultima corrida devolvieron")
    L.append("exactamente los mismos vecinos.")
    L.append("\n*Consultas cortas*: la frontera termino con menos de 10 elementos; ahi la CPU")
    L.append("devuelve memoria sin inicializar, por eso no se comparan vecino por vecino.\n")

    L.append("## Todas las configuraciones\n")
    L.append("| Algoritmo | Ls | visit | recall CPU | recall GPU | QPS CPU | QPS GPU total | "
             "QPS GPU kernel | x total |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in tabla.iterrows():
        L.append(f"| {r.algoritmo} | {r.Ls} | {'-' if r.visit < 0 else int(r.visit)} | "
                 f"{r.recall_cpu:.5f} | {r.recall_gpu:.5f} | {fmt(r.qps_cpu)} | {fmt(r.qps_gpu_total)} | "
                 f"{fmt(r.qps_gpu_kernel)} | {fmt(r.x_total, 2)} |")
    L.append("\nGraficas: `recall_qps.png` y `speedup.png`.")
    with open(ruta, "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# PASO 5: graficas
# ---------------------------------------------------------------------------
def graficar(tabla, carpeta):
    fig, ejes = plt.subplots(2, 2, figsize=(12, 9))
    for eje, algo in zip(ejes.flat, ALGOS):
        t = tabla[tabla.algoritmo == algo]
        if t.empty:
            eje.set_visible(False)
            continue
        c = t.sort_values("recall_cpu")
        g = t.sort_values("recall_gpu")
        # .to_numpy(): matplotlib 3.3 no acepta columnas de pandas 2.x directamente
        eje.plot(c.recall_cpu.to_numpy(), c.qps_cpu.to_numpy(), "o-", label="CPU (ParlayANN)")
        eje.plot(g.recall_gpu.to_numpy(), g.qps_gpu_total.to_numpy(), "s-",
                 label="GPU punta a punta")
        eje.plot(g.recall_gpu.to_numpy(), g.qps_gpu_kernel.to_numpy(), "^--",
                 label="GPU solo kernel", alpha=0.7)
        eje.set_yscale("log")
        eje.set_title(algo)
        eje.set_xlabel("recall@10")
        eje.set_ylabel("consultas por segundo")
        eje.grid(True, which="both", alpha=0.3)
        eje.legend(fontsize=8)
    fig.suptitle("MSSPACEV-1M: recall vs QPS (CPU vs GPU, mismos grafos)")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "recall_qps.png"), dpi=130)
    plt.close(fig)

    fig, ejes = plt.subplots(2, 2, figsize=(12, 9))
    for eje, algo in zip(ejes.flat, ALGOS):
        t = tabla[tabla.algoritmo == algo].sort_values("recall_cpu")
        if t.empty:
            eje.set_visible(False)
            continue
        eje.plot(t.recall_cpu.to_numpy(), t.x_total.to_numpy(), "s-", label="punta a punta")
        eje.plot(t.recall_cpu.to_numpy(), t.x_kernel.to_numpy(), "^--", label="solo kernel",
                 alpha=0.7)
        eje.axhline(1.0, color="gray", lw=1)
        eje.set_title(algo)
        eje.set_xlabel("recall@10 (configuracion)")
        eje.set_ylabel("speedup GPU / CPU")
        eje.grid(True, alpha=0.3)
        eje.legend(fontsize=8)
    fig.suptitle("Speedup de la busqueda en GPU por configuracion")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "speedup.png"), dpi=130)
    plt.close(fig)


def main():
    a = argumentos()
    if not os.path.exists(a.gpu):
        sys.exit(f"No existe {a.gpu}: primero correr comparacion1.exe")
    os.makedirs(a.salida, exist_ok=True)

    if a.hilos is None:
        disponibles = hilos_disponibles(a.hdf5)
        if not disponibles:
            sys.exit(f"No hay hdf5 de la CPU en {a.hdf5}: primero correr run_benchmark.sh")
        a.hilos = disponibles[-1]   # la corrida principal = la de mas hilos
        print(f"CPU: corridas disponibles con hilos {disponibles}, se usa {a.hilos}")
    print(f"CPU: leyendo hdf5 de {a.hdf5} (hilos = {a.hilos})")
    cpu, vecinos_cpu = leer_cpu(a.hdf5, a.hilos)
    if cpu.empty:
        sys.exit("No se encontraron hdf5 de la CPU con esos hilos")
    gpu = pd.read_csv(a.gpu)

    cols = ["algoritmo", "Ls", "visit", "recall_gpu", "qps_gpu_kernel", "qps_gpu_total",
            "comp_prom", "desbordes", "fronteras_cortas"]
    if "determinista" not in gpu.columns:
        gpu["determinista"] = np.nan
    cols.append("determinista")
    tabla = cpu.merge(gpu[cols], on=["algoritmo", "Ls", "visit"], how="inner")
    tabla["recall_igual"] = (tabla.recall_cpu - tabla.recall_gpu).abs() < 1e-12
    tabla["x_kernel"] = tabla.qps_gpu_kernel / tabla.qps_cpu
    tabla["x_total"] = tabla.qps_gpu_total / tabla.qps_cpu
    tabla["orden"] = tabla.algoritmo.map({x: i for i, x in enumerate(ALGOS)})
    tabla = tabla.sort_values(["orden", "visit", "Ls"], ascending=[True, False, True]).drop(columns="orden")
    tabla.to_csv(os.path.join(a.salida, "configuraciones.csv"), index=False)

    print("PASO 4: validacion vecino por vecino")
    base, Q = cargar_vectores(a.datos)
    valid = validar(vecinos_cpu, a.vecinos, base, Q)
    valid.to_csv(os.path.join(a.salida, "validacion.csv"), index=False)

    resumen = resumen_por_algoritmo(tabla)
    escribir_resumen(os.path.join(a.salida, "resumen.md"), tabla, resumen, valid, a.hilos)
    graficar(tabla, a.salida)

    # ---- consola ----
    pd.set_option("display.width", 160)
    print("\nMejor QPS que cumple cada recall minimo:")
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nValidacion:")
    for algo in ALGOS:
        t = tabla[tabla.algoritmo == algo]
        v = valid[valid.algoritmo == algo] if not valid.empty else valid
        if t.empty:
            continue
        txt = f"  {algo:15s} recall identico en {int(t.recall_igual.sum())} de {len(t)} configs"
        if not t.determinista.isna().all():
            txt += f" | deterministas {int(t.determinista.sum())} de {len(t)}"
        if not v.empty:
            pct = 100.0 * v.iguales_conjunto.sum() / max(v.consultas.sum(), 1)
            txt += f" | mismos 10 vecinos en {pct:.4f} % de las consultas | ids distintos {int(v.ids_distintos.sum())}"
            if not v.iguales_distancias.isna().all():
                pct_d = 100.0 * v.iguales_distancias.sum() / max(v.consultas.sum(), 1)
                txt += f" | mismas distancias {pct_d:.4f} %"
        print(txt)
    print(f"\nListo: {a.salida}/ (resumen.md, configuraciones.csv, validacion.csv, recall_qps.png, speedup.png)")


if __name__ == "__main__":
    main()
