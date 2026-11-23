"""hacer_figuras.py
Figuras extra para la presentacion (mismos datos que el paper).

    source ~/parlay/venv/bin/activate
    python presentacion/hacer_figuras.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "figuras")
os.makedirs(SALIDA, exist_ok=True)

AZUL = "#1f4e8c"
AZUL_CLARO = "#6f9fd8"
GRIS = "#9aa5b1"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---- 1. evolucion de las optimizaciones (RTX 3050, DiskANN Ls=75) ----
versiones = ["CPU\n12 hilos", "1 hilo por\ndistancia", "+ 8 hilos\npor vector",
             "+ lecturas\nagrupadas (final)", "final con\népocas"]
qps = [32079, 42856, 56781, 141071, 98320]
colores = [GRIS, AZUL_CLARO, AZUL_CLARO, AZUL, "#c0504d"]
fig, eje = plt.subplots(figsize=(8, 3.6))
barras = eje.bar(versiones, qps, color=colores)
for b, v in zip(barras, qps):
    eje.text(b.get_x() + b.get_width() / 2, v + 2500, f"{v/1000:.1f}k\n({v/qps[0]:.2f}x)",
             ha="center", fontsize=9)
eje.set_ylabel("consultas por segundo")
eje.set_ylim(0, 170000)
eje.set_title("DiskANN, Ls = 75, recall 0.955 (RTX 3050, solo kernel)")
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "optimizaciones.png"), dpi=200)
plt.close(fig)

# ---- 2. reparto del tiempo del kernel (DiskANN Ls=75) ----
secciones = ["distancias", "leer vecinos y filtro", "unión con la frontera",
             "ordenar candidatos", "corte y siguiente", "otros"]
datos = {
    "RTX 3050\n(1 hilo por dist.)": [64.8, 25.0, 4.5, 0.5, 0.9, 4.3],
    "RTX 3050\n(final)": [48.0, 24.2, 10.2, 3.0, 4.3, 10.3],
    "RTX 5050\n(final)": [28.1, 21.0, 13.7, 13.1, 11.0, 13.1],
}
paleta = [AZUL, AZUL_CLARO, "#a8c6ea", "#f2a541", "#e07a5f", GRIS]
fig, eje = plt.subplots(figsize=(8, 3.4))
filas = list(datos)
for i, fila in enumerate(filas):
    izq = 0
    for j, val in enumerate(datos[fila]):
        eje.barh(i, val, left=izq, color=paleta[j], label=secciones[j] if i == 0 else None)
        if val >= 6:
            eje.text(izq + val / 2, i, f"{val:.0f}%", ha="center", va="center", color="white",
                     fontsize=9, fontweight="bold")
        izq += val
eje.set_yticks(range(len(filas)))
eje.set_yticklabels(filas)
eje.invert_yaxis()
eje.set_xlim(0, 100)
eje.set_xlabel("% del tiempo de cada warp")
eje.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "perfil.png"), dpi=200)
plt.close(fig)

# ---- 3. microbenchmark de distancias ----
variantes = ["1 hilo", "1 warp", "8 hilos\n(usada)", "4 hilos"]
a = [148, 527, 555, 486]
b = [568, 996, 2590, 2350]
x = range(len(variantes))
fig, eje = plt.subplots(figsize=(7, 3.4))
ancho = 0.38
ba = eje.bar([i - ancho / 2 for i in x], a, ancho, color=AZUL_CLARO, label="RTX 3050")
bb = eje.bar([i + ancho / 2 for i in x], b, ancho, color=AZUL, label="RTX 5050")
for barras in (ba, bb):
    for r in barras:
        eje.text(r.get_x() + r.get_width() / 2, r.get_height() + 40, f"{r.get_height():.0f}",
                 ha="center", fontsize=9)
eje.set_xticks(list(x))
eje.set_xticklabels(variantes)
eje.set_ylabel("millones de distancias / s")
eje.set_title("Hilos que leen cada vector de 128 bytes")
eje.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "microbenchmark.png"), dpi=200)
plt.close(fig)

print("figuras en", SALIDA)
