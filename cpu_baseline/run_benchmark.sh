#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — Ejecuta los 4 algoritmos Parlay (CPU) con el protocolo de
#                    la presentacion y deja los resultados en ./results
#
# Uso:
#   bash run_benchmark.sh [DATASET] [HILOS]
#     DATASET  random-xs | msspacev-1M          (por defecto msspacev-1M)
#     HILOS    numero de hilos de construccion  (por defecto: todos, nproc)
#
# Variables opcionales:
#   ALGOS="ParDiskANN ParHCNNG ParPyNNDescent ParHNSW"   algoritmos a correr
#   RUNS=5                                                repeticiones por config.
#   PARLAY_HOME=~/parlay                                  donde corrio setup.sh
#
# Ejemplos:
#   bash run_benchmark.sh random-xs            # prueba rapida (~1 min)
#   bash run_benchmark.sh msspacev-1M          # experimento completo, todos los hilos
#   bash run_benchmark.sh msspacev-1M 8        # mismo numero de hilos que la laptop
#   ALGOS=ParDiskANN bash run_benchmark.sh msspacev-1M 1   # DiskANN con 1 hilo
# =============================================================================
set -euo pipefail

DATASET="${1:-msspacev-1M}"
THREADS="${2:-$(nproc)}"
ALGOS="${ALGOS:-ParDiskANN ParHCNNG ParPyNNDescent ParHNSW}"
RUNS="${RUNS:-5}"
PARLAY_HOME="${PARLAY_HOME:-$HOME/parlay}"
ANN_DIR="$PARLAY_HOME/ParlayANN-ppopp24"
BAB_DIR="$PARLAY_HOME/big-ann-benchmarks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SCRIPT_DIR/results/${DATASET}_t${THREADS}_${STAMP}"

[ -x "$PARLAY_HOME/venv/bin/python" ] || { echo "Primero ejecuta: bash setup.sh"; exit 1; }
# shellcheck disable=SC1091
source "$PARLAY_HOME/venv/bin/activate"
export PYTHONPATH="$ANN_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export PARLAY_NUM_THREADS="$THREADS"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/log.txt"

# --- Informacion del equipo (para el informe) -------------------------------
{
  echo "fecha: $(date -Iseconds)"
  echo "dataset: $DATASET   hilos: $THREADS   runs: $RUNS"
  echo "algoritmos: $ALGOS"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "nproc (WSL): $(nproc)"
  echo "ram (WSL): $(free -h | awk '/Mem:/{print $2}')"
  echo "ParlayANN commit: $(git -C "$ANN_DIR" rev-parse --short HEAD)"
  echo "big-ann-benchmarks: $(git -C "$BAB_DIR" describe --tags --always)"
} | tee "$OUT_DIR/hardware.txt"

cd "$BAB_DIR"

# --- YAML con los parametros del artifact + numero de hilos T ----------------
DEFS="$BAB_DIR/local_${DATASET}_t${THREADS}.yaml"
python - "$DATASET" "$THREADS" "$DEFS" <<'EOF'
import json, sys, yaml
ds, threads, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(open("artifact_eval.yaml"))
sec = cfg[ds]
for algo in sec.values():
    for group in algo["run-groups"].values():
        args = json.loads(group["args"])
        for a in args:
            a["T"] = threads          # mismo mecanismo que documentan los autores
        group["args"] = json.dumps(args)
yaml.safe_dump({ds: sec}, open(out, "w"), sort_keys=False)
print("definiciones:", out)
EOF

# --- Ejecucion ---------------------------------------------------------------
for A in $ALGOS; do
  echo -e "\n========== $A | $DATASET | $THREADS hilos ==========" | tee -a "$LOG"
  python run.py --nodocker --rebuild --runs "$RUNS" \
      --algorithm "$A" --dataset "$DATASET" --definitions "$DEFS" 2>&1 | tee -a "$LOG"
done

# --- Exportar y resumir --------------------------------------------------------
echo -e "\n========== Exportando resultados ==========" | tee -a "$LOG"
python data_export.py --output "$OUT_DIR/all_results.csv" 2>&1 | tail -3 | tee -a "$LOG"
python plot.py --dataset "$DATASET" --output "$OUT_DIR/${DATASET}.png" 2>&1 | tail -3 | tee -a "$LOG" || true
python "$SCRIPT_DIR/summarize.py" "$OUT_DIR/all_results.csv" "$DATASET" "$THREADS" \
    | tee "$OUT_DIR/resumen.txt"

echo -e "\nResultados en: $OUT_DIR"
