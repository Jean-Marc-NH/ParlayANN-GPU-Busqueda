#!/usr/bin/env bash
# =============================================================================
# setup.sh — Prepara la linea base en CPU de ParlayANN (artifact PPoPP'24)
#            dentro de Ubuntu/WSL2, SIN Docker y SIN sudo.
#
# Usa exactamente el mismo codigo que la presentacion:
#   - ParlayANN-ppopp24       commit 45587ba  (algoritmos C++, sin modificar)
#   - big-ann-benchmarks      tag ppopp-artifact-final (commit 80e5bc47)
#   - Parametros de artifact_eval.yaml para msspacev-1M
#
# Diferencia con la presentacion: en vez de correr el codigo C++ dentro de la
# imagen Docker, el modulo Python _ParlayANNpy se compila aqui mismo con el
# mismo comando que usa el Dockerfile (python/compile.sh) y se ejecuta
# run.py con --nodocker.
#
# Uso:   bash setup.sh
# Variable opcional: PARLAY_HOME (por defecto ~/parlay)
# =============================================================================
set -euo pipefail

PARLAY_HOME="${PARLAY_HOME:-$HOME/parlay}"
ANN_DIR="$PARLAY_HOME/ParlayANN-ppopp24"
BAB_DIR="$PARLAY_HOME/big-ann-benchmarks"
VENV="$PARLAY_HOME/venv"
ANN_COMMIT="45587ba"
BAB_TAG="ppopp-artifact-final"

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }

# --- 0. Herramientas del sistema ---------------------------------------------
for t in g++ make git curl; do
  command -v "$t" >/dev/null || { echo "Falta '$t'. Instalalo con: sudo apt install -y build-essential git curl"; exit 1; }
done
mkdir -p "$PARLAY_HOME"

# --- 1. uv (gestor de Python, sin sudo) --------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  log "Instalando uv en ~/.local/bin (no modifica .bashrc)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

# --- 2. Repositorios de los autores (versiones fijadas) ----------------------
if [ ! -d "$ANN_DIR/.git" ]; then
  log "Clonando ParlayANN-ppopp24"
  git clone https://github.com/magdalendobson/ParlayANN-ppopp24.git "$ANN_DIR"
fi
log "ParlayANN-ppopp24 -> commit $ANN_COMMIT"
git -C "$ANN_DIR" fetch -q origin || true
git -C "$ANN_DIR" checkout -q "$ANN_COMMIT"
git -C "$ANN_DIR" submodule update --init --recursive -q

if [ ! -d "$BAB_DIR/.git" ]; then
  log "Clonando big-ann-benchmarks ($BAB_TAG)"
  git clone --branch "$BAB_TAG" https://github.com/magdalendobson/big-ann-benchmarks.git "$BAB_DIR"
fi
git -C "$BAB_DIR" checkout -q "$BAB_TAG"

# --- 3. Entorno Python 3.10 (el artifact pide 3.10) --------------------------
if [ ! -x "$VENV/bin/python" ]; then
  log "Creando venv con Python 3.10"
  uv venv --python 3.10 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
log "Instalando dependencias de big-ann-benchmarks (requirements_py3.10.txt)"
if ! uv pip install -r "$BAB_DIR/requirements_py3.10.txt" pybind11; then
  # matplotlib 3.3.4 no publica wheels para 3.10; si su compilacion falla,
  # usamos la version mas cercana con wheel (solo afecta a los graficos).
  log "matplotlib==3.3.4 fallo; reintentando con matplotlib==3.5.3"
  sed 's/^matplotlib==.*/matplotlib==3.5.3/' "$BAB_DIR/requirements_py3.10.txt" > "$PARLAY_HOME/requirements_local.txt"
  uv pip install -r "$PARLAY_HOME/requirements_local.txt" pybind11
fi

# --- 4. Mismo ajuste que en la presentacion: perf_counter en el runner -------
log "Reemplazando time.time() por time.perf_counter() en base_runner.py"
sed -i 's/time\.time()/time.perf_counter()/g' "$BAB_DIR/benchmark/algorithms/base_runner.py"

# --- 5. Compilar el modulo C++ (mismas flags que python/compile.sh) ----------
log "Compilando _ParlayANNpy (puede tardar 3-8 minutos)"
cd "$ANN_DIR/python"
EXT_SUFFIX="$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
g++ -DSTATS -DHOMEGROWN -pthread -mcx16 -O3 -Wall -shared -std=c++17 -march=native \
    -DNDEBUG -I . -fPIC $(python -m pybind11 --includes) module.cpp \
    -o "_ParlayANNpy${EXT_SUFFIX}" -DHOMEGROWN -pthread -ldl 2> "$PARLAY_HOME/compile_warnings.log" \
  || { cat "$PARLAY_HOME/compile_warnings.log"; echo "Error compilando _ParlayANNpy"; exit 1; }
PYTHONPATH="$ANN_DIR/python" python -c "import wrapper; print('Modulo OK:', wrapper.__file__)"

# --- 6. Datasets --------------------------------------------------------------
cd "$BAB_DIR"
log "Generando dataset de prueba random-xs"
python create_dataset.py --dataset random-xs
if [ "${SKIP_MSSPACEV:-0}" != "1" ]; then
  log "Descargando MSSPACEV-1M (~127 MB: 1M vectores int8 + consultas + ground truth)"
  python create_dataset.py --dataset msspacev-1M
fi

log "Listo. Siguiente paso:  bash run_benchmark.sh random-xs   (prueba)"
log "                 luego: bash run_benchmark.sh msspacev-1M  (experimento)"
