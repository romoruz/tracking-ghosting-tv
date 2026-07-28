#!/usr/bin/env bash
#
# Orquestador del pipeline.
#
#   ./run.sh demo       datos sintéticos (sin red)
#   ./run.sh sportec    descarga datos Sportec y corre pasos 1-2
#   ./run.sh metrica    descarga dataset Metrica y corre pasos 1-2
#   ./run.sh report     re-ejecuta pasos 1-2 sobre datos existentes
#   ./run.sh train      entrena y evalúa el imputador residual
#   ./run.sh test       ejecuta tests unitarios
#   ./run.sh clean      borra salidas derivadas (conserva data/processed)
#
set -euo pipefail
cd "$(dirname "$0")"

[[ -d venv ]] || { echo "Falta el entorno. Corre ./setup.sh primero."; exit 1; }
# shellcheck disable=SC1091
source venv/bin/activate

CMD="${1:-demo}"
step() { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }

case "$CMD" in
  demo)
    step "Paso 0: generando partidos sintéticos"
    python scripts/00_download.py --provider synthetic
    step "Paso 1: validando el simulador de cámara"
    python scripts/01_viewport_report.py
    step "Paso 2: escalera de baselines"
    python scripts/02_run_baselines.py --boot 300
    ;;
  sportec|metrica)
    step "Paso 0: descargando $CMD (la primera vez tarda varios minutos)"
    python scripts/00_download.py --provider "$CMD"
    step "Paso 1: validando el simulador de cámara"
    python scripts/01_viewport_report.py
    step "Paso 2: escalera de baselines"
    python scripts/02_run_baselines.py --boot 1000
    ;;
  report)
    python scripts/01_viewport_report.py
    python scripts/02_run_baselines.py --boot 1000
    ;;
  train)
    step "Paso 3: entrenando el imputador residual (modo causal)"
    python scripts/04_train.py "${@:2}"
    step "Evaluando contra B4 en el test interno"
    python scripts/05_evaluate_model.py
    ;;
  test)
    python -m pytest tests/ -v
    ;;
  clean)
    rm -rf reports/figures/* reports/tables/* .pytest_cache
    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
    echo "Salidas derivadas borradas (data/processed intacto)."
    ;;
  *)
    sed -n '3,13p' "$0"; exit 1;;
esac

printf '\n\033[1;32mListo.\033[0m Resultados en reports/\n'
