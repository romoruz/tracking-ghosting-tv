#!/usr/bin/env bash
# Instalación del entorno (Python >= 3.10).
#   ./setup.sh          Entorno base (pasos 0-2)
#   ./setup.sh --torch  Añade PyTorch CPU (paso 3)
set -euo pipefail
cd "$(dirname "$0")"

WITH_TORCH=0
[[ "${1:-}" == "--torch" ]] && WITH_TORCH=1

command -v python3 >/dev/null || { echo "ERROR: falta python3"; exit 1; }
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
echo ">> Python $PYV"
python3 - <<'PY' || { echo "ERROR: se requiere Python >= 3.10"; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)
PY

if [[ ! -d venv ]]; then
  echo ">> Creando venv..."
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip -q

echo ">> Instalando dependencias base..."
pip install -q -r requirements.txt

if [[ $WITH_TORCH -eq 1 ]]; then
  echo ">> Instalando PyTorch (build CPU, sin CUDA)..."
  # Build CPU para evitar descargar dependencias de CUDA
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ">> Instalando el paquete en modo editable..."
pip install -q -e .

mkdir -p data/raw data/processed reports/figures reports/tables

cat <<'MSG'

  Listo.

  Activa el entorno con:      source venv/bin/activate
  Prueba todo sin red con:    ./run.sh demo
  Descarga datos reales con:  ./run.sh sportec

MSG
