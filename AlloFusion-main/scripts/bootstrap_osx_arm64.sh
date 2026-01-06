#!/usr/bin/env bash
set -euo pipefail

ENV_YML="${ENV_YML:-environment.allofusion-osx-arm64.yml}"
ENV_NAME="${ENV_NAME:-allofusion}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found."
  echo "Install Miniforge (Apple Silicon) and retry:"
  echo "  https://github.com/conda-forge/miniforge"
  exit 1
fi

BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${BASE}/etc/profile.d/conda.sh"

echo "[INFO] Creating/updating conda env '${ENV_NAME}' from '${ENV_YML}'..."
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ENV_YML}"
else
  conda env create -n "${ENV_NAME}" -f "${ENV_YML}"
fi

conda activate "${ENV_NAME}"

echo "[INFO] Verifying key tools..."
python -c "import numpy, pandas, requests, networkx, scipy, Bio; print('python deps OK')"
if command -v mkdssp >/dev/null 2>&1; then
  echo "[INFO] mkdssp found: $(command -v mkdssp)"
else
  echo "[WARN] mkdssp not found on PATH (DSSP features will fall back)."
fi
if command -v psiblast >/dev/null 2>&1; then
  echo "[INFO] psiblast found: $(command -v psiblast)"
else
  echo "[WARN] psiblast not found on PATH (PSSM features will fail)."
  echo "       On macOS arm64, the simplest install is Homebrew:"
  echo "         brew install blast"
fi

echo
echo "[NEXT] Activate the environment in your shell:"
echo "  conda activate ${ENV_NAME}"
echo
echo "[NEXT] Run a quick feature smoke test (no ML models needed):"
echo "  python scripts/smoke_features.py --pdb 4ZSI_gnm_zs.pdb --chain A"
echo
echo "[NEXT] Download ProtT5 locally (required for full AlloFusionMain runs):"
echo "  python scripts/prefetch_prot_t5.py --repo Rostlab/prot_t5_xl_uniref50 --dest data/models/prot_t5_xl_uniref50"
echo "  export PROT_T5_PATH=$(pwd)/data/models/prot_t5_xl_uniref50"
