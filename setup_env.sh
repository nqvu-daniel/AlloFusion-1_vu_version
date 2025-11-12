#!/usr/bin/env bash
# AlloFusion Environment Setup
# Source this file to activate the conda environment and set all required variables
#
# Usage:
#   source setup_env.sh
#   . setup_env.sh
#
# This script must be SOURCED, not executed directly.

# Check if being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: This script must be sourced, not executed."
  echo "Usage: source setup_env.sh"
  exit 1
fi

# Determine script directory (repo root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || return 1

# Check if .env.allofusion exists
if [[ ! -f .env.allofusion ]]; then
  echo "ERROR: .env.allofusion not found."
  echo "Run bootstrap first:"
  echo "  bash scripts/bootstrap_cloud.sh --env-yml environment.allofusion-osx-arm64.yml --blast-dir data/blast --dbs swissprot --model-dir data/models/prot_t5_xl_uniref50"
  return 1
fi

# Source environment variables
echo "[setup_env] Loading environment variables from .env.allofusion"
source .env.allofusion

# Get conda env name from .env file (set by bootstrap)
if [[ -z "$CONDA_ENV_NAME" ]]; then
  echo "ERROR: CONDA_ENV_NAME not set in .env.allofusion"
  echo "This usually means bootstrap_cloud.sh didn't complete successfully."
  echo "Please re-run bootstrap."
  return 1
fi

ENV_NAME="$CONDA_ENV_NAME"

# Activate conda environment
echo "[setup_env] Activating conda environment: $ENV_NAME"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH."
  return 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Check if env exists
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ERROR: Conda environment '$ENV_NAME' not found."
  echo "Run bootstrap first to create it."
  return 1
fi

conda activate "$ENV_NAME"

# Validate critical paths
ERRORS=0

if [[ -z "$PROT_T5_PATH" ]]; then
  echo "ERROR: PROT_T5_PATH not set in .env.allofusion"
  ERRORS=$((ERRORS + 1))
elif [[ ! -d "$PROT_T5_PATH" ]]; then
  echo "ERROR: ProtT5 model directory not found: $PROT_T5_PATH"
  ERRORS=$((ERRORS + 1))
elif [[ ! -f "$PROT_T5_PATH/pytorch_model.bin" ]]; then
  echo "ERROR: ProtT5 model weights not found: $PROT_T5_PATH/pytorch_model.bin"
  echo "Run: python scripts/prefetch_prot_t5.py --dest $PROT_T5_PATH"
  ERRORS=$((ERRORS + 1))
fi

if [[ -z "$BLAST_DB" ]]; then
  echo "ERROR: BLAST_DB not set in .env.allofusion"
  ERRORS=$((ERRORS + 1))
elif ! ls "${BLAST_DB}"*.* >/dev/null 2>&1; then
  echo "ERROR: BLAST database files not found at: ${BLAST_DB}.*"
  echo "Run: bash scripts/setup_blast_db.sh --dir data/blast --dbs swissprot"
  ERRORS=$((ERRORS + 1))
fi

if [[ $ERRORS -gt 0 ]]; then
  echo ""
  echo "SETUP INCOMPLETE: $ERRORS validation errors found."
  echo "Please fix the above errors before running AlloFusion."
  return 1
fi

echo ""
echo "[OK] Environment ready!"
echo "  Conda env: $ENV_NAME (activated)"
echo "  ProtT5:    $PROT_T5_PATH"
echo "  BLAST_DB:  $BLAST_DB"
echo ""
echo "You can now run:"
echo "  cd AlloFusion-main"
echo "  python AlloFusionMain.py --PDBID 4ZSI --CHAIN B"
