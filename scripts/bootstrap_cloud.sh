#!/usr/bin/env bash
# Bootstrap an AlloFusion environment and prefetch required assets.
#
# It will:
#  1) Create/update the conda env from a YAML
#  2) Prefetch ProtT5 to a local folder (models/...)
#  3) Download BLAST database(s) (e.g., swissprot) to a chosen directory
#  4) Emit an .env file you can source to set required variables
#
# Usage:
#   Mac (ARM64):
#     bash scripts/bootstrap_cloud.sh \
#         --env-yml environment.allofusion-osx-arm64.yml \
#         --blast-dir data/blast \
#         --dbs uniref90 \
#         --model-dir data/models/prot_t5_xl_uniref50
#   CUDA 12.1:
#     bash scripts/bootstrap_cloud.sh \
#         --env-yml environment.allofusion-cuda121.yml \
#         --blast-dir data/blast \
#         --dbs uniref90 \
#         --model-dir data/models/prot_t5_xl_uniref50
#
# Database options:
#   --dbs uniref90   : Recommended (30GB download, 50GB uncompressed, excellent coverage)
#   --dbs swissprot  : Smaller alternative (1GB, curated proteins only)
#   --dbs nr         : Full NCBI (189GB, not recommended unless required)
#
# After it completes:
#   source .env.allofusion
#
set -euo pipefail

ENV_YML="environment.allofusion-osx-arm64.yml"
BLAST_DIR="data/blast"
DBS="uniref90"
MODEL_DIR="data/models/prot_t5_xl_uniref50"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-yml) ENV_YML="$2"; shift 2 ;;
    --blast-dir) BLAST_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --dbs) DBS="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed -e 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$ENV_YML" ]]; then
  echo "Error: env YAML not found: $ENV_YML" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH." >&2
  exit 1
fi

# Extract env name from YAML
ENV_NAME=$(awk '/^name:/ {print $2; exit}' "$ENV_YML")
if [[ -z "${ENV_NAME:-}" ]]; then
  echo "Error: Failed to parse env name from $ENV_YML" >&2
  exit 1
fi

echo "[bootstrap] Creating/updating conda env: $ENV_NAME from $ENV_YML"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -f "$ENV_YML" --prune
else
  conda env create -f "$ENV_YML"
fi

echo "[bootstrap] Activating conda env: $ENV_NAME"
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[bootstrap] Prefetching ProtT5 into $MODEL_DIR (minimal files only)"
mkdir -p "$MODEL_DIR"
python scripts/prefetch_prot_t5.py --repo Rostlab/prot_t5_xl_uniref50 --dest "$MODEL_DIR"

# Validate prefetch succeeded
if [[ ! -f "$MODEL_DIR/pytorch_model.bin" ]]; then
  echo "[bootstrap] ERROR: ProtT5 prefetch failed - pytorch_model.bin not found" >&2
  exit 1
fi

echo "[bootstrap] Downloading BLAST databases ($DBS) into $BLAST_DIR"
bash scripts/setup_blast_db.sh --dir "$BLAST_DIR" --dbs "$DBS"

# Determine primary DB name (first item if comma-separated)
DB_MAIN=$(echo "$DBS" | awk -F',' '{print $1}')

# Detect DB prefix under BLAST_DIR for env export
ABS_BLAST_DIR="$(cd "$BLAST_DIR" && pwd)"
PREF1="$ABS_BLAST_DIR/$DB_MAIN"
PREF2="$ABS_BLAST_DIR/$DB_MAIN/$DB_MAIN"
if ls "${PREF1}".* >/dev/null 2>&1; then
  DB_PREFIX="$PREF1"
elif ls "${PREF2}".* >/dev/null 2>&1; then
  DB_PREFIX="$PREF2"
else
  echo "[bootstrap] WARNING: Could not auto-detect DB prefix for $DB_MAIN under $ABS_BLAST_DIR" >&2
  echo "           You may need to set BLAST_DB manually. Proceeding with $PREF1." >&2
  DB_PREFIX="$PREF1"
fi

# Write environment file
cat > .env.allofusion <<EOF
# Source this file to configure your shell for AlloFusion
export CONDA_ENV_NAME="$ENV_NAME"
export TRANSFORMERS_NO_TORCHVISION=1
export PROT_T5_PATH="$(cd "$MODEL_DIR" && pwd)"
export BLASTDB="$ABS_BLAST_DIR"
export BLAST_DB="$DB_PREFIX"
export BLAST_DB_NAME="$DB_MAIN"
EOF

echo ""
echo "========================================="
echo "[OK] Bootstrap complete!"
echo "========================================="
echo ""
echo "Setup created:"
echo "  - Conda environment: $ENV_NAME"
echo "  - ProtT5 model: $MODEL_DIR"
echo "  - BLAST database: $DB_PREFIX"
echo "  - Config file: .env.allofusion"
echo ""
echo "To activate everything in your current shell, run:"
echo ""
echo "    source setup_env.sh"
echo ""
echo "Then you can run AlloFusion:"
echo "    cd AlloFusion-main"
echo "    python AlloFusionMain.py --PDBID 4ZSI --CHAIN B"
echo ""
