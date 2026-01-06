#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Download NCBI BLAST databases using update_blastdb.pl (ships with BLAST+).

Usage:
  bash scripts/setup_blast_db.sh --dir /path/to/blastdb --db swissprot

Then configure AlloFusion PSSM with one of:
  export BLAST_DB=/path/to/blastdb/swissprot/swissprot
  # OR
  export BLASTDB=/path/to/blastdb
  export BLAST_DB_NAME=swissprot
EOF
}

BLASTDB_DIR=""
DB_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      BLASTDB_DIR="$2"
      shift 2
      ;;
    --db)
      DB_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${BLASTDB_DIR}" || -z "${DB_NAME}" ]]; then
  echo "[ERROR] Missing --dir or --db" >&2
  usage
  exit 2
fi

if ! command -v update_blastdb.pl >/dev/null 2>&1; then
  echo "[ERROR] update_blastdb.pl not found. Install BLAST+ first (e.g., conda-forge::blast)." >&2
  exit 1
fi

mkdir -p "${BLASTDB_DIR}/${DB_NAME}"
echo "[INFO] Downloading BLAST DB '${DB_NAME}' into '${BLASTDB_DIR}/${DB_NAME}'..."
(
  cd "${BLASTDB_DIR}/${DB_NAME}"
  update_blastdb.pl --decompress "${DB_NAME}"
)

echo "[INFO] Done."
echo "[NEXT] Set:"
echo "  export BLAST_DB=${BLASTDB_DIR}/${DB_NAME}/${DB_NAME}"
echo "or:"
echo "  export BLASTDB=${BLASTDB_DIR}"
echo "  export BLAST_DB_NAME=${DB_NAME}"

