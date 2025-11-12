#!/usr/bin/env bash
# Download and prepare BLAST databases (nr/swissprot) using update_blastdb.pl
#
# Usage:
#   scripts/setup_blast_db.sh --dir /data/blast --dbs nr,swissprot
#
# Notes:
# - Requires BLAST+ installed (conda install -c bioconda blast)
# - Exports BLASTDB to the chosen directory
# - Prints the export commands to add to your shell rc

set -euo pipefail

DIR="data/blast"
DBS="nr"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      DIR="$2"; shift 2 ;;
    --dbs)
      DBS="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed -e 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Default to repo-local data folder if not provided
if [[ -z "$DIR" ]]; then
  DIR="data/blast"
fi

if ! command -v update_blastdb\.pl >/dev/null 2>&1; then
  echo "Error: update_blastdb.pl not found in PATH. Install BLAST+ (e.g., 'conda install -c bioconda blast')." >&2
  exit 1
fi

mkdir -p "$DIR"
# Resolve absolute path to avoid update_blastdb writing to cwd
DIR="$(cd "$DIR" && pwd)"
export BLASTDB="$DIR"

IFS=',' read -r -a DB_ARR <<< "$DBS"
for db in "${DB_ARR[@]}"; do
  db_trimmed="$(echo "$db" | xargs)"
  if [[ -z "$db_trimmed" ]]; then continue; fi

  # Handle UniRef90 separately (from UniProt, not NCBI)
  # MUST come before generic existence check to handle resume logic
  if [[ "$db_trimmed" == "uniref90" ]]; then
    UNIREF_URL="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"
    UNIREF_FASTA_GZ="$DIR/uniref90.fasta.gz"
    UNIREF_FASTA="$DIR/uniref90.fasta"
    ARIA2_CONTROL="$DIR/uniref90.fasta.gz.aria2"
    PREFIX_CAND1="$DIR/$db_trimmed"

    # Check if BLAST database files exist (final state)
    if [[ -f "${PREFIX_CAND1}.phr" ]] && [[ -f "${PREFIX_CAND1}.pin" ]] && [[ -f "${PREFIX_CAND1}.psq" ]]; then
      echo "[setup_blast_db] UniRef90 BLAST database already complete: $PREFIX_CAND1"
      DB_PREFIX="$PREFIX_CAND1"
      continue
    fi

    echo "[setup_blast_db] Setting up UniRef90 database (~30GB download, ~50GB unpacked)..."

    # Check if FASTA is ready (skip download/decompression)
    if [[ -f "$UNIREF_FASTA" ]]; then
      echo "[setup_blast_db] UniRef90 FASTA already exists: $UNIREF_FASTA"
    else
      # Check if download is incomplete (aria2 control file exists)
      if [[ -f "$ARIA2_CONTROL" ]]; then
        echo "[setup_blast_db] Detected incomplete download, resuming..."
      elif [[ -f "$UNIREF_FASTA_GZ" ]]; then
        echo "[setup_blast_db] Compressed file exists: $UNIREF_FASTA_GZ"
      else
        echo "[setup_blast_db] Starting fresh download..."
      fi

      # Download if not complete (no .gz file OR .aria2 control file exists)
      if [[ ! -f "$UNIREF_FASTA_GZ" ]] || [[ -f "$ARIA2_CONTROL" ]]; then
        # Try aria2c first (MUCH faster with multiple connections)
        if command -v aria2c >/dev/null 2>&1; then
          echo "[setup_blast_db] Using aria2c for fast multi-connection download..."
          cd "$DIR"
          aria2c \
            --max-connection-per-server=12 \
            --split=12 \
            --min-split-size=5M \
            --continue=true \
            --max-tries=5 \
            --retry-wait=10 \
            --timeout=60 \
            --auto-file-renaming=false \
            --allow-overwrite=true \
            --file-allocation=none \
            --console-log-level=notice \
            --summary-interval=10 \
            "$UNIREF_URL" \
            -o uniref90.fasta.gz

          # Check if download succeeded (control file should be gone)
          if [[ -f "$ARIA2_CONTROL" ]]; then
            echo "[setup_blast_db] ERROR: Download incomplete (aria2 control file still exists)" >&2
            echo "[setup_blast_db] Run the script again to resume, or check network connection" >&2
            exit 1
          fi
        elif command -v curl >/dev/null 2>&1; then
          echo "[setup_blast_db] aria2c not found, falling back to curl (slower)..."
          echo "[setup_blast_db] Tip: Install aria2 for 5-10x faster downloads: brew install aria2"
          curl -C - -# -L "$UNIREF_URL" -o "$UNIREF_FASTA_GZ" --retry 5 --retry-delay 10
        elif command -v wget >/dev/null 2>&1; then
          echo "[setup_blast_db] curl not found, using wget (slower)..."
          echo "[setup_blast_db] Tip: Install aria2 for 5-10x faster downloads: brew install aria2"
          wget -c --progress=bar:force "$UNIREF_URL" -O "$UNIREF_FASTA_GZ"
        else
          echo "[setup_blast_db] ERROR: No download tool found (aria2c, curl, or wget)" >&2
          exit 1
        fi
      fi

      # Decompress if we have the .gz file but not the .fasta
      if [[ -f "$UNIREF_FASTA_GZ" ]] && [[ ! -f "$UNIREF_FASTA" ]]; then
        echo "[setup_blast_db] Decompressing UniRef90 (~50GB uncompressed)..."
        gunzip -v "$UNIREF_FASTA_GZ"
      fi
    fi

    # Build BLAST database
    echo "[setup_blast_db] Building BLAST database from FASTA (10-30 minutes)..."
    if ! command -v makeblastdb >/dev/null 2>&1; then
      echo "[setup_blast_db] ERROR: makeblastdb not found in PATH" >&2
      exit 1
    fi

    makeblastdb -in "$UNIREF_FASTA" -dbtype prot -out "$PREFIX_CAND1" \
      -title "UniRef90" -parse_seqids -blastdb_version 5

    echo "[setup_blast_db] UniRef90 database built successfully"
    DB_PREFIX="$PREFIX_CAND1"
    continue
  fi

  # For other databases (nr, swissprot), check if already exists
  PREFIX_CAND1="$DIR/$db_trimmed"
  PREFIX_CAND2="$DIR/$db_trimmed/$db_trimmed"
  if ls "${PREFIX_CAND1}".* >/dev/null 2>&1 || ls "${PREFIX_CAND2}".* >/dev/null 2>&1; then
    echo "[setup_blast_db] DB '$db_trimmed' already exists, skipping download"
    if ls "${PREFIX_CAND1}".* >/dev/null 2>&1; then
      DB_PREFIX="$PREFIX_CAND1"
    else
      DB_PREFIX="$PREFIX_CAND2"
    fi
    echo "[setup_blast_db] Detected DB prefix: $DB_PREFIX"
    continue
  fi

  echo "[setup_blast_db] Downloading DB: $db_trimmed into $DIR"
  
  # Try update_blastdb.pl first with progress monitoring
  DOWNLOAD_SUCCESS=false
  MONITOR_INTERVAL=30
  STALL_THRESHOLD=3
  
  (
    cd "$DIR"
    update_blastdb.pl --decompress "$db_trimmed" &
    DOWNLOAD_PID=$!
    
    # Monitor progress
    STALL_COUNT=0
    LAST_SIZE=0
    while kill -0 $DOWNLOAD_PID 2>/dev/null; do
      sleep $MONITOR_INTERVAL
      CURRENT_SIZE=$(du -sk "$DIR" 2>/dev/null | awk '{print $1}')
      SIZE_DIFF=$((CURRENT_SIZE - LAST_SIZE))
      
      if [[ $SIZE_DIFF -lt 1024 ]]; then
        STALL_COUNT=$((STALL_COUNT + 1))
        echo "[setup_blast_db] Warning: Download stalled ($STALL_COUNT/$STALL_THRESHOLD checks, ${SIZE_DIFF}KB in ${MONITOR_INTERVAL}s)"
        if [[ $STALL_COUNT -ge $STALL_THRESHOLD ]]; then
          echo "[setup_blast_db] Download appears stuck, killing update_blastdb.pl..."
          kill $DOWNLOAD_PID 2>/dev/null
          wait $DOWNLOAD_PID 2>/dev/null || true
          break
        fi
      else
        echo "[setup_blast_db] Progress: +${SIZE_DIFF}KB"
        STALL_COUNT=0
      fi
      LAST_SIZE=$CURRENT_SIZE
    done
    
    wait $DOWNLOAD_PID 2>/dev/null && DOWNLOAD_SUCCESS=true || DOWNLOAD_SUCCESS=false
  )
  
  if [[ "$DOWNLOAD_SUCCESS" == "true" ]]; then
    echo "[setup_blast_db] Successfully downloaded with update_blastdb.pl"
  else
    echo "[setup_blast_db] update_blastdb.pl failed or stalled, trying direct download with curl..."
    # Fallback to direct curl download
    NCBI_BASE="https://ftp.ncbi.nlm.nih.gov/blast/db"
    (
      cd "$DIR"
      # Download all tar.gz files for this database
      for i in $(seq -w 0 99); do
        FILE="${db_trimmed}.${i}.tar.gz"
        URL="${NCBI_BASE}/${FILE}"
        if curl -C - -f -O "$URL" --retry 10 --retry-delay 30 2>/dev/null; then
          echo "[setup_blast_db] Downloaded $FILE"
          tar -xzf "$FILE"
          rm "$FILE"
        else
          # Try without sequence number
          if [[ "$i" == "00" ]]; then
            FILE="${db_trimmed}.tar.gz"
            URL="${NCBI_BASE}/${FILE}"
            if curl -C - -f -O "$URL" --retry 10 --retry-delay 30; then
              echo "[setup_blast_db] Downloaded $FILE"
              tar -xzf "$FILE"
              rm "$FILE"
            fi
          fi
          break
        fi
      done
    )
  fi

  # Detect where the DB files landed and compute the prefix
  if ls "${PREFIX_CAND1}".* >/dev/null 2>&1; then
    DB_PREFIX="$PREFIX_CAND1"
  elif ls "${PREFIX_CAND2}".* >/dev/null 2>&1; then
    DB_PREFIX="$PREFIX_CAND2"
  else
    echo "[setup_blast_db] ERROR: Could not find unpacked DB files for '$db_trimmed' under $DIR" >&2
    echo "Tried: ${PREFIX_CAND1}.* and ${PREFIX_CAND2}.*" >&2
    echo "Contents of $DIR:" >&2
    ls -la "$DIR" >&2 || true
    exit 1
  fi

  echo "[setup_blast_db] Detected DB prefix: $DB_PREFIX"
done

# Prepare helpful examples for detected prefixes (if present)
SWISSPROT_PREFIX=""
if ls "$DIR/swissprot".* >/dev/null 2>&1; then
  SWISSPROT_PREFIX="$DIR/swissprot"
elif ls "$DIR/swissprot/swissprot".* >/dev/null 2>&1; then
  SWISSPROT_PREFIX="$DIR/swissprot/swissprot"
fi
NR_PREFIX=""
if ls "$DIR/nr".* >/dev/null 2>&1; then
  NR_PREFIX="$DIR/nr"
elif ls "$DIR/nr/nr".* >/dev/null 2>&1; then
  NR_PREFIX="$DIR/nr/nr"
fi

echo ""
echo "[OK] BLAST databases prepared in: $DIR"
echo ""
echo "To make BLAST find your databases in new shells, add:"
echo "  export BLASTDB=\"$DIR\""
echo ""
if [[ -n "$SWISSPROT_PREFIX" ]]; then
  echo "SwissProt prefix detected: $SWISSPROT_PREFIX"
  echo "  export BLAST_DB=\"$SWISSPROT_PREFIX\""
fi
if [[ -n "$NR_PREFIX" ]]; then
  echo "NR prefix detected: $NR_PREFIX"
  echo "  export BLAST_DB=\"$NR_PREFIX\""
fi
echo "  export BLAST_DB_NAME=swissprot  # optional convenience var"
