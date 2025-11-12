import math
import numpy as np
import pickle
import os
import pandas as pd
import subprocess
import logging


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. Configure BLAST by setting '{name}'."
        )
    return val


def _get_pssm_logger(out_dir: str, pdbid: str) -> logging.Logger:
    logger_name = f"pssm.{pdbid}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{pdbid}_pssm.log")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info("PSSM logging initialized (%s)", os.path.abspath(log_path))
    return logger


def gene_PSSM(pssmdir: str, pdbid: str, db_prefix: str | None = None, out_dir: str | None = None):
    """
    Generate ASCII PSSM files using psiblast for sequences in `{pssmdir}/{pdbid}.csv`.

    Parameters
    - pssmdir: folder containing `{pdbid}.csv` with two columns [pdb_name, sequence]
    - pdbid: dataset id (e.g., '4zsi')
    - db_prefix: full path prefix to BLAST database (e.g., '/data/blast/swissprot/swissprot').
                 If None, uses env 'BLAST_DB'.
    - out_dir: directory to write PSSM files; defaults to env 'PSSM_OUT_DIR' or `{pssmdir}`.
    """
    df = pd.read_csv(f'./{pssmdir}/{pdbid}.csv', header=None)
    pdb_names = df[0]
    seqs = df[1]

    db_prefix = db_prefix or os.environ.get('BLAST_DB')
    if not db_prefix:
        # If BLASTDB points to a directory, allow using a default DB name
        blastdb_dir = os.environ.get('BLASTDB')
        default_name = os.environ.get('BLAST_DB_NAME', 'swissprot')
        if blastdb_dir:
            db_prefix = os.path.join(blastdb_dir, default_name)
    if not db_prefix:
        raise EnvironmentError(
            "BLAST database path is not configured. Set BLAST_DB to the full database prefix (e.g., /path/to/swissprot/swissprot)."
        )

    out_dir = out_dir or os.environ.get('PSSM_OUT_DIR', pssmdir)
    os.makedirs(out_dir, exist_ok=True)
    logger = _get_pssm_logger(out_dir, pdbid)
    logger.info("Preparing to generate PSSM for %d sequences in %s", len(pdb_names), pssmdir)

    # Write sequences and run psiblast
    for i in range(len(pdb_names)):
        name = pdb_names[i]
        seq = seqs[i]
        fasta_path = os.path.join(out_dir, 'Temporary.fasta')
        with open(fasta_path, 'w') as f:
            f.write(name + '\n')
            f.write(seq)
        pssm_name = name.split('>')[1] if '>' in name else name.strip()
        out_pssm = os.path.join(out_dir, f"{pssm_name}.pssm")
        logger.info("Running psiblast for %s (%d/%d)", pssm_name, i + 1, len(pdb_names))
        psiblast_cmd = [
            'psiblast',
            '-query', fasta_path,
            '-db', db_prefix,
            '-evalue', '0.001',
            '-num_iterations', '3',
            '-out_ascii_pssm', out_pssm,
        ]
        try:
            subprocess.run(psiblast_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors='ignore') if e.stderr else "no stderr captured"
            logger.error("psiblast failed for %s (%s). stderr: %s", pssm_name, name, stderr)
            raise RuntimeError(
                f"psiblast failed for {name}. stderr: {stderr}"
            ) from e
        logger.info("psiblast completed for %s -> %s", pssm_name, out_pssm)


def getPSSM(pssmdir: str, pdbid: str, window_size: int = 3, db_prefix: str | None = None, out_dir: str | None = None):
    """
    Compute PSSM features with a sliding window from psiblast outputs.

    Requires either:
    - `db_prefix` argument pointing to a valid BLAST DB prefix, or
    - environment `BLAST_DB` (or `BLASTDB` directory with `BLAST_DB_NAME`, default 'swissprot')
    """
    resolved_out_dir = out_dir or os.environ.get('PSSM_OUT_DIR', pssmdir)
    logger = _get_pssm_logger(resolved_out_dir, pdbid)
    logger.info("Starting PSSM feature pipeline for %s (window=%d)", pdbid, window_size)
    gene_PSSM(pssmdir, pdbid, db_prefix=db_prefix, out_dir=resolved_out_dir)

    df = pd.read_csv(f'./{pssmdir}/{pdbid}.csv', header=None)
    pdb_names = df[0]
    seqs = df[1]
    pssms = []
    pdb_ids = []

    # Determine pssm output dir (should match gene_PSSM)
    out_base = resolved_out_dir

    for i in range(len(pdb_names)):
        pdb_name = pdb_names[i].split('>')[1]
        seq = seqs[i]
        pssm_path = os.path.join(out_base, f'{pdb_name}.pssm')
        logger.info("Parsing PSSM for %s from %s", pdb_name, pssm_path)

        if not os.path.exists(pssm_path):
            raise FileNotFoundError(
                f"PSSM file not found: {pssm_path}. Ensure psiblast ran and BLAST_DB is correctly set."
            )

        with open(pssm_path, 'r') as fin:
            fin_data = fin.readlines()
            pssm_begin_line = 3
            pssm_end_line = 0
            for j in range(1, len(fin_data)):
                if fin_data[j] == '\n':
                    pssm_end_line = j
                    break

            raw_pssm = np.zeros((pssm_end_line - pssm_begin_line, 20))
            for j in range(pssm_begin_line, pssm_end_line):
                raw_pssm[j - pssm_begin_line] = [float(x) for x in fin_data[j].split()[2:22]]

            # Sigmoid normalization to [0,1]
            raw_pssm = 1 / (1 + np.exp(-raw_pssm))

            # Sliding-window mean
            window_pssm = []
            half_window = window_size // 2
            for j in range(len(seq)):
                start = max(0, j - half_window)
                end = min(len(seq), j + half_window + 1)
                window = raw_pssm[start:end]
                window_mean = np.mean(window, axis=0)
                window_pssm.append(window_mean)

            window_pssm = np.array(window_pssm)

            if len(window_pssm) == len(seq):
                pdb_ids.append(pdb_name)
                pssms.append(window_pssm)
                logger.info("Collected sliding-window features for %s (%d residues)", pdb_name, len(seq))
            else:
                raise ValueError(f"PSSM length mismatch for {pdb_name}: seq={len(seq)} vs pssm={len(window_pssm)}")

    features_path = os.path.join(pssmdir, f'{pdbid}_pssm_features.pkl')
    # save pssm features
    with open(features_path, 'wb') as f:
        pickle.dump({"pdb_name": pdb_ids, "pssms": pssms}, f)
    logger.info("Saved %d PSSM feature entries to %s", len(pssms), features_path)
