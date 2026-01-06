# -*- coding: utf-8 -*-
"""

@author: hjb
"""
import warnings
warnings.filterwarnings('ignore')

import argparse
from utils.sequence_indices import sequence_indices
from utils.extract_sequence import extract_sequence
import os
import pandas as pd
import numpy as np
import pickle
from utils.download_pdb import download_pdb
from prediced import predict_allosteric_sites
from utils.embedding import get_embedding
from utils.pssm import getPSSM
from utils.bio import getBio
parser = argparse.ArgumentParser(description='Extract sequence, build features, and predict allosteric sites')
# Main parameters
parser.add_argument("--PDBID", type=str, help="Four-character PDB ID (no .pdb required)", required=True)
parser.add_argument("--CHAIN", type=str, help="Protein chain (single letter)", required=True)
# Optional: override CNN weights location
parser.add_argument(
    "--weights",
    type=str,
    default=None,
    help="Path to CNN weights .h5 file (e.g., myModel/final_model.h5) or a directory containing trial1.h5 (legacy all.h5)",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=None,
    help="Decision threshold for classifying residues (default 0.47)",
)
# Optional topology augmentation (non-invasive; disabled by default)
parser.add_argument(
    "--use-dci-betweenness",
    action="store_true",
    help="Append DCI (2D) + betweenness (1D) features to baseline (total 1050D). Requires matching CNN weights.",
)
parser.add_argument("--cutoff", type=float, default=8.0, help="Contact graph cutoff in Å (for topology)")
parser.add_argument("--sigma", type=float, default=4.0, help="Gaussian sigma for edge weighting in Å (for topology)")
parser.add_argument("--tag", type=str, default=None, help="Optional suffix to copy outputs for comparison")
parser.add_argument(
    "--use-topo-extended",
    action="store_true",
    help=(
        "Append an extended topology bundle: DCI (2D) + MSF (1D) + current-flow betweenness (1D) "
        "+ current-flow closeness (1D) + Laplacian positional encoding (k dims). "
        "Requires matching CNN weights."
    ),
)
parser.add_argument("--lap-pe-k", type=int, default=8, help="k dims for Laplacian positional encoding")
parser.add_argument(
    "--lap-pe-normalize",
    action="store_true",
    help="Z-score normalize each Laplacian PE dimension per protein",
)
parser.add_argument(
    "--use-stingallo",
    action="store_true",
    help=(
        "Append StingAllo-inspired 11D structural features per residue "
        "(centroid distance, density, hydrophobic ratio, secondary structure, RSA, B-factor, graph centralities). "
        "Requires matching CNN weights."
    ),
)

def extract_seq(pdb_id,chain_id):
    base_dir = os.path.dirname(__file__)
    case_dir = os.path.join(base_dir, 'Case Study')
    os.makedirs(case_dir, exist_ok=True)
    pdb_path = os.path.join(case_dir, f'{pdb_id}.pdb')
    if not os.path.exists(pdb_path):
        download_pdb(pdb_id, case_dir)
    seq = extract_sequence(pdb_path, chain_id)
    seq_indices = sequence_indices(pdb_id, chain_id)
    return seq, seq_indices


if __name__ == "__main__":
    args = parser.parse_args()
    pdb_id = args.PDBID
    chain_id = args.CHAIN

    if args.use_dci_betweenness and args.use_topo_extended:
        raise ValueError("Use only one of --use-dci-betweenness or --use-topo-extended.")

    # If custom weights provided, expose to predictor via env
    if args.weights:
        w = os.path.abspath(args.weights)
        if os.path.isfile(w):
            os.environ["ALLOFUSION_CNN_WEIGHTS_FILE"] = w
            print(f"[INFO] Using CNN weights file: {w}")
        elif os.path.isdir(w):
            os.environ["ALLOFUSION_CNN_WEIGHTS_DIR"] = w
            print(f"[INFO] Using CNN weights dir: {w}")
        else:
            raise FileNotFoundError(f"--weights path not found: {w}")

    # Optional threshold override
    if args.threshold is not None:
        os.environ["ALLOFUSION_THRESHOLD"] = str(args.threshold)
        print(f"[INFO] Using decision threshold: {args.threshold}")

    # extract sequence from PDB file
    seq,seq_indices=extract_seq(pdb_id, chain_id)
    seq_indices= '" ' + ', '.join([f'{key}' for key in seq_indices.keys()]) + ' " '
    id='>'+pdb_id+'_'+chain_id
    case_dir = os.path.join(os.path.dirname(__file__), 'Case Study')
    os.makedirs(case_dir, exist_ok=True)
    csv_file=os.path.join(case_dir, f'{pdb_id}.csv')
    with open(csv_file, 'w') as f:
        f.write(f'{id},{seq.strip()},{seq_indices}\n')

    # 1-1 （Embedding）
    get_embedding(pdb_id)
    print('Embedding done!')
    # 1-2 （pssm）
    getPSSM('Case Study', pdb_id)
    print("PSSM done!")
    # #1-3 （bio）
    getBio('Case Study', pdb_id, chain_id)
    print("Bio done!")
    # load embeddings
    with open(os.path.join(case_dir, f'{pdb_id}_T5.pkl'), 'rb') as f:
        embeddings = pickle.load(f)
  
    # read csv file
    df = pd.read_csv(os.path.join(case_dir, f'{pdb_id}.csv'), header=None)

    seqs=df[1].tolist()
    pdb_ids=df[0].tolist()
    # Handle prediction mode: labels column may contain residue indices, not AFR labels
    raw_label_str = df[2].tolist()[0] if len(df.columns) > 2 else ""
    raw_labels = raw_label_str.strip().strip('"').split(',')
    seq_len = len(seqs[0].strip())
    # If labels count matches sequence length and are 0/1, use them; otherwise use dummy
    labels = raw_labels if len(raw_labels) == seq_len else ['0'] * seq_len
    pdb_names=[]
    residues=[]
    res_features=[]
    res_lables=[]
    # load Z-scores
    with open(os.path.join(case_dir, f'{pdb_id}_bio_features.pkl'), 'rb') as f:
        bio=pickle.load(f)

    # load pssm_features
    with open(os.path.join(case_dir, f'{pdb_id}_pssm_features.pkl'), 'rb') as f:
        pssm_features = pickle.load(f)
    pdb_names1 = pssm_features['pdb_name']
    pssm_features = pssm_features['pssms']

    i=0
    pdb_ids[i]=pdb_ids[i].split('>')[1].strip()
    seqs[i]=seqs[i].strip()

    # Optional topology features (DCI + betweenness)
    topo_dci = None
    topo_betw = None
    topo_msf = None
    topo_cfclose = None
    topo_lappe = None
    stingallo = None
    if args.use_dci_betweenness or args.use_topo_extended:
        try:
            from tda_adapter.dci import compute_and_cache_dci, compute_and_cache_msf
            from utils.graph_features import (
                compute_betweenness_centrality,
                compute_current_flow_closeness_centrality,
                compute_laplacian_positional_encoding,
            )
            pdb_path = os.path.join(case_dir, f'{pdb_id}.pdb')
            # DCI returns (N,2) in CA order; align/pad to sequence length
            dci_raw = compute_and_cache_dci(pdb_path, chain_id, cache_dir=case_dir, cutoff=args.cutoff)
            msf_raw = None
            if args.use_topo_extended:
                msf_raw = compute_and_cache_msf(pdb_path, chain_id, cache_dir=case_dir, cutoff=args.cutoff, normalize=True)
            # Graph metrics attempt to align to sequence indices when mapping available
            betw_raw = compute_betweenness_centrality(pdb_path, chain_id, cutoff=args.cutoff, sigma=args.sigma, normalize=True)
            L = len(seqs[i])
            topo_dci = np.zeros((L, 2), dtype=float)
            if dci_raw is not None and dci_raw.ndim == 2 and dci_raw.shape[1] == 2:
                m = min(L, dci_raw.shape[0])
                topo_dci[:m] = dci_raw[:m]
            topo_msf = None
            if args.use_topo_extended:
                topo_msf = np.zeros(L, dtype=float)
                if msf_raw is not None and msf_raw.ndim == 1:
                    m0 = min(L, msf_raw.shape[0])
                    topo_msf[:m0] = msf_raw[:m0]
            topo_betw = np.zeros(L, dtype=float)
            if betw_raw is not None and betw_raw.ndim == 1:
                m2 = min(L, betw_raw.shape[0])
                topo_betw[:m2] = betw_raw[:m2]
            if args.use_topo_extended:
                cfclose_raw = compute_current_flow_closeness_centrality(
                    pdb_path, chain_id, cutoff=args.cutoff, sigma=args.sigma, normalize=True
                )
                topo_cfclose = np.zeros(L, dtype=float)
                if cfclose_raw is not None and cfclose_raw.ndim == 1:
                    m3 = min(L, cfclose_raw.shape[0])
                    topo_cfclose[:m3] = cfclose_raw[:m3]
                k = max(0, int(args.lap_pe_k))
                lappe_raw = compute_laplacian_positional_encoding(
                    pdb_path,
                    chain_id,
                    k=k,
                    cutoff=args.cutoff,
                    sigma=args.sigma,
                    normalize=bool(args.lap_pe_normalize),
                )
                topo_lappe = np.zeros((L, k), dtype=float)
                if lappe_raw is not None and lappe_raw.ndim == 2 and lappe_raw.shape[1] == k:
                    m4 = min(L, lappe_raw.shape[0])
                    topo_lappe[:m4] = lappe_raw[:m4]
                print(
                    f"[INFO] Extended topology ready: DCI={topo_dci.shape}, MSF={topo_msf.shape}, "
                    f"CFbetw={topo_betw.shape}, CFclose={topo_cfclose.shape}, LapPE={topo_lappe.shape}"
                )
                print("[INFO] Ensure your CNN weights match 1060D features when using --use-topo-extended.")
            else:
                print(f"[INFO] Topology features ready: DCI shape={topo_dci.shape}, Betweenness len={topo_betw.shape[0]}")
                print("[INFO] Ensure your CNN weights match 1050D features when using --use-dci-betweenness.")
        except Exception as e:
            print(f"[WARN] Topology feature computation failed: {e}. Proceeding with zero-padded augmentation.")
            L = len(seqs[i])
            if topo_dci is None:
                topo_dci = np.zeros((L, 2), dtype=float)
            if topo_betw is None:
                topo_betw = np.zeros(L, dtype=float)
            if args.use_topo_extended:
                if topo_msf is None:
                    topo_msf = np.zeros(L, dtype=float)
                if topo_cfclose is None:
                    topo_cfclose = np.zeros(L, dtype=float)
                k = max(0, int(args.lap_pe_k))
                if topo_lappe is None:
                    topo_lappe = np.zeros((L, k), dtype=float)

    if args.use_stingallo:
        try:
            from utils.stingallo_features import compute_stingallo_features

            pdb_path = os.path.join(case_dir, f"{pdb_id}.pdb")
            L = len(seqs[i])
            stingallo_raw = compute_stingallo_features(pdb_path, chain_id, seq_length=L, cutoff=args.cutoff)
            stingallo = np.zeros((L, 11), dtype=float)
            if (
                stingallo_raw is not None
                and isinstance(stingallo_raw, np.ndarray)
                and stingallo_raw.ndim == 2
                and stingallo_raw.shape[1] == 11
            ):
                m = min(L, stingallo_raw.shape[0])
                stingallo[:m] = stingallo_raw[:m]
            print(f"[INFO] StingAllo features ready: {stingallo.shape}")
            print("[INFO] Ensure your CNN weights match augmented feature dims when using --use-stingallo.")
        except Exception as e:
            print(f"[WARN] StingAllo feature computation failed: {e}. Proceeding with zero-padded augmentation.")
            L = len(seqs[i])
            if stingallo is None:
                stingallo = np.zeros((L, 11), dtype=float)

    for j in range(len(seqs[i])):
        pdb_names.append(pdb_ids[i])
        residues.append(seqs[i][j])
        feat_vec = np.concatenate((embeddings[i][j], pssm_features[i][j], bio[i][j]))
        if args.use_topo_extended:
            feat_vec = np.concatenate((feat_vec, topo_dci[j], [topo_msf[j]], [topo_betw[j]], [topo_cfclose[j]], topo_lappe[j]))
        elif topo_dci is not None and topo_betw is not None and args.use_dci_betweenness:
            feat_vec = np.concatenate((feat_vec, topo_dci[j], [topo_betw[j]]))
        if args.use_stingallo and stingallo is not None:
            feat_vec = np.concatenate((feat_vec, stingallo[j]))
        res_features.append(feat_vec)
        res_lables.append(labels[j])

    dataset_path = os.path.join(case_dir, f'{pdb_id}_{chain_id}_dataset.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump({"pdb_name":pdb_names,"residue":residues,"features":res_features,"label":res_lables},f)

    # Optional: write a tagged copy alongside the default dataset for side-by-side runs
    if args.tag:
        tagged = os.path.join(case_dir, f'{pdb_id}_{chain_id}_dataset.{args.tag}.pkl')
        try:
            with open(tagged, 'wb') as tf:
                pickle.dump({"pdb_name":pdb_names,"residue":residues,"features":res_features,"label":res_lables}, tf)
            print(f"[INFO] Wrote tagged dataset copy: {tagged}")
        except Exception as e:
            print(f"[WARN] Failed to write tagged dataset copy: {e}")

    #2 predict allosteric sites
    print(f"[DEBUG] Starting prediction for {pdb_id} chain {chain_id}...")
    predict_allosteric_sites(pdb_id, chain_id)
    print(f"[DEBUG] Prediction completed!")
    
