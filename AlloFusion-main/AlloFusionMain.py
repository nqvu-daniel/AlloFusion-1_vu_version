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
    labels=df[2].tolist()[0].split(',')
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
    if args.use_dci_betweenness:
        try:
            from tda_adapter.dci import compute_and_cache_dci
            from utils.graph_features import compute_betweenness_centrality
            pdb_path = os.path.join(case_dir, f'{pdb_id}.pdb')
            # DCI returns (N,2) in CA order; align/pad to sequence length
            dci_raw = compute_and_cache_dci(pdb_path, chain_id, cache_dir=case_dir, cutoff=args.cutoff)
            # Betweenness attempts to align to sequence indices when mapping available
            betw_raw = compute_betweenness_centrality(pdb_path, chain_id, cutoff=args.cutoff, sigma=args.sigma, normalize=True)
            L = len(seqs[i])
            topo_dci = np.zeros((L, 2), dtype=float)
            if dci_raw is not None and dci_raw.ndim == 2 and dci_raw.shape[1] == 2:
                m = min(L, dci_raw.shape[0])
                topo_dci[:m] = dci_raw[:m]
            topo_betw = np.zeros(L, dtype=float)
            if betw_raw is not None and betw_raw.ndim == 1:
                m2 = min(L, betw_raw.shape[0])
                topo_betw[:m2] = betw_raw[:m2]
            print(f"[INFO] Topology features ready: DCI shape={topo_dci.shape}, Betweenness len={topo_betw.shape[0]}")
            print("[INFO] Ensure your CNN weights match 1050D features when using --use-dci-betweenness.")
        except Exception as e:
            print(f"[WARN] Topology feature computation failed: {e}. Proceeding without augmentation.")
            topo_dci, topo_betw = None, None

    for j in range(len(seqs[i])):
        pdb_names.append(pdb_ids[i])
        residues.append(seqs[i][j])
        feat_vec = np.concatenate((embeddings[i][j], pssm_features[i][j], bio[i][j]))
        if topo_dci is not None and topo_betw is not None:
            feat_vec = np.concatenate((feat_vec, topo_dci[j], [topo_betw[j]]))
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
    
