import argparse
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test AlloFusion feature modules on a local PDB.")
    ap.add_argument("--pdb", required=True, help="Path to a local PDB file")
    ap.add_argument("--chain", required=True, help="Chain ID (e.g., A)")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--lap-pe-k", type=int, default=8)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    pdb = os.path.abspath(args.pdb)
    if not os.path.exists(pdb):
        raise FileNotFoundError(pdb)

    from tda_adapter.dci import compute_dci_features, compute_msf_features
    from utils.graph_features import (
        compute_betweenness_centrality,
        compute_current_flow_closeness_centrality,
        compute_laplacian_positional_encoding,
    )
    from utils.stingallo_features import compute_stingallo_features

    dci = compute_dci_features(pdb, args.chain, cutoff=args.cutoff)
    msf = compute_msf_features(pdb, args.chain, cutoff=args.cutoff, normalize=True)
    betw = compute_betweenness_centrality(pdb, args.chain, cutoff=args.cutoff, sigma=args.sigma, normalize=True)
    cfcl = compute_current_flow_closeness_centrality(pdb, args.chain, cutoff=args.cutoff, sigma=args.sigma, normalize=True)
    pe = compute_laplacian_positional_encoding(
        pdb, args.chain, k=int(args.lap_pe_k), cutoff=args.cutoff, sigma=args.sigma, normalize=True
    )
    sting = compute_stingallo_features(pdb, args.chain, seq_length=dci.shape[0], cutoff=args.cutoff)

    bundle = np.concatenate([dci, msf[:, None], betw[:, None], cfcl[:, None], pe], axis=1)

    print("PDB:", pdb)
    print("DCI:", dci.shape, "MSF:", msf.shape, "CFbetw:", betw.shape, "CFclose:", cfcl.shape, "LapPE:", pe.shape)
    print("Extended-topology bundle dim (expected 5+k):", bundle.shape[1])
    print("StingAllo:", sting.shape, "(expected Nx11)")
    print("All finite:", all(np.isfinite(x).all() for x in (dci, msf, betw, cfcl, pe, sting)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

