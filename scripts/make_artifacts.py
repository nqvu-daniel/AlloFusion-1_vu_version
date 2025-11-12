#!/usr/bin/env python3
"""
Package smoke-test artifacts for a given PDB/chain into a single tarball.

Usage:
  python scripts/make_artifacts.py --pdb-id 4zsi --chain B \
      --case-dir "AlloFusion-main/Case Study" \
      --out smoke_4zsi.tgz

It collects:
  - dataset.pkl, allosteric_residues.txt, allosteric_sites.pml
  - optional: *_pssm_features.pkl, *_bio_features.pkl
  - env snapshot: conda explicit (if available), versions.txt, blastdb_info.txt (if $BLAST_DB)
  - shapes.json summarizing sizes/shapes
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import tarfile
import tempfile


def exists(p):
    return os.path.exists(p)


def main():
    ap = argparse.ArgumentParser(description="Make smoke-test artifact tarball")
    ap.add_argument("--pdb-id", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--case-dir", default=os.path.join("AlloFusion-main", "Case Study"))
    ap.add_argument("--out", default="smoke_artifacts.tgz")
    args = ap.parse_args()

    case = args.case_dir
    pdb = args.pdb_id
    chain = args.chain
    items = []

    # Required outputs
    items.append(os.path.join(case, f"{pdb}_{chain}_dataset.pkl"))
    items.append(os.path.join(case, f"{pdb}_allosteric_residues.txt"))
    items.append(os.path.join(case, f"{pdb}_allosteric_sites.pml"))

    # Optional feature files
    opt = [
        os.path.join(case, f"{pdb}_pssm_features.pkl"),
        os.path.join(case, f"{pdb}_bio_features.pkl"),
    ]
    items.extend([p for p in opt if exists(p)])

    # Staging dir
    with tempfile.TemporaryDirectory() as td:
        stage = os.path.join(td, "artifacts")
        os.makedirs(stage, exist_ok=True)
        for p in items:
            if exists(p):
                shutil.copy2(p, stage)

        # Env snapshots
        # Conda explicit
        try:
            txt = subprocess.check_output(["conda", "list", "--explicit"], text=True)
            with open(os.path.join(stage, "conda-explicit.txt"), "w") as f:
                f.write(txt)
        except Exception:
            pass
        # BLAST DB info
        blast_db = os.environ.get("BLAST_DB")
        if blast_db:
            try:
                info = subprocess.check_output(["blastdbcmd", "-db", blast_db, "-info"], text=True)
                with open(os.path.join(stage, "blastdb_info.txt"), "w") as f:
                    f.write(info)
            except Exception:
                pass
        # Versions
        versions = {}
        for m in ["torch", "transformers", "prody", "numba", "numpy", "scipy", "networkx", "tensorflow"]:
            try:
                versions[m] = __import__(m).__version__
            except Exception:
                versions[m] = "n/a"
        with open(os.path.join(stage, "versions.txt"), "w") as f:
            for k, v in versions.items():
                f.write(f"{k}={v}\n")

        # Shapes summary
        summary = {}
        ds = os.path.join(stage, f"{pdb}_{chain}_dataset.pkl")
        if exists(ds):
            try:
                d = pickle.load(open(ds, "rb"))
                summary["dataset"] = {
                    "n": len(d.get("features", [])),
                    "feat_dim": len(d["features"][0]) if d.get("features") else 0,
                }
            except Exception:
                pass
        with open(os.path.join(stage, "shapes.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Create tarball
        with tarfile.open(args.out, "w:gz") as tar:
            tar.add(stage, arcname=os.path.basename(stage))
        print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()

