# **AlloFusion: Allosteric Site Prediction Based on Language Models and Multi-Feature Fusion**

The AlloFusion program is a method for protein allosteric site prediction. 



## Requirement
- python 3.10+

- numpy  1.26.4

- pandas  2.2.3

- joblib  1.4.0

- ProDy  2.4.1 (optional; used for topology parsing if installed)

- torch  2.5.0+cu118

- tensorflow  2.12.0

- transformers  4.42.4
	
- networkx (required for graph-based features)
	
- scipy (required for current-flow betweenness/closeness)

- dssp / mkdssp (recommended for StingAllo RSA + secondary structure; falls back if missing)
	
  
---

## Reproducible setup (macOS Apple Silicon + conda)

This repo includes a ready-to-use conda environment file and a bootstrap script:

1) Create/update the environment:
- `bash scripts/bootstrap_osx_arm64.sh`

2) Activate:
- `conda activate allofusion`

3) Quick smoke test (topology + StingAllo only; no ProtT5/TensorFlow models needed):
- `python scripts/smoke_features.py --pdb 4ZSI_gnm_zs.pdb --chain A`

Notes:
- `mkdssp` comes from `conda-forge::dssp`. If it’s missing, StingAllo will fall back (still fixed dims).
- `psiblast` is provided by NCBI BLAST+. On macOS arm64, the simplest install is Homebrew: `brew install blast`. Then download/configure a BLAST DB for PSSM (see below).
- For full runs (ProtT5 embeddings), download the model locally and set `PROT_T5_PATH`:
  - `python scripts/prefetch_prot_t5.py --repo Rostlab/prot_t5_xl_uniref50 --dest data/models/prot_t5_xl_uniref50`
  - `export PROT_T5_PATH=$PWD/data/models/prot_t5_xl_uniref50`
	
	
## To run the AlloFusion, you need to install the bioinformatics tools and download the corresponding databases.
（1）Download the prot_t5_xl_uniref50 model from the following link:
	 	https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main  

（2）Install blast+ for extracting PSSM(position-specific scoring matrix) profiles

​			To install blast-2.15.0+ and download the SwissProt database (ftp://ftp.ncbi.nlm.nih.gov/blast/db/) for psiblast, please refer to BLAST(https://www.ncbi.nlm.nih.gov/books/NBK52640/).

Automation (Enhanced AlloFusion)
- Download BLAST databases via script (requires BLAST+):
  - SwissProt curated DB: `bash scripts/setup_blast_db.sh --dir /path/to/blastdb --db swissprot`
  - Then set env for AlloFusion PSSM:
    - `export BLAST_DB=/path/to/blastdb/swissprot/swissprot`
    - or `export BLASTDB=/path/to/blastdb` and `export BLAST_DB_NAME=swissprot`
- CNN weights:
  - This repo includes `myModel/trial1_final_model.h5` (baseline weights); you can also point to any `.h5` via `--weights /path/to/model.h5`.

Optional Topology Augmentation (DCI + Betweenness)
- What: Append per-residue DCI (2D) + betweenness (1D) to baseline features (ProtT5 1024D + PSSM 20D + Bio 3D) → 1050D total.
- How: Enable at runtime; defaults preserve baseline behavior.
- Dependencies: `networkx` (+ `scipy` for current-flow); `prody` is optional (a lightweight PDB parser fallback is included). Install via your environment file or `pip install networkx scipy prody`.
- CLI usage:
  - Example (feature build + prediction with matching weights):
    - `python AlloFusionMain.py --PDBID 4ZSI --CHAIN B --use-dci-betweenness --cutoff 8.0 --sigma 4.0 --weights myModel/all_1050d.h5`
  - Tagged dataset copy for comparison:
    - `python AlloFusionMain.py --PDBID 4ZSI --CHAIN B --use-dci-betweenness --tag topo`
- Notes:
  - If you enable `--use-dci-betweenness`, ensure your CNN weights were trained for 1050D inputs; otherwise weight loading will fail.
  - The default run (without the flag) uses 1047D features and baseline weights.

Extended Topology Augmentation (DCI + MSF + current-flow + LapPE)
- What: Append an extended bundle: DCI (2D) + MSF (1D) + current-flow betweenness (1D) + current-flow closeness (1D) + Laplacian positional encoding (k dims).
  - Added dims = `2 + 1 + 1 + 1 + k = (5 + k)` per residue.
  - With default `--lap-pe-k 8`: 1047D + 13D = **1060D** total.
- How: Enable with `--use-topo-extended` (and optionally `--lap-pe-k`, `--lap-pe-normalize`).
- Notes:
  - Feature dims change if you change `--lap-pe-k`; weights must match the exact dim.
  - Sequence-index alignment uses mmCIF download when available; if that fails (e.g., offline), features fall back to Cα order (still fixed dims via padding).

StingAllo-inspired Structural Features (11D)
- What: Append per-residue 11D structural descriptors inspired by STINGAllo (distance to centroid, local density, hydrophobic contact ratio, secondary structure one-hot, RSA, B-factor, graph centralities).
  - Added dims = **+11D** per residue.
  - Baseline 1047D → **1058D** when enabled alone.
  - With `--use-dci-betweenness`: 1050D + 11D → **1061D**.
  - With `--use-topo-extended` and default `--lap-pe-k 8`: 1060D + 11D → **1071D**.
- How: Enable with `--use-stingallo`.
- Notes:
  - Some sub-features may fall back to safe defaults when external tools are missing (e.g., DSSP/STRIDE for secondary structure); dims stay fixed via padding/zeros.

One-command bootstrap
- Mac (ARM64):
  - `bash scripts/bootstrap_osx_arm64.sh`



---
## 🔍 Model Download Instructions

Due to the large size of the trained model file (exceeding GitHub's file size limit), it has been uploaded to the [Releases](https://github.com/hjb-001/AlloFusion/releases) section of this repository.

Please visit the Releases page to download the full model file:

👉 [Click here to download the model from the Releases page](https://github.com/hjb-001/AlloFusion/releases)

After downloading, place the model file in the project root directory or the designated model folder to use it or reproduce the experimental results.

---
## 📁 Dataset Download Instructions

Due to the large size of the dataset (exceeding GitHub's file size limit), it has also been uploaded to the [Releases](https://github.com/hjb-001/AlloFusion/releases) section of this repository.

Please visit the Releases page to download the full dataset:

👉 [Click here to download the dataset from the Releases page](https://github.com/hjb-001/AlloFusion/releases)

After downloading, extract the dataset to the project root directory or the designated data folder to proceed with training or evaluation.


----
## How to run

**Step 1:** Extract protein sequence based on input pdbid and chain

**Step 2:** Residue feature extraction

**Step 3:** Combined residue characterization

**Step 4:** Loading the model for predicting AFRs



---

## Example

An allosteric protein with PDB ID "4ZSI" is used as an example to show the process. This PDB file is 4ZSI.pdb. Only the protein functional chain is preserved.

```python
python AlloFusionMain.py --PDBID [pdbid] --CHAIN [chain]
```

> for example:
```python
python AlloFusionMain.py --PDBID 4ZSI --CHAIN B
```

> The parameter [pdbid] is the PDB file name of the allosteric protein.

> The parameter [chain] is the functional chain of the target protein.

Then, AlloFusion program will perform the feature extraction and prediction process, which will take some time.

The final prediction result is a file containing the residue IDs of AFRs residues:

`4ZSI_allosteric_residues.txt`

```
AlloFusion Allosteric Site Forming Residues:
Residues: ( Chain B and resid 92, 95, 96, 97, 98, 99, 109, 111, 140, 142, 144, 152, 154, 156, 174, 175, 177, 193, 219, 221, 223, 232, 233, 234, 236, 238, 247)
```

and the script for viewing the allosteric sites composed of AFRs in PyMol: `4ZSI_allosteric_sites.pml`.
<img width="388" alt="image" src="https://github.com/user-attachments/assets/11f21e81-3da1-4c9a-9bab-e8f488bc8290" />

