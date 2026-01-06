"""
Dynamic Coupling Index (DCI) computation (optional topology features).

Cloned from ALLO_TOPO_PLM/AlloFusion-main/tda_adapter/dci.py
(reason: add opt-in DCI features to Vu pipeline without changing vendor code).
"""
import json
import os
import time
from typing import Tuple, Optional

import numpy as np


def _float_close(a: object, b: object, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _cache_params_match(meta: Optional[dict], expected: dict) -> bool:
    if not meta:
        return True  # preserve legacy behavior when no metadata exists
    for k, v in expected.items():
        if k not in meta:
            return False
        mv = meta.get(k)
        if isinstance(v, float) or isinstance(mv, float):
            if not _float_close(mv, v):
                return False
        else:
            if mv != v:
                return False
    return True


def _extract_calpha_coords(pdb_file: str, chain_id: str) -> np.ndarray:
    # Prefer ProDy when available, but fall back to a lightweight parser to
    # avoid requiring compiled dependencies for topology features.
    try:
        from prody import parsePDB  # type: ignore

        ag = parsePDB(pdb_file)
        if ag is None:
            raise ValueError(f"Failed to parse PDB file: {pdb_file}")

        sel_str = f"protein and chain {chain_id} and name CA"
        sel = ag.select(sel_str)
        if sel is None or sel.numAtoms() == 0:
            raise ValueError(
                f"No C-alpha atoms found for chain {chain_id} in {pdb_file}"
            )

        coords = sel.getCoords().astype(np.float64)
        return coords
    except Exception:
        coords_by_resno = {}
        with open(pdb_file, "r") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                if line[21].strip() != chain_id:
                    continue
                if line[12:16].strip() != "CA":
                    continue
                resno = line[22:26].strip()
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                coords_by_resno.setdefault(resno, []).append([x, y, z])

        def _key(r: str):
            try:
                return (0, int(r))
            except Exception:
                return (1, r)

        resnos = sorted(coords_by_resno.keys(), key=_key)
        if not resnos:
            raise ValueError(f"No C-alpha atoms found for chain {chain_id} in {pdb_file}")
        coords = np.array(
            [np.mean(np.asarray(coords_by_resno[r], dtype=np.float64), axis=0) for r in resnos],
            dtype=np.float64,
        )
        return coords


def _build_anm_hessian(coords: np.ndarray, cutoff: float = 8.0, gamma: float = 1.0) -> np.ndarray:
    """
    Build ANM Hessian matrix (3N×3N) with 3×3 blocks for directional spring coupling.
    H_ij = -gamma * (r_ij ⊗ r_ij) / |r_ij|^2  for i≠j within cutoff
    H_ii = -Σ_{j≠i} H_ij
    """
    N = coords.shape[0]
    H = np.zeros((3*N, 3*N), dtype=np.float64)

    for i in range(N):
        for j in range(i+1, N):
            diff = coords[j] - coords[i]
            dist_sq = np.dot(diff, diff)
            dist = np.sqrt(dist_sq)

            if dist <= cutoff and dist > 0.0:
                # 3×3 block: -gamma * (r ⊗ r) / |r|^2
                outer = np.outer(diff, diff) / dist_sq
                block = -gamma * outer

                # Off-diagonal blocks
                H[3*i:3*i+3, 3*j:3*j+3] = block
                H[3*j:3*j+3, 3*i:3*i+3] = block

                # Accumulate diagonal blocks
                H[3*i:3*i+3, 3*i:3*i+3] -= block
                H[3*j:3*j+3, 3*j:3*j+3] -= block

    return H


def _remove_rigid_modes(H: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove 6 rigid-body modes (3 translation + 3 rotation) from ANM Hessian.
    Returns eigenvalues and eigenvectors with rigid modes removed.
    """
    N = coords.shape[0]

    # Compute eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(H)

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Remove first 6 modes (should be near-zero eigenvalues for rigid-body motion)
    # Keep modes 7 onwards (non-rigid modes)
    n_remove = 6
    eigvals_nonrigid = eigvals[n_remove:]
    eigvecs_nonrigid = eigvecs[:, n_remove:]

    return eigvals_nonrigid, eigvecs_nonrigid


def _compute_green_function_anm(eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """
    Compute Green's function from non-rigid modes: G = Σ_k (1/λ_k) * v_k ⊗ v_k
    Returns 3N×3N Green's function matrix.
    """
    # Filter out near-zero or negative eigenvalues
    valid_mask = eigvals > 1e-8
    eigvals_valid = eigvals[valid_mask]
    eigvecs_valid = eigvecs[:, valid_mask]

    # G = V * diag(1/λ) * V^T
    inv_eigvals = 1.0 / eigvals_valid
    # Some structures can yield very small non-rigid eigenvalues, which may
    # trigger harmless NumPy warnings during the matrix multiplications.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            G = eigvecs_valid @ np.diag(inv_eigvals) @ eigvecs_valid.T

    return G


def _compute_msf_from_green(G: np.ndarray) -> np.ndarray:
    """
    Mean-square fluctuation (MSF) per residue from the Green's function.
    MSF(i) = trace(G_ii)/3 where G_ii is the 3x3 block for residue i.
    """
    N = G.shape[0] // 3
    if G.shape != (3 * N, 3 * N):
        raise ValueError(f"Expected G shape {(3*N, 3*N)}, got {G.shape}")
    G_blocks = G.reshape(N, 3, N, 3)
    diag_blocks = G_blocks[np.arange(N), :, np.arange(N), :]  # (N,3,3)
    msf = np.einsum("nii->n", diag_blocks, optimize=True) / 3.0
    return msf.astype(np.float64)


def _compute_dci_from_green(
    G: np.ndarray,
    coords: np.ndarray,
    mode: str = "range",
    distance_threshold: float = 8.0,
    subset_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two per-residue dynamic descriptors from Green's function using PRS blocks.

    Common:
      PRS_ij = trace(G_ji^T G_ji)/3 with G 3N×3N.

    Modes (select exactly one):
      - "isotropic":
          eff(i) = Σ_j PRS_ij
          sens(i) = Σ_j PRS_ji  (identical to eff due to symmetry)
      - "msf":
          eff(i) = Σ_j PRS_ij
          sens(i) = MSF(i) = trace(G_ii)/3
      - "range" (default): split by distance using Cα distances d_ij
          eff(i)  = Σ_{j: d_ij > T} PRS_ij       (far-outgoing coupling)
          sens(i) = Σ_{j: d_ij ≤ T, j≠i} PRS_ji  (near-incoming coupling)
      - "subset": use a boolean mask F over residues (length N)
          eff(i)  = Σ_{j∈F} PRS_ij  (influence from i to the functional subset F)
          sens(i) = Σ_{j∈F} PRS_ji  (sensitivity of i to the subset F)
    """
    N = G.shape[0] // 3
    dci_eff = np.zeros(N, dtype=np.float64)
    dci_sens = np.zeros(N, dtype=np.float64)

    # Validate subset mask if used
    if mode == "subset":
        if subset_mask is None or subset_mask.shape != (N,):
            raise ValueError("subset mode requires subset_mask of shape (N,)")
        subset_idx = np.where(subset_mask)[0]

    # Reshape into residue blocks for vectorized PRS:
    # G_blocks[j, :, i, :] = 3x3 block G_ji
    if G.shape != (3 * N, 3 * N):
        raise ValueError(f"Expected G shape {(3*N, 3*N)}, got {G.shape}")
    G_blocks = G.reshape(N, 3, N, 3)

    # PRS_ij = ||G_ji||_F^2 / 3  (rows=j respond, cols=i perturb)
    prs = np.einsum("jcia,jcia->ji", G_blocks, G_blocks, optimize=True) / 3.0
    np.fill_diagonal(prs, 0.0)

    if mode == "isotropic":
        dci_eff = prs.sum(axis=0)
        dci_sens = prs.sum(axis=1)
        return dci_eff, dci_sens

    if mode == "msf":
        dci_eff = prs.sum(axis=0)
        diag_blocks = G_blocks[np.arange(N), :, np.arange(N), :]  # (N,3,3)
        dci_sens = np.einsum("nii->n", diag_blocks, optimize=True) / 3.0
        return dci_eff, dci_sens

    if mode == "range":
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        D = np.sqrt(np.maximum(d2, 0.0))
        nonself = ~np.eye(N, dtype=bool)
        far_prs_mask = (D > distance_threshold).T & nonself  # align to prs[j,i]
        near_prs_mask = (D <= distance_threshold).T & nonself

        dci_eff = (prs * far_prs_mask).sum(axis=0)   # sum over responders j
        dci_sens = (prs * near_prs_mask).sum(axis=1)  # sum over perturbations i
        return dci_eff, dci_sens

    if mode == "subset":
        # Align with docstring:
        # eff(i) = Σ_{j∈F} PRS_ij  -> sum over responders j in F for each perturbation i (cols)
        # sens(i) = Σ_{j∈F} PRS_ji -> sum over perturbations j in F for each responder i (rows)
        dci_eff = prs[subset_idx, :].sum(axis=0)
        dci_sens = prs[:, subset_idx].sum(axis=1)
        return dci_eff, dci_sens

    raise ValueError(f"Unknown DCI mode: {mode}")


def _normalize_features(dci_eff: np.ndarray, dci_sens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def zscore(x):
        mu = x.mean()
        sigma = x.std()
        if sigma < 1e-10:
            return np.zeros_like(x)
        return (x - mu) / sigma
    return zscore(dci_eff), zscore(dci_sens)


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma < 1e-10:
        return np.zeros_like(x)
    y = (x - mu) / sigma
    y[~np.isfinite(y)] = 0.0
    return y


def compute_dci_features(
    pdb_file: str,
    chain_id: str,
    cutoff: float = 8.0,
    gamma: float = 1.0,
    normalize: bool = True,
    mode: str = "range",
    distance_threshold: float = 8.0,
    subset_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute ANM-based DCI features using proper Hessian (3N×3N), rigid mode removal,
    and PRS formula: trace(G_ji^T G_ji)/3.
    """
    coords = _extract_calpha_coords(pdb_file, chain_id)
    N = coords.shape[0]

    # Build ANM Hessian (3N×3N)
    H = _build_anm_hessian(coords, cutoff=cutoff, gamma=gamma)

    # Remove 6 rigid-body modes
    eigvals, eigvecs = _remove_rigid_modes(H, coords)

    # Compute Green's function from non-rigid modes
    G = _compute_green_function_anm(eigvals, eigvecs)

    # Compute DCI-based descriptors
    dci_eff, dci_sens = _compute_dci_from_green(
        G, coords, mode=mode, distance_threshold=distance_threshold, subset_mask=subset_mask
    )

    if normalize:
        dci_eff, dci_sens = _normalize_features(dci_eff, dci_sens)

    dci_features = np.stack([dci_eff, dci_sens], axis=1)
    assert dci_features.shape == (N, 2)
    assert np.all(np.isfinite(dci_features))
    return dci_features


def compute_msf_features(
    pdb_file: str,
    chain_id: str,
    cutoff: float = 8.0,
    gamma: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute ANM/GNM-style mean-square fluctuation (MSF) per residue from Green's function.
    Returns (N,) in Cα order for the selected chain.
    """
    coords = _extract_calpha_coords(pdb_file, chain_id)
    N = coords.shape[0]
    H = _build_anm_hessian(coords, cutoff=cutoff, gamma=gamma)
    eigvals, eigvecs = _remove_rigid_modes(H, coords)
    G = _compute_green_function_anm(eigvals, eigvecs)
    msf = _compute_msf_from_green(G)
    assert msf.shape == (N,)
    if normalize:
        msf = _zscore_1d(msf)
    assert np.all(np.isfinite(msf))
    return msf


def compute_and_cache_dci(
    pdb_file: str,
    chain_id: str,
    cache_dir: str,
    cutoff: float = 8.0,
    gamma: float = 1.0,
    mode: str = "range",
    distance_threshold: float = 8.0,
    subset_mask: Optional[np.ndarray] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]
    cache_npz = os.path.join(cache_dir, f"{pdb_base}_{chain_id}_dci.npz")
    cache_meta = os.path.join(cache_dir, f"{pdb_base}_{chain_id}_dci_meta.json")

    if not force_recompute and os.path.exists(cache_npz):
        meta = _load_json(cache_meta)
        if _cache_params_match(
            meta,
            {
                "cutoff": float(cutoff),
                "gamma": float(gamma),
                "mode": mode,
                "distance_threshold": float(distance_threshold),
            },
        ):
            data = np.load(cache_npz)
            dci_features = data.get("dci_features")
            if (
                dci_features is not None
                and isinstance(dci_features, np.ndarray)
                and dci_features.ndim == 2
                and dci_features.shape[1] == 2
            ):
                print(f"[DCI] Loaded from cache: {cache_npz}")
                return dci_features
        else:
            print(f"[DCI] Cache params mismatch; recomputing: {cache_npz}")

    print(f"[DCI] Computing ANM-based DCI for {pdb_base}_{chain_id}...")
    start_time = time.time()
    dci_features = compute_dci_features(
        pdb_file,
        chain_id,
        cutoff=cutoff,
        gamma=gamma,
        mode=mode,
        distance_threshold=distance_threshold,
        subset_mask=subset_mask,
    )
    elapsed = time.time() - start_time

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_npz, dci_features=dci_features)

    metadata = {
        "pdb_file": pdb_file,
        "chain_id": chain_id,
        "method": "ANM",
        "cutoff": cutoff,
        "gamma": gamma,
        "rigid_modes_removed": 6,
        "prs_formula": "trace(G_ji^T G_ji)/3",
        "mode": mode,
        "distance_threshold": distance_threshold,
        "n_residues": dci_features.shape[0],
        "feature_dims": 2,
        "feature_names": [
            "DCI_effector_far" if mode == "range" else "DCI_effector",
            (
                "DCI_sensor_near" if mode == "range"
                else ("MSF" if mode == "msf" else "DCI_sensor")
            ),
        ],
        "computation_time_sec": elapsed,
        "variance": {
            "DCI_effector": float(dci_features[:, 0].std()),
            "DCI_sensor": float(dci_features[:, 1].std()),
        },
    }
    with open(cache_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DCI] Computed in {elapsed:.2f}s, cached to {cache_npz}")
    return dci_features


def compute_and_cache_msf(
    pdb_file: str,
    chain_id: str,
    cache_dir: str,
    cutoff: float = 8.0,
    gamma: float = 1.0,
    normalize: bool = True,
    force_recompute: bool = False,
) -> np.ndarray:
    pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]
    cache_npz = os.path.join(cache_dir, f"{pdb_base}_{chain_id}_msf.npz")
    cache_meta = os.path.join(cache_dir, f"{pdb_base}_{chain_id}_msf_meta.json")

    if not force_recompute and os.path.exists(cache_npz):
        meta = _load_json(cache_meta)
        if _cache_params_match(
            meta,
            {
                "cutoff": float(cutoff),
                "gamma": float(gamma),
                "normalize": bool(normalize),
            },
        ):
            data = np.load(cache_npz)
            msf = data.get("msf")
            if (
                msf is not None
                and isinstance(msf, np.ndarray)
                and msf.ndim == 1
                and np.all(np.isfinite(msf))
            ):
                print(f"[MSF] Loaded from cache: {cache_npz}")
                return msf
        else:
            print(f"[MSF] Cache params mismatch; recomputing: {cache_npz}")

    print(f"[MSF] Computing ANM-based MSF for {pdb_base}_{chain_id}...")
    start_time = time.time()
    msf = compute_msf_features(
        pdb_file,
        chain_id,
        cutoff=cutoff,
        gamma=gamma,
        normalize=normalize,
    )
    elapsed = time.time() - start_time

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_npz, msf=msf)

    metadata = {
        "pdb_file": pdb_file,
        "chain_id": chain_id,
        "method": "ANM",
        "cutoff": cutoff,
        "gamma": gamma,
        "rigid_modes_removed": 6,
        "msf_formula": "trace(G_ii)/3",
        "n_residues": int(msf.shape[0]),
        "feature_dims": 1,
        "feature_names": ["MSF"],
        "normalize": bool(normalize),
        "computation_time_sec": elapsed,
        "variance": float(np.std(msf)),
    }
    with open(cache_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[MSF] Computed in {elapsed:.2f}s, cached to {cache_npz}")
    return msf
