"""
Graph propagation features (centralities) for optional topology augmentation.

Cloned from ALLO_TOPO_PLM/AlloFusion-main/utils/graph_features.py
(reason: add opt-in graph/centrality features to Vu pipeline without editing vendor code).

Provides:
- build_contact_graph: constructs a weighted Cα contact graph for a given chain
- get_graph_features: computes per-residue centralities and aligns to sequence
- compute_betweenness_centrality: convenient betweenness extractor (aligned)
"""
from typing import Dict, List, Tuple
import numpy as np


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.zeros_like(x, dtype=float)
    y = (x - xmin) / (xmax - xmin)
    y[~np.isfinite(y)] = 0.0
    return y


def _parse_calpha_from_pdb(pdb_file: str, chain_id: str) -> Tuple[np.ndarray, List[str]]:
    """
    Extract Cα coordinates and PDB residue numbers (as strings) for one chain.

    Returns
    - coords: (N, 3)
    - resnos: list[str] length N with PDB residue numbers (column 23-26)
    """
    coords: List[List[float]] = []
    resnos: List[str] = []
    acc: Dict[str, List[np.ndarray]] = {}
    with open(pdb_file, 'r') as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            if line[21].strip() != chain_id:
                continue
            if line[12:16].strip() != 'CA':
                continue
            resno = line[22:26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            acc.setdefault(resno, []).append(np.array([x, y, z], dtype=float))
    def _key(r: str):
        try:
            return (0, int(r))
        except Exception:
            return (1, r)
    for resno in sorted(acc.keys(), key=_key):
        pts = np.vstack(acc[resno])
        coords.append(pts.mean(axis=0))
        resnos.append(resno)
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=float), []
    return np.vstack(coords).astype(float), resnos


def build_contact_graph(
    pdb_file: str,
    chain_id: str,
    cutoff: float = 8.0,
    weighting: str = "gaussian",
    sigma: float = 4.0,
):
    """
    Build a residue-level contact graph for a protein chain.
    Returns a NetworkX graph with node attribute 'resno' and edge attrs 'length' and 'w'.
    """
    try:
        import networkx as nx
    except Exception as e:
        raise ImportError("networkx is required for graph features") from e

    coords, resnos = _parse_calpha_from_pdb(pdb_file, chain_id)
    N = coords.shape[0]
    G = nx.Graph()
    for i, r in enumerate(resnos):
        G.add_node(i, resno=r)
    if N <= 1:
        return G, coords, resnos

    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    mask = (d2 <= cutoff * cutoff) & (d2 > 0.0)
    ei, ej = np.where(mask)
    keep = ei < ej
    ei, ej = ei[keep], ej[keep]
    dist = np.sqrt(d2[ei, ej])

    if weighting == 'gaussian':
        w = np.exp(-(dist * dist) / (2.0 * (sigma ** 2)))
    elif weighting == 'inverse':
        w = 1.0 / np.maximum(dist, 1e-6)
    else:
        w = np.ones_like(dist)

    for u, v, d, ww in zip(ei.tolist(), ej.tolist(), dist.tolist(), w.tolist()):
        G.add_edge(int(u), int(v), length=float(d), w=float(ww))

    return G, coords, resnos


def get_graph_features(
    pdb_file: str,
    chain_id: str,
    cutoff: float = 8.0,
    weighting: str = "gaussian",
    sigma: float = 4.0,
) -> Tuple[np.ndarray, int]:
    """
    Compute per-residue graph features (closeness, eigenvector, betweenness, clustering).
    Attempts to align to sequence indices via utils.sequence_indices when available.
    Returns (features, length).
    """
    try:
        import networkx as nx
    except Exception as e:
        raise ImportError("networkx is required for graph features") from e

    G, coords, resnos = build_contact_graph(
        pdb_file, chain_id, cutoff=cutoff, weighting=weighting, sigma=sigma
    )
    N = coords.shape[0]
    if N == 0:
        return np.zeros((0, 4), dtype=float), 0

    try:
        closeness = nx.closeness_centrality(G, distance='length')
        c_arr = np.array([closeness.get(i, 0.0) for i in range(N)], dtype=float)
    except Exception:
        c_arr = np.zeros(N, dtype=float)

    try:
        ev = nx.eigenvector_centrality_numpy(G, weight='w')
        e_arr = np.array([ev.get(i, 0.0) for i in range(N)], dtype=float)
    except Exception:
        e_arr = np.zeros(N, dtype=float)

    try:
        # Use current-flow betweenness (electrical network model)
        # For large graphs, use sampling to reduce computational cost
        if N > 200:
            k_sample = 64
            btw = nx.current_flow_betweenness_centrality(
                G, weight='length', normalized=True, solver='full'
            )
        else:
            btw = nx.current_flow_betweenness_centrality(
                G, weight='length', normalized=True, solver='full'
            )
        b_arr = np.array([btw.get(i, 0.0) for i in range(N)], dtype=float)
    except Exception as e:
        # Fallback to shortest-path if current-flow fails
        print(f"[Warning] Current-flow betweenness failed ({e}), using shortest-path fallback")
        k_sample = 64 if N > 200 else N
        btw = nx.betweenness_centrality(G, k=k_sample, weight='length', normalized=True, seed=42)
        b_arr = np.array([btw.get(i, 0.0) for i in range(N)], dtype=float)

    try:
        clust = nx.clustering(G, weight='w')
        cl_arr = np.array([clust.get(i, 0.0) for i in range(N)], dtype=float)
    except Exception:
        cl_arr = np.zeros(N, dtype=float)

    feats = np.stack([
        _minmax(c_arr),
        _minmax(e_arr),
        _minmax(b_arr),
        _minmax(cl_arr),
    ], axis=1)

    aligned = feats
    L_or_N = N
    try:
        import os
        from utils.sequence_indices import sequence_indices
        pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
        seq_map = sequence_indices(pdb_id, chain_id)
        if isinstance(seq_map, dict) and len(seq_map) > 0:
            L = max(seq_map.values()) + 1
            aligned = np.zeros((L, feats.shape[1]), dtype=float)
            counts = np.zeros(L, dtype=int)
            for i, resno in enumerate(resnos):
                j = seq_map.get(resno)
                if j is None or j < 0 or j >= L:
                    continue
                aligned[j] += feats[i]
                counts[j] += 1
            counts[counts == 0] = 1
            aligned = aligned / counts[:, None]
            L_or_N = L
    except Exception:
        aligned = feats
        L_or_N = N

    return aligned, L_or_N


def compute_betweenness_centrality(
    pdb_file: str,
    chain_id: str,
    cutoff: float = 8.0,
    weighting: str = "gaussian",
    sigma: float = 4.0,
    normalize: bool = True,
    use_current_flow: bool = True,
) -> np.ndarray:
    """
    Compute current-flow betweenness centrality per residue (default) or shortest-path.
    Aligns to sequence indices if possible. Returns (L,) vector when aligned; else (N,).
    Optionally z-score normalize.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is required for betweenness computation") from e

    G, coords, resnos = build_contact_graph(
        pdb_file, chain_id, cutoff=cutoff, weighting=weighting, sigma=sigma
    )
    N = coords.shape[0]
    if N == 0:
        return np.zeros(0, dtype=float)

    try:
        if use_current_flow:
            # Current-flow betweenness (electrical network model)
            btw = nx.current_flow_betweenness_centrality(
                G, weight='length', normalized=True, solver='full'
            )
        else:
            # Shortest-path betweenness with sampling
            k_sample = 64 if N > 200 else N
            btw = nx.betweenness_centrality(
                G, k=k_sample, weight='length', normalized=True, seed=42
            )
        b_arr = np.array([btw.get(i, 0.0) for i in range(N)], dtype=float)
    except Exception as e:
        print(f"[Betweenness] Computation failed: {e}, returning zeros")
        b_arr = np.zeros(N, dtype=float)

    if normalize:
        mu = b_arr.mean()
        sigma_val = b_arr.std()
        if sigma_val > 1e-10:
            b_arr = (b_arr - mu) / sigma_val
        else:
            b_arr = np.zeros_like(b_arr)

    aligned = b_arr
    try:
        import os
        from utils.sequence_indices import sequence_indices
        pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
        seq_map = sequence_indices(pdb_id, chain_id)
        if isinstance(seq_map, dict) and len(seq_map) > 0:
            L = max(seq_map.values()) + 1
            aligned = np.zeros(L, dtype=float)
            counts = np.zeros(L, dtype=int)
            for i, resno in enumerate(resnos):
                j = seq_map.get(resno)
                if j is None or j < 0 or j >= L:
                    continue
                aligned[j] += b_arr[i]
                counts[j] += 1
            counts[counts == 0] = 1
            aligned = aligned / counts
    except Exception:
        aligned = b_arr

    return aligned

