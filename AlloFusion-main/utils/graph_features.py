"""
Graph propagation features (centralities) for optional topology augmentation.

Cloned from ALLO_TOPO_PLM/AlloFusion-main/utils/graph_features.py
(reason: add opt-in graph/centrality features to Vu pipeline without editing vendor code).

Provides:
- build_contact_graph: constructs a weighted Cα contact graph for a given chain
- get_graph_features: computes per-residue centralities and aligns to sequence
- compute_betweenness_centrality: convenient betweenness extractor (aligned)
- compute_current_flow_closeness_centrality: current-flow closeness / info centrality proxy (aligned)
- compute_laplacian_positional_encoding: fixed-k Laplacian eigenvector PE (aligned)
"""
from typing import Dict, List, Tuple, Optional
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


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma < 1e-10:
        return np.zeros_like(x, dtype=float)
    y = (x - mu) / sigma
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


def _align_to_sequence(
    arr: np.ndarray,
    resnos: List[str],
    pdb_file: str,
    chain_id: str,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Align node-wise arrays to sequence indices when utils.sequence_indices is available.
    If mapping fails (e.g., no network access), returns arr unchanged.
    """
    aligned = arr
    try:
        import os
        from utils.sequence_indices import sequence_indices

        pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
        seq_map = sequence_indices(pdb_id, chain_id)
        if isinstance(seq_map, dict) and len(seq_map) > 0:
            L = max(seq_map.values()) + 1
            if arr.ndim == 1:
                aligned = np.full((L,), fill_value, dtype=float)
                counts = np.zeros(L, dtype=int)
                for i, resno in enumerate(resnos):
                    j = seq_map.get(resno)
                    if j is None or j < 0 or j >= L:
                        continue
                    aligned[j] += float(arr[i])
                    counts[j] += 1
                counts[counts == 0] = 1
                aligned = aligned / counts
            else:
                d = int(arr.shape[1])
                aligned = np.full((L, d), fill_value, dtype=float)
                counts = np.zeros(L, dtype=int)
                for i, resno in enumerate(resnos):
                    j = seq_map.get(resno)
                    if j is None or j < 0 or j >= L:
                        continue
                    aligned[j] += arr[i]
                    counts[j] += 1
                counts[counts == 0] = 1
                aligned = aligned / counts[:, None]
    except Exception:
        aligned = arr
    return aligned


def _componentwise_apply(G, fn):
    """
    Apply a NetworkX metric to each connected component, returning a dict over all nodes.
    Current-flow metrics in NetworkX require connected graphs; this keeps behavior robust.
    """
    import networkx as nx

    out = {n: 0.0 for n in G.nodes()}
    if G.number_of_nodes() <= 1:
        return out
    for comp in nx.connected_components(G):
        if len(comp) <= 1:
            continue
        H = G.subgraph(comp)
        vals = fn(H)
        for n, v in vals.items():
            out[n] = float(v)
    return out


def _current_flow_betweenness(
    G,
    *,
    weight: str,
    normalized: bool,
    solver: str,
    k_sample: Optional[int],
    seed: int,
):
    """
    Compute current-flow betweenness, using an approximation for large graphs when available.
    Falls back to shortest-path betweenness if NetworkX lacks current-flow routines.
    """
    import networkx as nx

    n = G.number_of_nodes()
    if n == 0:
        return {}

    # For small graphs, exact current-flow is typically fine.
    if k_sample is None or n <= 200:
        import warnings

        def _call(H):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    return nx.current_flow_betweenness_centrality(
                        H, weight=weight, normalized=normalized, solver=solver
                    )

        return _componentwise_apply(G, _call)

    # Prefer NetworkX's approximation routine if present (API varies by version).
    approx = getattr(nx, "approximate_current_flow_betweenness_centrality", None)
    if callable(approx):
        for kwargs in (
            {"k": k_sample, "weight": weight, "normalized": normalized, "solver": solver, "seed": seed},
            {"k": k_sample, "weight": weight, "normalized": normalized, "solver": solver},
            {"k": k_sample, "weight": weight, "normalized": normalized},
        ):
            try:
                return _componentwise_apply(G, lambda H: approx(H, **kwargs))
            except TypeError:
                pass
            except Exception:
                break

    # Otherwise, try subset current-flow (sources/targets sampling).
    subset = getattr(nx, "current_flow_betweenness_centrality_subset", None)
    if callable(subset):
        rng = np.random.default_rng(seed)

        def _subset(H):
            nodes = list(H.nodes())
            m = min(k_sample, len(nodes))
            sources = rng.choice(nodes, size=m, replace=False).tolist()
            targets = rng.choice(nodes, size=m, replace=False).tolist()
            for kwargs in (
                {"sources": sources, "targets": targets, "weight": weight, "normalized": normalized, "solver": solver},
                {"sources": sources, "targets": targets, "weight": weight, "normalized": normalized},
            ):
                try:
                    return subset(H, **kwargs)
                except TypeError:
                    continue
            # If signature mismatches, let caller handle.
            return subset(H, sources, targets, normalized=normalized, weight=weight)

        try:
            return _componentwise_apply(G, _subset)
        except Exception:
            pass

    # Fallback: shortest-path betweenness with sampling.
    k_sp = min(k_sample, n)
    return nx.betweenness_centrality(G, k=k_sp, weight=weight, normalized=normalized, seed=seed)


def _current_flow_closeness(
    G,
    *,
    weight: str,
    k_sample: Optional[int],
    seed: int,
):
    """
    Current-flow closeness centrality (aka information centrality proxy) on each component.
    Falls back to (shortest-path) closeness when current-flow is unavailable.
    """
    import networkx as nx

    fn = getattr(nx, "current_flow_closeness_centrality", None)
    if callable(fn):
        # current_flow_closeness_centrality does not support sampling in most versions; k_sample ignored.
        import warnings

        def _call(H):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    return fn(H, weight=weight)

        return _componentwise_apply(G, _call)
    # Fallback: standard closeness (distance-based) on each component
    return _componentwise_apply(G, lambda H: nx.closeness_centrality(H, distance=weight))


def _laplacian_pe_from_adjacency(A: np.ndarray, k: int) -> np.ndarray:
    """
    Compute fixed-k Laplacian positional encodings for one connected component.
    Uses normalized Laplacian L = I - D^{-1/2} A D^{-1/2} and returns the
    k smallest non-trivial eigenvectors (skip the constant eigenvector).
    """
    n = int(A.shape[0])
    if n <= 1 or k <= 0:
        return np.zeros((n, k), dtype=float)
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 0.0))
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(inv_sqrt)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            L = np.eye(n, dtype=float) - (D_inv_sqrt @ A @ D_inv_sqrt)

    # Dense eigen-decomposition; keep sizes modest for efficiency.
    w, v = np.linalg.eigh(L)
    order = np.argsort(w)
    v = v[:, order]

    # Skip the first eigenvector (near-constant / zero-eigenvalue for connected component)
    start = 1
    take = min(k, max(0, n - start))
    out = np.zeros((n, k), dtype=float)
    if take > 0:
        pe = v[:, start : start + take]
        # Deterministic sign: flip so max-|entry| is positive
        for j in range(pe.shape[1]):
            col = pe[:, j]
            idx = int(np.argmax(np.abs(col)))
            if col[idx] < 0:
                pe[:, j] = -col
        out[:, :take] = pe
    return out


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
        btw = _current_flow_betweenness(
            G,
            weight="length",
            normalized=True,
            solver="full",
            k_sample=64 if N > 200 else None,
            seed=42,
        )
        b_arr = np.array([btw.get(i, 0.0) for i in range(N)], dtype=float)
    except Exception as e:
        # Fallback to shortest-path if current-flow fails
        print(f"[Warning] Current-flow betweenness failed ({e}), using shortest-path fallback")
        k_sp = 64 if N > 200 else N
        btw = nx.betweenness_centrality(G, k=k_sp, weight='length', normalized=True, seed=42)
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
    aligned = _align_to_sequence(feats, resnos, pdb_file, chain_id, fill_value=0.0)
    L_or_N = aligned.shape[0]

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
            btw = _current_flow_betweenness(
                G,
                weight="length",
                normalized=True,
                solver="full",
                k_sample=64 if N > 200 else None,
                seed=42,
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

    aligned = _align_to_sequence(b_arr, resnos, pdb_file, chain_id, fill_value=0.0)

    return aligned


def compute_current_flow_closeness_centrality(
    pdb_file: str,
    chain_id: str,
    cutoff: float = 8.0,
    weighting: str = "gaussian",
    sigma: float = 4.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute current-flow closeness centrality per residue when available (NetworkX),
    falling back to shortest-path closeness. Aligns to sequence indices if possible.
    Returns (L,) when aligned; else (N,). Optionally z-score normalize per protein.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is required for graph closeness computation") from e

    G, coords, resnos = build_contact_graph(
        pdb_file, chain_id, cutoff=cutoff, weighting=weighting, sigma=sigma
    )
    N = coords.shape[0]
    if N == 0:
        return np.zeros(0, dtype=float)

    try:
        # Note: current-flow closeness is defined on connected components; we compute per component.
        clo = _current_flow_closeness(G, weight="length", k_sample=None, seed=42)
        c_arr = np.array([clo.get(i, 0.0) for i in range(N)], dtype=float)
    except Exception as e:
        print(f"[CF-Closeness] Computation failed: {e}, returning zeros")
        c_arr = np.zeros(N, dtype=float)

    if normalize:
        c_arr = _zscore(c_arr)

    return _align_to_sequence(c_arr, resnos, pdb_file, chain_id, fill_value=0.0)


def compute_laplacian_positional_encoding(
    pdb_file: str,
    chain_id: str,
    k: int = 8,
    cutoff: float = 8.0,
    weighting: str = "gaussian",
    sigma: float = 4.0,
    normalize: bool = False,
    max_nodes_dense: int = 800,
) -> np.ndarray:
    """
    Fixed-k Laplacian positional encodings (PE) on the residue contact graph.

    - Computes normalized Laplacian eigenvectors per connected component.
    - Uses dense eigen-decomposition up to max_nodes_dense; larger graphs -> zeros.
    - Aligns to sequence indices if possible.
    - Optional per-dimension z-score normalization (per protein).
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is required for Laplacian PE computation") from e

    G, coords, resnos = build_contact_graph(
        pdb_file, chain_id, cutoff=cutoff, weighting=weighting, sigma=sigma
    )
    N = coords.shape[0]
    if N == 0 or k <= 0:
        return np.zeros((0, max(k, 0)), dtype=float)

    out = np.zeros((N, k), dtype=float)
    for comp in nx.connected_components(G):
        nodes = sorted(comp)
        m = len(nodes)
        if m <= 1:
            continue
        if m > max_nodes_dense:
            # Too big for dense eigendecomposition in this lightweight implementation
            continue

        idx = {n: t for t, n in enumerate(nodes)}
        A = np.zeros((m, m), dtype=float)
        for u, v, data in G.subgraph(nodes).edges(data=True):
            w = float(data.get("w", 1.0))
            iu = idx[u]
            iv = idx[v]
            A[iu, iv] = w
            A[iv, iu] = w

        pe = _laplacian_pe_from_adjacency(A, k)
        out[nodes, :] = pe

    if normalize and out.shape[0] > 0:
        for j in range(out.shape[1]):
            out[:, j] = _zscore(out[:, j])

    return _align_to_sequence(out, resnos, pdb_file, chain_id, fill_value=0.0)
