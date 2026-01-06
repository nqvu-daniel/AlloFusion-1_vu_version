"""
StingAllo-inspired structural features for allosteric site prediction.

Based on: Omage et al. (2025) "STINGAllo: a web server for high-throughput
prediction of allosteric site-forming residues"

Features computed per-residue:
1. Distance to centroid (1D) - distance from residue Cα to protein geometric center
2. Local density (1D) - number of Cα atoms within 10Å cutoff
3. Hydrophobic contact ratio (1D) - fraction of contacts that are hydrophobic
4. Secondary structure (3D) - one-hot encoding [Helix, Sheet, Coil]
5. Relative solvent accessibility (1D) - RSA from DSSP (preferred) or estimated
6. B-factor normalized (1D) - crystallographic flexibility (skipped if RSA used B-factor)
7. Graph centralities (3D) - degree, closeness, betweenness from contact graph

Total: 11 new features per residue
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence

# Hydrophobic residues for contact classification
HYDROPHOBIC_RESIDUES = set(['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P', 'G'])

# Max ASA values for RSA calculation (Tien et al. 2013)
MAX_ASA = {
    'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167,
    'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
    'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159,
    'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174,
    'X': 200  # Unknown
}


def _warn(msg: str) -> None:
    """Lightweight logger to surface fallbacks."""
    print(f"[StingAllo] {msg}")


def _parse_pdb_calpha(pdb_file: str, chain_id: str) -> Tuple[np.ndarray, List[str], List[str], List[float]]:
    """
    Parse PDB file to extract Cα coordinates, residue numbers, residue names, and B-factors.
    
    Returns:
        coords: (N, 3) array of Cα coordinates
        resnos: list of residue numbers (strings)
        resnames: list of 1-letter residue codes
        bfactors: list of B-factors
    """
    coords = []
    resnos = []
    resnames = []
    bfactors = []
    
    aa_3to1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    seen_resnos = set()
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[21].strip() != chain_id:
                continue
            if line[12:16].strip() != 'CA':
                continue
            
            resno = line[22:27].strip()  # Include insertion code
            if resno in seen_resnos:
                continue
            seen_resnos.add(resno)
            
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66])
            except ValueError:
                continue
            
            resname_3 = line[17:20].strip()
            resname_1 = aa_3to1.get(resname_3, 'X')
            
            coords.append([x, y, z])
            resnos.append(resno)
            resnames.append(resname_1)
            bfactors.append(bfactor)
    
    if len(coords) == 0:
        return np.zeros((0, 3)), [], [], []
    
    return np.array(coords, dtype=float), resnos, resnames, bfactors


def compute_distance_to_centroid(coords: np.ndarray) -> np.ndarray:
    """Compute distance from each Cα to the protein geometric center."""
    if len(coords) == 0:
        return np.array([])
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    # Normalize to [0, 1]
    if distances.max() > distances.min():
        distances = (distances - distances.min()) / (distances.max() - distances.min())
    return distances


def compute_local_density(coords: np.ndarray, cutoff: float = 10.0) -> np.ndarray:
    """Count number of Cα atoms within cutoff distance for each residue."""
    N = len(coords)
    if N == 0:
        return np.array([])
    
    # Vectorized distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = (dists > 0) & (dists <= cutoff)
    density = mask.sum(axis=1).astype(float)
    
    # Normalize to [0, 1]
    if density.max() > density.min():
        density = (density - density.min()) / (density.max() - density.min())
    return density


def compute_hydrophobic_contact_ratio(
    coords: np.ndarray, 
    resnames: List[str], 
    cutoff: float = 8.0
) -> np.ndarray:
    """
    Compute fraction of contacts that are with hydrophobic residues.
    """
    N = len(coords)
    if N == 0:
        return np.array([])
    
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    contacts = (dists > 0) & (dists <= cutoff)

    hydrophobic_mask = np.array([r in HYDROPHOBIC_RESIDUES for r in resnames])[None, :]
    hydrophobic_contacts = (contacts & hydrophobic_mask).sum(axis=1)
    contact_counts = contacts.sum(axis=1)

    ratios = np.zeros(N, dtype=float)
    valid = contact_counts > 0
    ratios[valid] = hydrophobic_contacts[valid] / contact_counts[valid]

    return ratios


def compute_secondary_structure_onehot(
    pdb_file: str, 
    chain_id: str,
    n_residues: int
) -> np.ndarray:
    """
    Compute secondary structure one-hot encoding [Helix, Sheet, Coil].
    Attempts DSSP first, falls back to ProDy, then to all-coil.
    """
    ss_onehot = np.zeros((n_residues, 3), dtype=float)
    ss_onehot[:, 2] = 1.0  # Default: all coil
    
    # Try DSSP via BioPython
    try:
        from Bio.PDB import PDBParser, DSSP
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        dssp = DSSP(model, pdb_file, dssp='mkdssp')
        
        idx = 0
        for key in dssp.keys():
            if key[0] != chain_id:
                continue
            if idx >= n_residues:
                break
            ss = dssp[key][2]  # Secondary structure code
            ss_onehot[idx] = [0, 0, 0]
            if ss in ['H', 'G', 'I']:  # Helix types
                ss_onehot[idx, 0] = 1.0
            elif ss in ['E', 'B']:  # Sheet types
                ss_onehot[idx, 1] = 1.0
            else:  # Coil/other
                ss_onehot[idx, 2] = 1.0
            idx += 1
        return ss_onehot
    except Exception:
        _warn("DSSP not available or failed; falling back to STRIDE/coil.")
    
    # Try ProDy stride
    try:
        from prody import parsePDB, STRIDE
        ag = parsePDB(pdb_file)
        stride = STRIDE(ag.select(f'chain {chain_id}'))
        ss_codes = stride.getSecstrs()
        
        for i, ss in enumerate(ss_codes[:n_residues]):
            ss_onehot[i] = [0, 0, 0]
            if ss in ['H', 'G', 'I']:
                ss_onehot[i, 0] = 1.0
            elif ss in ['E', 'B']:
                ss_onehot[i, 1] = 1.0
            else:
                ss_onehot[i, 2] = 1.0
        return ss_onehot
    except Exception:
        _warn("ProDy/STRIDE not available; defaulting to coil for secondary structure.")
    
    # Fallback: all coil
    return ss_onehot


def compute_rsa_dssp(
    pdb_file: str,
    chain_id: str,
    resnos: List[str],
    resnames: List[str]
) -> Optional[np.ndarray]:
    """
    Compute RSA using DSSP ASA values. Returns None if DSSP unavailable.
    """
    try:
        from Bio.PDB import PDBParser, DSSP
    except Exception:
        return None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        dssp = DSSP(model, pdb_file, dssp='mkdssp')

        rsa = np.zeros(len(resnos), dtype=float)
        for i, resno in enumerate(resnos):
            # DSSP keys are (chain, residue id), where residue id is a tuple
            # Handle simple numeric residue indices; otherwise fall back to order
            key = (chain_id, (' ', int(resno.split()[0]), '')) if resno.strip('-').isdigit() else None
            if key and key in dssp:
                asa = dssp[key][3]
                aa = dssp[key][1]
            else:
                # fallback: align by index
                try:
                    key = list(dssp.keys())[i]
                    asa = dssp[key][3]
                    aa = dssp[key][1]
                except Exception:
                    asa = None
                    aa = 'X'

            max_asa = MAX_ASA.get(aa, MAX_ASA['X'])
            rsa[i] = asa / max_asa if asa is not None else 0.0

        rsa = np.clip(rsa, 0.0, 1.0)
        return rsa
    except Exception:
        _warn("DSSP ASA extraction failed; using proxy RSA.")
        return None


def compute_rsa_estimated(resnames: List[str], bfactors: List[float]) -> np.ndarray:
    """
    Estimate relative solvent accessibility.
    Uses B-factor as proxy (higher B-factor → more exposed).
    This is a rough approximation when DSSP is unavailable.
    """
    N = len(resnames)
    if N == 0:
        return np.array([])
    
    # Normalize B-factors to [0, 1] as RSA proxy
    bfactors = np.array(bfactors)
    if bfactors.max() > bfactors.min():
        rsa = (bfactors - bfactors.min()) / (bfactors.max() - bfactors.min())
    else:
        rsa = np.ones(N) * 0.5
    
    return rsa


def compute_bfactor_normalized(bfactors: List[float]) -> np.ndarray:
    """Normalize B-factors to [0, 1]."""
    bfactors = np.array(bfactors)
    if len(bfactors) == 0:
        return np.array([])
    if bfactors.max() > bfactors.min():
        return (bfactors - bfactors.min()) / (bfactors.max() - bfactors.min())
    return np.ones(len(bfactors)) * 0.5


def compute_graph_centralities(
    coords: np.ndarray, 
    cutoff: float = 8.0
) -> np.ndarray:
    """
    Compute graph centralities: degree, closeness, betweenness.
    Returns (N, 3) array.
    """
    N = len(coords)
    if N == 0:
        return np.zeros((0, 3))
    
    try:
        import networkx as nx
    except ImportError:
        # Fallback: return zeros
        _warn("networkx not installed; centralities set to zero.")
        return np.zeros((N, 3), dtype=float)
    
    # Build contact graph (vectorized edge detection)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.where((dists > 0) & (dists <= cutoff) & (np.triu(np.ones((N, N)), k=1) > 0))
    for i, j in zip(i_idx, j_idx):
        G.add_edge(i, j, weight=dists[i, j])
    
    # Check connectivity
    if not nx.is_connected(G):
        n_components = nx.number_connected_components(G)
        _warn(f"Contact graph disconnected ({n_components} components); centralities may be unreliable.")
    
    # Degree centrality
    degree = nx.degree_centrality(G)
    degree_arr = np.array([degree.get(i, 0) for i in range(N)])
    
    # Closeness centrality
    try:
        closeness = nx.closeness_centrality(G, distance='weight')
        closeness_arr = np.array([closeness.get(i, 0) for i in range(N)])
    except:
        closeness_arr = np.zeros(N)
        _warn("Closeness centrality failed; set to zero.")
    
    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
        betweenness_arr = np.array([betweenness.get(i, 0) for i in range(N)])
    except:
        betweenness_arr = np.zeros(N)
        _warn("Betweenness centrality failed; set to zero.")
    
    # Normalize each to [0, 1]
    def normalize(arr):
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        return arr
    
    centralities = np.stack([
        normalize(degree_arr),
        normalize(closeness_arr),
        normalize(betweenness_arr)
    ], axis=1)
    
    return centralities


def compute_stingallo_features(
    pdb_file: str,
    chain_id: str,
    seq_length: int,
    cutoff: float = 8.0,
    seq_resnos: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Compute all StingAllo-inspired features for a protein chain.
    
    Args:
        pdb_file: Path to PDB file
        chain_id: Chain identifier
        seq_length: Expected sequence length (for alignment)
        cutoff: Distance cutoff for contacts (Å)
        seq_resnos: Optional sequence residue numbers to align features exactly
    
    Returns:
        features: (seq_length, 11) array of features
            [0] distance_to_centroid
            [1] local_density
            [2] hydrophobic_contact_ratio
            [3:6] secondary_structure_onehot (H, E, C)
            [6] rsa_estimated
            [7] bfactor_normalized
            [8:11] graph_centralities (degree, closeness, betweenness)
    """
    # Parse PDB
    coords, resnos, resnames, bfactors = _parse_pdb_calpha(pdb_file, chain_id)
    N = len(coords)
    
    if N == 0:
        print(f"[StingAllo] Warning: No Cα atoms found for {chain_id} in {pdb_file}")
        return np.zeros((seq_length, 11), dtype=float)
    
    # Compute features
    dist_centroid = compute_distance_to_centroid(coords)
    local_density = compute_local_density(coords, cutoff=10.0)
    hydro_ratio = compute_hydrophobic_contact_ratio(coords, resnames, cutoff=cutoff)
    ss_onehot = compute_secondary_structure_onehot(pdb_file, chain_id, N)
    rsa = compute_rsa_estimated(resnames, bfactors)
    bfactor_norm = compute_bfactor_normalized(bfactors)
    centralities = compute_graph_centralities(coords, cutoff=cutoff)

    # Prefer true RSA from DSSP; if using B-factor proxy, avoid duplicating it as a separate channel
    rsa_dssp = compute_rsa_dssp(pdb_file, chain_id, resnos, resnames)
    if rsa_dssp is not None:
        rsa = rsa_dssp
    else:
        _warn("Using B-factor proxy for RSA; setting bfactor channel to zeros to avoid duplication.")
        bfactor_norm = np.zeros_like(rsa)
    
    # Stack all features: (N, 11)
    features = np.column_stack([
        dist_centroid,      # 1
        local_density,      # 1
        hydro_ratio,        # 1
        ss_onehot,          # 3
        rsa,                # 1
        bfactor_norm,       # 1
        centralities        # 3
    ])
    
    # Align to sequence length (pad or truncate)
    aligned = np.zeros((seq_length, 11), dtype=float)
    if seq_resnos:
        resno_to_idx = {res: i for i, res in enumerate(resnos)}
        for i, res in enumerate(seq_resnos):
            if i >= seq_length:
                break
            j = resno_to_idx.get(str(res))
            if j is not None and j < N:
                aligned[i] = features[j]
    else:
        m = min(seq_length, N)
        aligned[:m] = features[:m]
    
    return aligned


def compute_stingallo_features_batch(
    pdb_dir: str,
    pdb_names: List[str],
    seq_lengths: List[int],
    cutoff: float = 8.0
) -> List[np.ndarray]:
    """
    Compute StingAllo features for a batch of proteins.
    
    Args:
        pdb_dir: Directory containing PDB files
        pdb_names: List of "PDBID_CHAIN" strings
        seq_lengths: List of sequence lengths
        cutoff: Distance cutoff
    
    Returns:
        List of (seq_length, 11) feature arrays
    """
    features_list = []
    
    for name, seq_len in zip(pdb_names, seq_lengths):
        name = str(name).strip().lstrip('>')
        parts = name.split('_')
        
        if len(parts) >= 2:
            pdb_id = parts[0]
            chain_id = parts[1]
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            
            if os.path.exists(pdb_file):
                try:
                    feats = compute_stingallo_features(pdb_file, chain_id, seq_len, cutoff)
                    features_list.append(feats)
                    continue
                except Exception as e:
                    print(f"[StingAllo] Error processing {name}: {e}")
        
        # Fallback: zeros
        features_list.append(np.zeros((seq_len, 11), dtype=float))
    
    return features_list
