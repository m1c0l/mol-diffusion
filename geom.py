"""
Internal coordinate geometry for 2D molecular diffusion.

Two things live here:

1. INTERNAL COORDINATE DIFFUSION
   Instead of corrupting (x, y) positions directly, we corrupt the molecule's
   internal coordinates — bond lengths and bond angles — then reconstruct
   Cartesian positions from the noisy internals. This keeps the noise in a
   space that directly represents what the physics constraints are about.

   Internally we use:
     - bond length per spanning-tree edge (one scalar, ~1.0 normalized)
     - absolute bond angle per spanning-tree edge (one scalar, in radians)
     - root atom position (two scalars for atom 0)

   Non-spanning-tree edges (ring closure bonds) are not diffused separately —
   their geometry is implied by the spanning tree; ring closure error is
   what shows up as constraint violation in rings under noise.

2. CONSTRAINT LOSS
   During training, given the model's predicted clean coordinates x0_hat,
   compute bond lengths and angles and penalize deviation from ideal values:
     - Bond lengths should be ~1.0 (tight spike in the dataset)
     - Bond angles should be near 60 / 109.5 / 120 / 180 degrees
   The angle loss snaps each predicted angle to the nearest valid value
   rather than requiring a single fixed target — the model learns the
   multi-modal distribution directly.
"""

import numpy as np
import torch
from collections import defaultdict, deque


# ── Spanning tree ─────────────────────────────────────────────────────────────

def build_spanning_tree(bonds: list, n_atoms: int) -> list:
    """
    BFS spanning tree rooted at atom 0.
    Returns list of (parent, child) in BFS order — parent is always placed
    before child, which is the order needed for sequential coordinate reconstruction.
    """
    adj = defaultdict(list)
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    visited = {0}
    tree_edges = []
    queue = deque([0])

    while queue:
        node = queue.popleft()
        for nbr in sorted(adj[node]):  # sorted for determinism
            if nbr not in visited:
                visited.add(nbr)
                tree_edges.append((node, nbr))
                queue.append(nbr)

    return tree_edges


# ── Cartesian ↔ internal coordinate conversion ───────────────────────────────

def to_internal(coords: np.ndarray, tree_edges: list) -> tuple:
    """
    Decompose 2D Cartesian coordinates into internal coordinates.

    For each spanning tree edge (parent → child):
      length: Euclidean bond length
      angle:  absolute direction of the bond in the 2D plane (atan2), radians

    Returns (lengths, angles, root_pos) as flat numpy arrays:
      lengths: (E,)  one per tree edge
      angles:  (E,)  one per tree edge, in radians
      root_pos:(2,)  position of atom 0
    """
    lengths = np.zeros(len(tree_edges), dtype=np.float32)
    angles  = np.zeros(len(tree_edges), dtype=np.float32)

    for idx, (p, c) in enumerate(tree_edges):
        vec = coords[c] - coords[p]
        lengths[idx] = float(np.linalg.norm(vec)) + 1e-8
        angles[idx]  = float(np.arctan2(vec[1], vec[0]))

    return lengths, angles, coords[0].copy().astype(np.float32)


def from_internal(lengths: np.ndarray, angles: np.ndarray,
                  tree_edges: list, root_pos: np.ndarray, n_atoms: int) -> np.ndarray:
    """
    Reconstruct 2D Cartesian coordinates from internal coordinates.
    Tree edges must be in BFS order (guaranteed by build_spanning_tree).
    """
    coords = np.zeros((n_atoms, 2), dtype=np.float32)
    coords[0] = root_pos

    for idx, (p, c) in enumerate(tree_edges):
        l     = lengths[idx]
        theta = angles[idx]
        coords[c] = coords[p] + l * np.array([np.cos(theta), np.sin(theta)],
                                              dtype=np.float32)
    return coords


# ── Internal coordinate forward diffusion ────────────────────────────────────

def internal_q_sample(coords: np.ndarray, tree_edges: list,
                      t: int, alpha_bar: np.ndarray) -> tuple:
    """
    Forward diffusion in internal coordinate space.

    Steps:
      1. Decompose clean coords into (lengths, angles, root_pos)
      2. Add independent Gaussian noise to each internal scalar
         using the same alpha_bar schedule as Cartesian diffusion
      3. Reconstruct noisy Cartesian coords from noisy internals

    The noise magnitudes are scaled separately for each coordinate type:
      - Lengths:    noise std scales with (1 - abar)^0.5, same as Cartesian
      - Angles:     noise std scaled by π so full noise ≈ uniform over [−π, π]
      - Root pos:   plain Cartesian noise

    Returns:
      noisy_coords (N, 2): reconstructed from noisy internals
      noise_lengths (E,):  Gaussian noise added to lengths
      noise_angles  (E,):  Gaussian noise added to angles (in radians, scaled)
      noise_root    (2,):  Gaussian noise added to root position
    """
    lengths, angles, root_pos = to_internal(coords, tree_edges)

    abar         = float(alpha_bar[t])
    sqrt_abar    = np.sqrt(abar)
    sqrt_1m_abar = np.sqrt(1.0 - abar)

    # Root position: standard Cartesian noise
    noise_root    = np.random.randn(2).astype(np.float32)
    noisy_root    = sqrt_abar * root_pos + sqrt_1m_abar * noise_root

    # Bond lengths: noise in length space (clamp to stay positive)
    noise_lengths = np.random.randn(len(lengths)).astype(np.float32)
    noisy_lengths = sqrt_abar * lengths + sqrt_1m_abar * noise_lengths
    noisy_lengths = np.clip(noisy_lengths, 0.01, None)

    # Bond angles: noise scaled by π so at t=T the angle is fully random
    noise_angles  = np.random.randn(len(angles)).astype(np.float32)
    noisy_angles  = sqrt_abar * angles + sqrt_1m_abar * noise_angles * np.pi

    noisy_coords = from_internal(noisy_lengths, noisy_angles,
                                 tree_edges, noisy_root, len(coords))
    return noisy_coords, noise_lengths, noise_angles, noise_root


def pack_internal_noise(noise_lengths, noise_angles, noise_root,
                        tree_edges, n_atoms) -> np.ndarray:
    """
    Scatter internal noise back to per-atom arrays for loss computation.
    Each atom's noise = mean of the noise contributions from its incident
    spanning tree edges (as parent) plus its root noise if it's atom 0.
    This gives a (N, 3) array: [noise_from_length, noise_from_angle, 0]
    per atom — used to define what the model should predict.

    In practice we just store it as two separate edge-level arrays and
    compute the loss directly on edge predictions in the model output.
    """
    # Returns flat internal noise vector: [noise_root(2), noise_lengths(E), noise_angles(E)]
    return np.concatenate([noise_root, noise_lengths, noise_angles])


# ── Constraint losses ─────────────────────────────────────────────────────────

# Valid bond angles in degrees for drug-like molecules
VALID_ANGLES_DEG = torch.tensor([60.0, 90.0, 109.5, 120.0, 135.0, 150.0, 180.0])


def bond_length_loss(pred_coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Penalize bond lengths that deviate from 1.0 (our normalized target).
    Uses only one direction of each undirected edge (src < dst) to avoid double counting.
    """
    src, dst = edge_index
    mask = src < dst  # deduplicate undirected edges
    src, dst = src[mask], dst[mask]

    if src.shape[0] == 0:
        return pred_coords.new_zeros(1).squeeze()

    vecs    = pred_coords[src] - pred_coords[dst]
    lengths = torch.norm(vecs, dim=-1)
    return ((lengths - 1.0) ** 2).mean()


def angle_constraint_loss(pred_coords: torch.Tensor,
                          angle_triples: torch.Tensor) -> torch.Tensor:
    """
    For each bond angle triple (i, center, k), compute the angle at center
    and penalize its distance to the nearest valid angle (60/109.5/120/180°).

    This is a soft many-modal constraint: the model learns to push angles
    toward one of the valid peaks without being told which peak applies to
    which atom — it learns that from the molecular context.
    """
    if angle_triples is None or angle_triples.shape[0] == 0:
        return pred_coords.new_zeros(1).squeeze()

    i_idx = angle_triples[:, 0]
    j_idx = angle_triples[:, 1]
    k_idx = angle_triples[:, 2]

    v1 = pred_coords[i_idx] - pred_coords[j_idx]  # (A, 2)
    v2 = pred_coords[k_idx] - pred_coords[j_idx]  # (A, 2)

    # Use atan2 instead of acos: no singularity at ±1, backward is always finite.
    # In 2D: |cross| = |v1x*v2y - v1y*v2x|, dot = v1·v2
    # atan2(|cross|, dot) gives angle in [0, π] without acos singularity.
    dot   = (v1 * v2).sum(-1)                                  # (A,)
    cross = (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]).abs() # (A,) 2D cross magnitude
    angles_deg = torch.atan2(cross, dot) * (180.0 / torch.pi)  # (A,)

    valid = VALID_ANGLES_DEG.to(pred_coords.device)            # (V,)
    # Distance from each predicted angle to each valid angle
    dists = (angles_deg.unsqueeze(-1) - valid.unsqueeze(0)) ** 2  # (A, V)
    # Penalize by distance to nearest valid angle
    return dists.min(dim=-1).values.mean()
