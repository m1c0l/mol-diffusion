"""
E(2)-equivariant GNN denoiser for 2D molecular diffusion (EGNN).

Replaces the original non-equivariant MessagePassingLayer with EGNN layers
(Satorras et al., 2021). Key properties:

  - Coordinate track x is updated equivariantly: the update is a weighted sum
    of displacement vectors (x_i - x_j), which rotates correctly with the input.
  - Feature track h uses only invariant quantities: atom type, timestep, and
    pairwise squared distances ||x_i - x_j||^2. Raw coordinates never enter h.
  - Output is the refined coordinate tensor x (= predicted x0), not noise.

This fixes mean-collapse for rings: the network reasons about relative atom
positions rather than absolute (x, y), so it can't hedge across orientations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


ATOM_VOCAB = ['C', 'N', 'O', 'F', 'S', 'Cl', 'P', 'Br', 'I', 'B', 'X']
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_VOCAB)}

BOND_VOCAB = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
BOND_TO_IDX = {b: i for i, b in enumerate(BOND_VOCAB)}


def atom_idx(symbol: str) -> int:
    return ATOM_TO_IDX.get(symbol, ATOM_TO_IDX['X'])


def bond_idx(bond_type: str) -> int:
    return BOND_TO_IDX.get(bond_type, 0)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out


class EGNNLayer(nn.Module):
    """
    One EGNN layer. Updates both node features h (invariant) and
    coordinates x (equivariant) jointly.

    Message:        m_ij = φ_e(h_i, h_j, ||x_i-x_j||², edge_attr)
    Coord update:   x_i ← x_i + Σ_j (x_i-x_j) · φ_x(m_ij)   [equivariant]
    Node update:    h_i ← φ_h(h_i, Σ_j m_ij)                  [invariant]
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        # Invariant message: h_i, h_j, dist², edge_attr → message vector
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Scalar weight for equivariant coord update.
        # Last layer zero-init: coord updates start at 0 and grow during training,
        # preventing cascade amplification through deep stacks of EGNN layers.
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)
        # Node feature update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, x, edge_index, edge_attr):
        src, dst = edge_index
        N = h.shape[0]

        # Clamp coords before computing dist² to prevent float32 overflow in deep stacks
        x     = x.clamp(-20, 20)
        rel   = x[src] - x[dst]                              # (E, 2) equivariant
        dist2 = (rel ** 2).sum(-1, keepdim=True)             # (E, 1) invariant

        msgs    = self.msg_mlp(torch.cat([h[src], h[dst], dist2, edge_attr], dim=-1))
        weights = self.coord_mlp(msgs)                        # (E, 1) scalar
        x = x + scatter_add(rel * weights, dst, N)           # equivariant update

        agg   = scatter_add(msgs, dst, N)                     # (N, hidden_dim)
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h     = self.norm(h + h_new)

        return h, x


class GNNDenoiser(nn.Module):
    """
    EGNN-based denoiser. Takes noisy coords + graph, outputs predicted x0
    (clean coordinates) via equivariant coordinate refinement.

    The coordinate track starts at noisy_coords and is updated equivariantly
    through each layer — the final x is the x0 prediction.
    The feature track h encodes only invariant information (atom type, timestep,
    pairwise distances) and guides the coordinate updates.
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 4, t_dim: int = 64):
        super().__init__()
        n_atom_types = len(ATOM_VOCAB)
        n_bond_types  = len(BOND_VOCAB)
        edge_dim = n_bond_types

        self.atom_emb = nn.Embedding(n_atom_types, 32)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # h init: atom_emb + t_emb only — no raw coords (preserves invariance)
        self.input_proj = nn.Linear(32 + hidden_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim) for _ in range(n_layers)
        ])

        self.t_dim = t_dim

    def forward(self, coords_noisy, atom_types, edge_index, edge_attr, t,
                batch=None, cond_emb=None):
        """
        coords_noisy: (N, 2)
        atom_types:   (N,)
        edge_index:   (2, E)
        edge_attr:    (E, n_bond_types)
        t:            (B,)
        batch:        (N,) or None
        cond_emb:     (N, hidden_dim) or None — per-atom condition embedding
                      (e.g. ring count, scaffold flags) added to h after input_proj

        Returns x0_pred: (N, 2) — predicted clean coordinates.
        """
        N = coords_noisy.shape[0]

        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))  # (B, hidden_dim)

        if batch is None:
            t_emb_per_atom = t_emb.expand(N, -1)
        else:
            t_emb_per_atom = t_emb[batch]

        h = self.input_proj(torch.cat([self.atom_emb(atom_types), t_emb_per_atom], dim=-1))
        if cond_emb is not None:
            h = h + cond_emb  # additive injection — same hidden_dim, preserves invariance
        x = coords_noisy

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        return x  # predicted x0
