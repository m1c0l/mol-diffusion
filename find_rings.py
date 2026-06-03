"""
Find and display the best-generated ring molecules.

Samples a large batch, scores each molecule by how circular its 6-membered
rings are (low variance in atom-to-centroid distances = tight hexagon), then
plots the top N alongside the RDKit ground truth for comparison.

Usage:
    python find_rings.py                                        # base model
    python find_rings.py --checkpoint ring_cond_large_model.pt \
        --hidden_dim 256 --n_layers 6 --t_dim 128 --rings 1   # large model
"""

import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
import random

from dataset import build_dataset
from train import mol_to_pyg_with_bond_types
from model import GNNDenoiser
from noise import cosine_schedule, p_sample_step_x0
from sample import load_model, sample_molecule


def get_six_membered_rings(smiles: str):
    """Return list of atom-index tuples for each 6-membered ring."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [list(r) for r in mol.GetRingInfo().AtomRings() if len(r) == 6]


def ring_circularity(coords: np.ndarray, ring_atoms: list) -> float:
    """
    Score how circular a ring is. Lower = better.
    Computes std of atom distances from the ring centroid,
    normalized by mean distance (scale-invariant).
    """
    pts = coords[ring_atoms]
    centroid = pts.mean(0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    if dists.mean() < 1e-6:
        return float('inf')
    return dists.std() / dists.mean()


def score_molecule(coords: np.ndarray, rings: list) -> float:
    """Mean circularity across all 6-membered rings. Lower = better."""
    if not rings:
        return float('inf')
    scores = [ring_circularity(coords, r) for r in rings]
    return np.mean(scores)


def find_good_samples(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    schedule = {k: v.to(device) for k, v in cosine_schedule(args.T).items()}
    model, ring_emb = load_model(args.checkpoint, device,
                                 hidden_dim=args.hidden_dim,
                                 n_layers=args.n_layers,
                                 t_dim=args.t_dim)

    mols = build_dataset(source='chembl')
    with open('chembl_mols.json') as f:
        entries = json.load(f)
    smiles_list = [e['smiles'] for e in entries[:len(mols)]]

    random.seed(args.seed)
    indices = list(range(len(mols)))
    random.shuffle(indices)
    val_idx = indices[int(0.9 * len(indices)):]

    # Only consider molecules with at least one 6-membered ring
    candidates = []
    for idx in val_idx:
        rings = get_six_membered_rings(smiles_list[idx])
        if rings:
            candidates.append((idx, rings))

    sample_pool = candidates[:args.n_sample]
    print(f"Sampling {len(sample_pool)} molecules with 6-membered rings...")

    results = []
    for i, (idx, rings) in enumerate(sample_pool):
        mol = mols[idx]
        pyg = mol_to_pyg_with_bond_types(mol, smiles_list[idx])
        gen = sample_molecule(pyg, model, schedule, args.T, device,
                              ring_emb=ring_emb, rings=args.rings,
                              guidance=args.guidance)

        # Normalize scale
        from sample import compute_bond_lengths
        bl = compute_bond_lengths(gen, mol.bonds)
        if len(bl) > 0 and bl.mean() > 1e-6:
            gen /= bl.mean()

        sc = score_molecule(gen, rings)
        if np.isfinite(sc):
            results.append((sc, idx, gen, rings))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample_pool)}")

    results.sort(key=lambda x: x[0])
    top = results[:args.top_n]
    print(f"\nTop {len(top)} ring scores (lower=more circular):")
    for sc, idx, _, _ in top:
        print(f"  {smiles_list[idx][:50]:50s}  score={sc:.4f}  "
              f"n_atoms={mols[idx].n_atoms}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_cols = 4
    n_rows = (args.top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    for plot_i, (sc, idx, gen_coords, rings) in enumerate(top):
        mol = mols[idx]
        ax  = axes[plot_i]

        c_gt  = mol.coords   - mol.coords.mean(0)
        c_gen = gen_coords   - gen_coords.mean(0)

        # Bonds
        for i_b, j_b in mol.bonds:
            ax.plot([c_gt[i_b,0],  c_gt[j_b,0]],  [c_gt[i_b,1],  c_gt[j_b,1]],
                    color='steelblue', alpha=0.5, lw=1.5)
            ax.plot([c_gen[i_b,0], c_gen[j_b,0]], [c_gen[i_b,1], c_gen[j_b,1]],
                    color='coral', alpha=0.5, lw=1.5, linestyle='--')

        # Atoms
        ax.scatter(*c_gt.T,  s=40, c='steelblue', zorder=3)
        ax.scatter(*c_gen.T, s=40, c='coral',     zorder=3, marker='^')

        # Highlight ring atoms
        for ring in rings:
            ring_pts = c_gen[ring]
            poly = plt.Polygon(ring_pts, fill=False, edgecolor='gold',
                               linewidth=2, zorder=4)
            ax.add_patch(poly)

        n_rings = len(rings)
        ax.set_title(f"{mol.name[:16]}  ({mol.n_atoms}a, {n_rings}×6r)\nscore={sc:.3f}",
                     fontsize=7)
        ax.set_aspect('equal')
        ax.axis('off')

    for ax in axes[len(top):]:
        ax.axis('off')

    cond_label = f"rings={args.rings}" if ring_emb is not None else "unconditional"
    plt.suptitle(
        f"Best-generated 6-membered rings [{cond_label}] — Gold outline = generated ring\n"
        f"Blue=RDKit  Coral=Generated  (lower circularity score = tighter hexagon)",
        fontsize=9
    )
    plt.tight_layout()
    out = f"{args.prefix}_best_rings.png" if args.prefix else "best_rings.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='ring_cond_large_model.pt')
    parser.add_argument('--hidden_dim', type=int,   default=256)
    parser.add_argument('--n_layers',   type=int,   default=6)
    parser.add_argument('--t_dim',      type=int,   default=128)
    parser.add_argument('--T',          type=int,   default=200)
    parser.add_argument('--n_sample',   type=int,   default=300,
                        help='How many ring-containing val mols to sample')
    parser.add_argument('--top_n',      type=int,   default=12,
                        help='How many best results to plot')
    parser.add_argument('--rings',      type=int,   default=1,
                        help='Ring-count condition passed to model (-1=unconditional)')
    parser.add_argument('--guidance',   type=float, default=1.0)
    parser.add_argument('--prefix',     default='large_ep100')
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()
    find_good_samples(args)
