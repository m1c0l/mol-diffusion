"""
Reverse diffusion: generate 2D molecule layouts from noise.

Given a molecule's graph (atom types + bond connectivity),
start from random coordinates and run T denoising steps.
Evaluate generated geometry against ground truth constraint distributions.

The molecule plot shows a curated ladder of complexity: from 1-ring simple
molecules up to the most complex in the val set, so you can see how well
the model handles each tier.
"""

import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem

from dataset import build_dataset, compute_angle
from train import MolDataset, mol_to_pyg_with_bond_types
from model import GNNDenoiser
from noise import cosine_schedule, p_sample_step_x0


def load_model(path: str, device, hidden_dim=128, n_layers=4) -> GNNDenoiser:
    model = GNNDenoiser(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def sample_molecule(pyg_data, model, schedule, T: int, device) -> np.ndarray:
    """
    Run full reverse diffusion for one molecule.
    Returns generated coordinates: (N, 2) numpy array.
    """
    data = pyg_data.to(device)
    N = data.pos.shape[0]

    coords = torch.randn(N, 2, device=device)

    for t in range(T, 0, -1):
        t_tensor = torch.tensor([t], device=device)
        x0_pred = model(
            coords, data.atom_types,
            data.edge_index, data.edge_attr,
            t_tensor, batch=torch.zeros(N, dtype=torch.long, device=device)
        )
        coords = p_sample_step_x0(coords, x0_pred, t, schedule)

    return coords.cpu().numpy()


def ring_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetRingInfo().NumRings()


def compute_bond_lengths(coords: np.ndarray, bonds: list) -> np.ndarray:
    return np.array([np.linalg.norm(coords[i] - coords[j]) for i, j in bonds])


def compute_bond_angles(coords: np.ndarray, bonds: list, n_atoms: int) -> np.ndarray:
    from collections import defaultdict
    adj = defaultdict(list)
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    angles = []
    for center in range(n_atoms):
        nbrs = adj[center]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                angles.append(compute_angle(
                    coords[nbrs[a]], coords[center], coords[nbrs[b]]
                ))
    return np.array(angles)


def select_complexity_ladder(mols, smiles_list, val_idx, n_plot=12) -> list:
    """
    Pick n_plot molecules from val set spanning the complexity range:
    sorted by (ring_count, n_atoms), then evenly sampled.
    """
    scored = []
    for idx in val_idx:
        r = ring_count(smiles_list[idx])
        scored.append((r, mols[idx].n_atoms, idx))
    scored.sort()

    # Evenly space across the sorted list to get a complexity ladder
    positions = np.linspace(0, len(scored) - 1, n_plot, dtype=int)
    return [scored[p][2] for p in positions]


def evaluate(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    schedule = {k: v.to(device) for k, v in cosine_schedule(args.T).items()}
    model = load_model(args.checkpoint, device,
                       hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    mols = build_dataset(source='chembl')
    with open('chembl_mols.json') as f:
        entries = json.load(f)
    smiles_list = [e['smiles'] for e in entries[:len(mols)]]

    import random; random.seed(42)
    indices = list(range(len(mols)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    val_idx = indices[split:]

    # Stats: random sample from val set
    n_sample = min(args.n_sample, len(val_idx))
    sample_idx = val_idx[:n_sample]

    # Plot: curated complexity ladder
    plot_idx = select_complexity_ladder(mols, smiles_list, val_idx, n_plot=12)
    plot_set  = set(plot_idx)

    print(f"Sampling {n_sample} molecules (12 curated for plot)...")

    gt_bl, gen_bl = [], []
    gt_ba, gen_ba = [], []

    # Pre-generate the 12 plot molecules
    plot_results = {}
    for idx in plot_idx:
        mol    = mols[idx]
        smiles = smiles_list[idx]
        pyg    = mol_to_pyg_with_bond_types(mol, smiles)
        gen    = sample_molecule(pyg, model, schedule, args.T, device)
        bl     = compute_bond_lengths(gen, mol.bonds)
        if len(bl) > 0:
            gen /= bl.mean()
        plot_results[idx] = gen

    # Stats pass over random sample
    for i, idx in enumerate(sample_idx):
        mol    = mols[idx]
        smiles = smiles_list[idx]
        pyg    = mol_to_pyg_with_bond_types(mol, smiles)

        if idx in plot_results:
            gen_coords = plot_results[idx]
        else:
            gen_coords = sample_molecule(pyg, model, schedule, args.T, device)
            bl_gen = compute_bond_lengths(gen_coords, mol.bonds)
            if len(bl_gen) > 0:
                gen_coords /= bl_gen.mean()

        bl_gen = compute_bond_lengths(gen_coords, mol.bonds)
        gt_bl.extend(mol.bond_lengths.tolist())
        gen_bl.extend(bl_gen.tolist())

        ba_gt  = compute_bond_angles(mol.coords,  mol.bonds, mol.n_atoms)
        ba_gen = compute_bond_angles(gen_coords,  mol.bonds, mol.n_atoms)
        gt_ba.extend(ba_gt.tolist())
        gen_ba.extend(ba_gen.tolist())

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_sample}")

    # ── Complexity ladder plot ────────────────────────────────────────────────
    fig_mols, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()

    for plot_count, idx in enumerate(plot_idx):
        mol       = mols[idx]
        gen_coords = plot_results[idx]
        c_gt  = mol.coords  - mol.coords.mean(0)
        c_gen = gen_coords  - gen_coords.mean(0)

        ax = axes[plot_count]
        for i_b, j_b in mol.bonds:
            ax.plot([c_gt[i_b,0],  c_gt[j_b,0]],  [c_gt[i_b,1],  c_gt[j_b,1]],
                    color='steelblue', alpha=0.6, lw=1.5)
            ax.plot([c_gen[i_b,0], c_gen[j_b,0]], [c_gen[i_b,1], c_gen[j_b,1]],
                    color='coral', alpha=0.6, lw=1.5, linestyle='--')
        ax.scatter(*c_gt.T,  s=40, c='steelblue', zorder=3, label='RDKit')
        ax.scatter(*c_gen.T, s=40, c='coral',     zorder=3, label='Generated', marker='^')

        r = ring_count(smiles_list[idx])
        ax.set_title(f"{mol.name[:14]}  ({mol.n_atoms}a, {r}r)", fontsize=7)
        ax.set_aspect('equal'); ax.axis('off')
        if plot_count == 0:
            ax.legend(fontsize=6, loc='upper right')

    plt.suptitle("Complexity ladder: simple → complex  |  Blue=RDKit  /  Coral=Generated",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig('generated_molecules.png', dpi=150)
    plt.close()
    print("Saved generated_molecules.png")

    # ── Constraint distribution comparison ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].hist(gt_bl,  bins=50, alpha=0.6, label='RDKit (ground truth)', color='steelblue')
    axes[0,0].hist(gen_bl, bins=50, alpha=0.6, label='Generated',            color='coral')
    axes[0,0].set_title('Bond lengths'); axes[0,0].legend()
    axes[0,0].set_xlabel('Normalized bond length')

    axes[0,1].hist(gt_ba,  bins=60, alpha=0.6, label='RDKit', color='steelblue', range=(0,180))
    axes[0,1].hist(gen_ba, bins=60, alpha=0.6, label='Generated', color='coral', range=(0,180))
    for a in [60, 109.5, 120]:
        axes[0,1].axvline(a, color='gray', linestyle=':', alpha=0.5)
    axes[0,1].set_title('Bond angles'); axes[0,1].legend()
    axes[0,1].set_xlabel('Degrees')

    bl_gt_s  = np.sort(gt_bl);  bl_gen_s = np.sort(gen_bl)
    axes[1,0].plot(bl_gt_s,  np.linspace(0,1,len(bl_gt_s)),  label='RDKit', color='steelblue')
    axes[1,0].plot(bl_gen_s, np.linspace(0,1,len(bl_gen_s)), label='Generated', color='coral')
    axes[1,0].set_title('Bond length CDF'); axes[1,0].legend()
    axes[1,0].set_xlabel('Normalized bond length')

    ba_gt_s  = np.sort(gt_ba);  ba_gen_s = np.sort(gen_ba)
    axes[1,1].plot(ba_gt_s,  np.linspace(0,1,len(ba_gt_s)),  label='RDKit', color='steelblue')
    axes[1,1].plot(ba_gen_s, np.linspace(0,1,len(ba_gen_s)), label='Generated', color='coral')
    axes[1,1].set_title('Bond angle CDF'); axes[1,1].legend()
    axes[1,1].set_xlabel('Degrees')

    plt.suptitle('Generated vs Ground Truth Constraint Distributions', fontsize=11)
    plt.tight_layout()
    plt.savefig('constraint_comparison.png', dpi=150)
    plt.close()
    print("Saved constraint_comparison.png")

    print(f"\n{'':30s}  {'Ground truth':>15}  {'Generated':>12}")
    print(f"  {'Bond length mean':30s}  {np.mean(gt_bl):15.3f}  {np.mean(gen_bl):12.3f}")
    print(f"  {'Bond length std':30s}  {np.std(gt_bl):15.3f}  {np.std(gen_bl):12.3f}")
    print(f"  {'Bond angle mean (°)':30s}  {np.mean(gt_ba):15.1f}  {np.mean(gen_ba):12.1f}")
    print(f"  {'Bond angle std (°)':30s}  {np.std(gt_ba):15.1f}  {np.std(gen_ba):12.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='best_model.pt')
    parser.add_argument('--n_sample',   type=int,   default=200)
    parser.add_argument('--T',          type=int,   default=200)
    parser.add_argument('--hidden_dim', type=int,   default=128)
    parser.add_argument('--n_layers',   type=int,   default=4)
    args = parser.parse_args()
    evaluate(args)
