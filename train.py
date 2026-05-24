"""
Training loop for the unconditional GNN denoiser.

Two modes controlled by flags:

  --use_internal         Diffuse in internal coordinate space (bond lengths +
                         bond angles) rather than raw Cartesian (x, y).

  --lambda_constraint    Weight for the constraint loss added on top of the
                         noise prediction MSE. Penalizes bond lengths != 1.0
                         and bond angles far from 60/109.5/120/180 degrees.

Both can be combined. The default (no flags) reproduces the original
Cartesian diffusion with no explicit constraint signal.
"""

import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


class MolData(Data):
    """PyG Data subclass that correctly offsets custom index tensors when batching.

    PyG auto-increments edge_index by num_nodes, but not custom tensors.
    angle_triples, tree_src, and tree_dst store atom indices → need same treatment.
    dataset_idx is a scalar identifier → must NOT be incremented.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('angle_triples', 'tree_src', 'tree_dst'):
            return self.num_nodes
        if key == 'dataset_idx':
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'dataset_idx':
            return 0  # stack as a 1-D tensor across batch
        return super().__cat_dim__(key, value, *args, **kwargs)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import build_dataset, Molecule2D
from model import GNNDenoiser, atom_idx, BOND_VOCAB
from noise import cosine_schedule
from geom import (
    build_spanning_tree, to_internal, from_internal, internal_q_sample,
    bond_length_loss, angle_constraint_loss,
)


# ── Dataset ──────────────────────────────────────────────────────────────────

def mol_to_pyg_with_bond_types(mol: Molecule2D, smiles: str) -> Data:
    from rdkit import Chem

    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is not None:
        rdmol = Chem.RemoveHs(rdmol)
        bond_type_map = {}
        for bond in rdmol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondTypeAsDouble()
            idx = {1.0: 0, 2.0: 1, 3.0: 2}.get(bt, 3)  # SINGLE/DOUBLE/TRIPLE/AROMATIC
            bond_type_map[(i, j)] = idx
            bond_type_map[(j, i)] = idx
    else:
        bond_type_map = {}

    coords     = torch.tensor(mol.coords,     dtype=torch.float32)
    atom_types = torch.tensor([atom_idx(s) for s in mol.atom_types], dtype=torch.long)

    if len(mol.bonds) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr  = torch.zeros(0, len(BOND_VOCAB), dtype=torch.float32)
    else:
        src_list, dst_list, attr_list = [], [], []
        for i, j in mol.bonds:
            one_hot = torch.zeros(len(BOND_VOCAB))
            one_hot[bond_type_map.get((i, j), 0)] = 1.0
            for s, d in [(i, j), (j, i)]:
                src_list.append(s)
                dst_list.append(d)
                attr_list.append(one_hot)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.stack(attr_list)

    # Spanning tree + angle triples — used for internal diffusion and constraint loss
    tree_edges = build_spanning_tree(mol.bonds, mol.n_atoms)

    # angle_triples: (A, 3) tensor of (i, center, k) indices
    if mol.angle_triples:
        angle_triples = torch.tensor(mol.angle_triples, dtype=torch.long)
    else:
        angle_triples = torch.zeros(0, 3, dtype=torch.long)

    return MolData(
        pos=coords,
        atom_types=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_atoms=torch.tensor(mol.n_atoms),
        angle_triples=angle_triples,
        tree_src=torch.tensor([e[0] for e in tree_edges], dtype=torch.long),
        tree_dst=torch.tensor([e[1] for e in tree_edges], dtype=torch.long),
        dataset_idx=torch.tensor([-1], dtype=torch.long),  # filled in by MolDataset
    )


class MolDataset(Dataset):
    def __init__(self, mols: list[Molecule2D], smiles_list: list[str]):
        self.data = [
            mol_to_pyg_with_bond_types(m, s)
            for m, s in zip(mols, smiles_list)
        ]
        # Stamp each Data object with its dataset index for fast lookup in training.
        for i, d in enumerate(self.data):
            d.dataset_idx = torch.tensor([i], dtype=torch.long)

        # Pre-cache internal coordinates (lengths, angles, root) per molecule.
        # Computed once at dataset build time so the training loop only does
        # fast vectorized NumPy noise ops, not Python-level geometry computation.
        self.internals = []
        for mol, d in zip(mols, self.data):
            tree_edges = list(zip(d.tree_src.tolist(), d.tree_dst.tolist()))
            if len(tree_edges) > 0:
                lengths, angles, root = to_internal(mol.coords, tree_edges)
            else:
                lengths = np.zeros(0, dtype=np.float32)
                angles  = np.zeros(0, dtype=np.float32)
                root    = mol.coords[0].copy().astype(np.float32)
            self.internals.append((lengths, angles, root, tree_edges))
        self.coords = [m.coords for m in mols]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Corruption helpers ────────────────────────────────────────────────────────

def cartesian_corrupt(pos, t_per_atom, schedule):
    """Standard Cartesian noise. Returns (noisy_pos, noise)."""
    noise    = torch.randn_like(pos)
    abar     = schedule['alpha_bar'][t_per_atom].unsqueeze(-1)
    noisy    = pos * abar.sqrt() + noise * (1 - abar).sqrt()
    return noisy, noise


def internal_corrupt(internals_list, clean_coords_list, t_vals, alpha_bar_np):
    """
    Corrupt molecules using pre-cached internal coordinates.
    Vectorized per molecule: all bond lengths and angles are noised in one
    NumPy call instead of looping per-bond, then Cartesian is reconstructed.

    Returns (noisy_pos_list, clean_pos_list).
    """
    noisy_list, clean_list = [], []
    for (lengths, angles, root, tree_edges), coords, t in zip(
            internals_list, clean_coords_list, t_vals):

        abar         = float(alpha_bar_np[t])
        sqrt_abar    = np.sqrt(abar)
        sqrt_1m_abar = np.sqrt(1.0 - abar)

        if len(tree_edges) == 0:
            noise = np.random.randn(*coords.shape).astype(np.float32)
            noisy = sqrt_abar * coords + sqrt_1m_abar * noise
        else:
            # Vectorized: noise all lengths and angles in one call
            n_edges = len(lengths)
            eps_l = np.random.randn(n_edges).astype(np.float32)
            eps_a = np.random.randn(n_edges).astype(np.float32)
            eps_r = np.random.randn(2).astype(np.float32)

            noisy_lengths = np.maximum(sqrt_abar * lengths + sqrt_1m_abar * eps_l, 0.01)
            noisy_angles  = sqrt_abar * angles  + sqrt_1m_abar * eps_a * np.pi
            noisy_root    = sqrt_abar * root    + sqrt_1m_abar * eps_r

            noisy = from_internal(noisy_lengths, noisy_angles,
                                   tree_edges, noisy_root, len(coords))

        noisy_list.append(noisy)
        clean_list.append(coords.astype(np.float32))

    return noisy_list, clean_list


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Internal coord diffusion: {args.use_internal}")
    print(f"Constraint loss weight:   {args.lambda_constraint}")

    print("Loading dataset...")
    mols = build_dataset(source='chembl')
    with open('chembl_mols.json') as f:
        entries = json.load(f)
    smiles_list = [e['smiles'] for e in entries[:len(mols)]]

    random.seed(42)
    indices = list(range(len(mols)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = MolDataset([mols[i] for i in train_idx], [smiles_list[i] for i in train_idx])
    val_ds   = MolDataset([mols[i] for i in val_idx],   [smiles_list[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=Batch.from_data_list)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=Batch.from_data_list)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    schedule     = cosine_schedule(args.T)
    schedule_dev = {k: v.to(device) for k, v in schedule.items()}
    alpha_bar_np = schedule['alpha_bar'].numpy()  # CPU numpy for internal_q_sample

    model = GNNDenoiser(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.lr_epochs, eta_min=args.lr * 0.05
    )

    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            B = batch.num_graphs
            t = torch.randint(1, args.T + 1, (B,))  # keep on CPU for indexing

            if args.use_internal:
                # Use stamped dataset_idx to look up pre-cached internals —
                # avoids to_data_list() overhead and per-molecule geometry recomputation.
                idx_list   = batch.dataset_idx.tolist()
                internals  = [train_ds.internals[i] for i in idx_list]
                coords_l   = [train_ds.coords[i]    for i in idx_list]
                noisy_list, clean_list = internal_corrupt(internals, coords_l, t.tolist(), alpha_bar_np)
                noisy_pos  = torch.from_numpy(np.concatenate(noisy_list, 0)).to(device)
                x0_target  = torch.from_numpy(np.concatenate(clean_list,  0)).to(device)
                batch      = batch.to(device)
                t          = t.to(device)

                x0_pred = model(noisy_pos, batch.atom_types,
                                batch.edge_index, batch.edge_attr,
                                t, batch=batch.batch)
                loss = F.mse_loss(x0_pred, x0_target)

                if args.lambda_constraint > 0:
                    c_loss = bond_length_loss(x0_pred, batch.edge_index)
                    if batch.angle_triples.shape[0] > 0:
                        # Normalize angle loss to ~[0,1] (divide by max squared degree error)
                        c_loss = c_loss + angle_constraint_loss(x0_pred, batch.angle_triples) / (180.0 ** 2)
                    loss = loss + args.lambda_constraint * c_loss
            else:
                batch      = batch.to(device)
                t_per_atom = t.to(device)[batch.batch]
                noise      = torch.randn_like(batch.pos)
                abar       = schedule_dev['alpha_bar'][t_per_atom].unsqueeze(-1)
                noisy_pos  = batch.pos * abar.sqrt() + noise * (1 - abar).sqrt()
                t          = t.to(device)

                pred_noise = model(noisy_pos, batch.atom_types,
                                   batch.edge_index, batch.edge_attr,
                                   t, batch=batch.batch)
                loss = F.mse_loss(pred_noise, noise)

                if args.lambda_constraint > 0:
                    t_per_atom  = t[batch.batch]
                    abar        = schedule_dev['alpha_bar'][t_per_atom].unsqueeze(-1)
                    x0_hat      = (noisy_pos - pred_noise * (1 - abar).sqrt()) / (abar.sqrt() + 1e-8)
                    x0_hat      = x0_hat.clamp(-10, 10)  # guard against t≈T explosion
                    # Weight by mean ᾱ: constraint signal is only meaningful at low noise
                    abar_weight = schedule_dev['alpha_bar'][t].mean()
                    c_loss      = bond_length_loss(x0_hat, batch.edge_index)
                    if batch.angle_triples.shape[0] > 0:
                        c_loss = c_loss + angle_constraint_loss(x0_hat, batch.angle_triples) / (180.0 ** 2)
                    loss = loss + args.lambda_constraint * abar_weight * c_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * B

        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                B = batch.num_graphs
                t = torch.randint(1, args.T + 1, (B,))

                if args.use_internal:
                    idx_list   = batch.dataset_idx.tolist()
                    internals  = [val_ds.internals[i] for i in idx_list]
                    coords_l   = [val_ds.coords[i]    for i in idx_list]
                    noisy_list, clean_list = internal_corrupt(internals, coords_l, t.tolist(), alpha_bar_np)
                    noisy_pos  = torch.from_numpy(np.concatenate(noisy_list, 0)).to(device)
                    x0_target  = torch.from_numpy(np.concatenate(clean_list,  0)).to(device)
                    batch      = batch.to(device)
                    t          = t.to(device)
                    x0_pred    = model(noisy_pos, batch.atom_types,
                                       batch.edge_index, batch.edge_attr,
                                       t, batch=batch.batch)
                    val_loss  += F.mse_loss(x0_pred, x0_target).item() * B
                else:
                    batch      = batch.to(device)
                    t_per_atom = t.to(device)[batch.batch]
                    noise      = torch.randn_like(batch.pos)
                    abar       = schedule_dev['alpha_bar'][t_per_atom].unsqueeze(-1)
                    noisy_pos  = batch.pos * abar.sqrt() + noise * (1 - abar).sqrt()
                    t          = t.to(device)
                    pred_noise = model(noisy_pos, batch.atom_types,
                                       batch.edge_index, batch.edge_attr,
                                       t, batch=batch.batch)
                    val_loss  += F.mse_loss(pred_noise, noise).item() * B

        val_loss /= len(val_ds)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={epoch_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='train')
    ax.plot(val_losses,   label='val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'Denoising loss  (internal={args.use_internal}, λ={args.lambda_constraint})')
    ax.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    plt.close()
    print(f"\nBest val loss: {best_val:.4f}")
    print("Saved best_model.pt and loss_curve.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',            type=int,   default=150)
    parser.add_argument('--batch_size',        type=int,   default=32)
    parser.add_argument('--hidden_dim',        type=int,   default=128)
    parser.add_argument('--n_layers',          type=int,   default=4)
    parser.add_argument('--lr',                type=float, default=3e-4)
    parser.add_argument('--T',                 type=int,   default=200)
    parser.add_argument('--use_internal',      action='store_true',
                        help='Diffuse in internal coordinate space')
    parser.add_argument('--lambda_constraint', type=float, default=0.0,
                        help='Weight for bond length + angle constraint loss')
    parser.add_argument('--resume',            type=str,   default=None,
                        help='Path to checkpoint to resume from (e.g. best_model.pt)')
    parser.add_argument('--lr_epochs',         type=int,   default=200,
                        help='T_max for cosine LR schedule (independent of --epochs)')
    args = parser.parse_args()
    train(args)
