"""
Training entry point for the unconditional EGNN denoiser.

Structured as a PyTorch Lightning LightningModule + LightningDataModule,
configured via Hydra (config/config.yaml + experiment overrides).

  python train.py                                     # full training
  python train.py experiment=sanity                   # fast smoke test
  python train.py training.epochs=30 training.resume=best_model.pt
  python train.py training.ckpt_path=last.ckpt        # full Lightning resume
"""

import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig


# ── PyG Data subclass ────────────────────────────────────────────────────────

class MolData(Data):
    """Offsets custom index tensors correctly when PyG batches graphs."""
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('angle_triples', 'tree_src', 'tree_dst'):
            return self.num_nodes
        if key == 'dataset_idx':
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'dataset_idx':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import build_dataset, Molecule2D
from model import GNNDenoiser, atom_idx, BOND_VOCAB
from noise import cosine_schedule
from geom import (
    build_spanning_tree, to_internal, from_internal,
    bond_length_loss, angle_constraint_loss,
)


# ── Dataset helpers ──────────────────────────────────────────────────────────

def mol_to_pyg_with_bond_types(mol: Molecule2D, smiles: str) -> MolData:
    from rdkit import Chem
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is not None:
        rdmol = Chem.RemoveHs(rdmol)
        bond_type_map = {}
        for bond in rdmol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            idx = {1.0: 0, 2.0: 1, 3.0: 2}.get(bond.GetBondTypeAsDouble(), 3)
            bond_type_map[(i, j)] = bond_type_map[(j, i)] = idx
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
                src_list.append(s); dst_list.append(d); attr_list.append(one_hot)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.stack(attr_list)

    tree_edges    = build_spanning_tree(mol.bonds, mol.n_atoms)
    angle_triples = (torch.tensor(mol.angle_triples, dtype=torch.long)
                     if mol.angle_triples else torch.zeros(0, 3, dtype=torch.long))

    return MolData(
        pos=coords, atom_types=atom_types,
        edge_index=edge_index, edge_attr=edge_attr,
        n_atoms=torch.tensor(mol.n_atoms),
        angle_triples=angle_triples,
        tree_src=torch.tensor([e[0] for e in tree_edges], dtype=torch.long),
        tree_dst=torch.tensor([e[1] for e in tree_edges], dtype=torch.long),
        dataset_idx=torch.tensor([-1], dtype=torch.long),
    )


class MolDataset(Dataset):
    def __init__(self, mols: list[Molecule2D], smiles_list: list[str]):
        self.data = [mol_to_pyg_with_bond_types(m, s) for m, s in zip(mols, smiles_list)]
        for i, d in enumerate(self.data):
            d.dataset_idx = torch.tensor([i], dtype=torch.long)

        self.internals = []
        for mol, d in zip(mols, self.data):
            tree_edges = list(zip(d.tree_src.tolist(), d.tree_dst.tolist()))
            if tree_edges:
                lengths, angles, root = to_internal(mol.coords, tree_edges)
            else:
                lengths = np.zeros(0, dtype=np.float32)
                angles  = np.zeros(0, dtype=np.float32)
                root    = mol.coords[0].copy().astype(np.float32)
            self.internals.append((lengths, angles, root, tree_edges))
        self.coords = [m.coords for m in mols]

    def __len__(self):  return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


# ── Corruption helpers ────────────────────────────────────────────────────────

def internal_corrupt(internals_list, clean_coords_list, t_vals, alpha_bar_np):
    noisy_list, clean_list = [], []
    for (lengths, angles, root, tree_edges), coords, t in zip(
            internals_list, clean_coords_list, t_vals):
        abar         = float(alpha_bar_np[t])
        sqrt_abar    = np.sqrt(abar)
        sqrt_1m_abar = np.sqrt(1.0 - abar)
        if not tree_edges:
            eps   = np.random.randn(*coords.shape).astype(np.float32)
            noisy = sqrt_abar * coords + sqrt_1m_abar * eps
        else:
            n = len(lengths)
            eps_l = np.random.randn(n).astype(np.float32)
            eps_a = np.random.randn(n).astype(np.float32)
            eps_r = np.random.randn(2).astype(np.float32)
            noisy_lengths = np.maximum(sqrt_abar * lengths + sqrt_1m_abar * eps_l, 0.01)
            noisy_angles  = sqrt_abar * angles  + sqrt_1m_abar * eps_a * np.pi
            noisy_root    = sqrt_abar * root    + sqrt_1m_abar * eps_r
            noisy = from_internal(noisy_lengths, noisy_angles,
                                  tree_edges, noisy_root, len(coords))
        noisy_list.append(noisy)
        clean_list.append(coords.astype(np.float32))
    return noisy_list, clean_list


# ── Lightning DataModule ──────────────────────────────────────────────────────

class MolDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds: MolDataset | None = None
        self.val_ds:   MolDataset | None = None

    def setup(self, stage=None):
        source = self.cfg.dataset.source
        mols   = build_dataset(source=source)
        if source == 'chembl':
            with open('chembl_mols.json') as f:
                entries = json.load(f)
            smiles_list = [e['smiles'] for e in entries[:len(mols)]]
        else:
            from molecules import SMILES_LIST
            smiles_list = [s for _, s in SMILES_LIST[:len(mols)]]

        random.seed(42)
        indices  = list(range(len(mols)))
        random.shuffle(indices)
        split    = int(0.9 * len(indices))
        train_i, val_i = indices[:split], indices[split:]

        self.train_ds = MolDataset([mols[i] for i in train_i], [smiles_list[i] for i in train_i])
        self.val_ds   = MolDataset([mols[i] for i in val_i],   [smiles_list[i] for i in val_i])
        print(f"Train: {len(self.train_ds)}, Val: {len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.training.batch_size,
                          shuffle=True, collate_fn=Batch.from_data_list)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.training.batch_size,
                          shuffle=False, collate_fn=Batch.from_data_list)


# ── Lightning Module ──────────────────────────────────────────────────────────

class MolDiffusionModule(L.LightningModule):
    """
    EGNN denoiser wrapped as a LightningModule.

    Benefits over the plain train loop:
      - Device handling is automatic (MPS/CUDA/CPU)
      - ModelCheckpoint saves both Lightning .ckpt (full state: optimizer,
        scheduler, epoch) and a weight-only .pt for backward compat with sample.py
      - Resuming with training.ckpt_path restores optimizer + scheduler state,
        fixing the LR annealing regression we saw with weight-only resuming
      - training_step / validation_step are clean and symmetric
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        tc, mc = cfg.training, cfg.model

        self.model = GNNDenoiser(hidden_dim=mc.hidden_dim,
                                 n_layers=mc.n_layers, t_dim=mc.t_dim)

        schedule = cosine_schedule(tc.T)
        for k, v in schedule.items():
            self.register_buffer(k, v)  # moved to device automatically

        if tc.resume:
            self.model.load_state_dict(torch.load(tc.resume, map_location='cpu'))
            print(f"Loaded weights from {tc.resume}")

        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def on_fit_start(self):
        self.train_ds = self.trainer.datamodule.train_ds
        self.val_ds   = self.trainer.datamodule.val_ds

    def _forward(self, batch, ds):
        tc = self.cfg.training
        B  = batch.num_graphs
        t_step = torch.randint(1, tc.T + 1, (B,))

        if tc.use_internal:
            idx_list  = batch.dataset_idx.tolist()
            internals = [ds.internals[i] for i in idx_list]
            coords_l  = [ds.coords[i]    for i in idx_list]
            ab_np     = self.alpha_bar.cpu().numpy()
            noisy_list, clean_list = internal_corrupt(internals, coords_l,
                                                      t_step.tolist(), ab_np)
            noisy_pos = torch.from_numpy(
                np.concatenate(noisy_list, 0).astype(np.float32)).to(self.device)
            x0_target = torch.from_numpy(
                np.concatenate(clean_list, 0).astype(np.float32)).to(self.device)
            batch  = batch.to(self.device)
            t_dev  = t_step.to(self.device)

            x0_pred = self.model(noisy_pos, batch.atom_types,
                                 batch.edge_index, batch.edge_attr,
                                 t_dev, batch=batch.batch)
            loss = F.mse_loss(x0_pred, x0_target)

            if tc.lambda_constraint > 0:
                c = bond_length_loss(x0_pred, batch.edge_index)
                if batch.angle_triples.shape[0] > 0:
                    c = c + angle_constraint_loss(x0_pred, batch.angle_triples) / (180.0 ** 2)
                loss = loss + tc.lambda_constraint * c
        else:
            batch      = batch.to(self.device)
            t_dev      = t_step.to(self.device)
            t_per_atom = t_dev[batch.batch]
            noise      = torch.randn_like(batch.pos)
            abar       = self.alpha_bar[t_per_atom].unsqueeze(-1)
            noisy_pos  = batch.pos * abar.sqrt() + noise * (1 - abar).sqrt()

            pred_noise = self.model(noisy_pos, batch.atom_types,
                                    batch.edge_index, batch.edge_attr,
                                    t_dev, batch=batch.batch)
            loss = F.mse_loss(pred_noise, noise)

            if tc.lambda_constraint > 0:
                x0_hat = ((noisy_pos - pred_noise * (1 - abar).sqrt())
                          / (abar.sqrt() + 1e-8)).clamp(-10, 10)
                aw = self.alpha_bar[t_dev].mean()
                c  = bond_length_loss(x0_hat, batch.edge_index)
                if batch.angle_triples.shape[0] > 0:
                    c = c + angle_constraint_loss(x0_hat, batch.angle_triples) / (180.0 ** 2)
                loss = loss + tc.lambda_constraint * aw * c

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._forward(batch, self.train_ds)
        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward(batch, self.val_ds)
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        tc  = self.cfg.training
        opt = torch.optim.AdamW(self.parameters(), lr=tc.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=tc.lr_epochs, eta_min=tc.lr * 0.05)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sch, 'interval': 'epoch'}}


class WeightCheckpoint(L.Callback):
    """Saves model weights as a plain .pt file whenever val_loss improves.
    Keeps sample.py and the existing resume workflow working unchanged.
    """
    def __init__(self, path: str):
        self.path     = path
        self.best_val = float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get('val_loss', float('inf'))
        if isinstance(val, torch.Tensor):
            val = val.item()
        if val < self.best_val:
            self.best_val = val
            torch.save(pl_module.model.state_dict(), self.path)


# ── Entry point ───────────────────────────────────────────────────────────────

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    tc = cfg.training

    dm     = MolDataModule(cfg)
    module = MolDiffusionModule(cfg)

    accel = 'mps' if torch.backends.mps.is_available() else 'auto'

    ckpt_cb   = ModelCheckpoint(monitor='val_loss', mode='min',
                                filename='lightning_best', dirpath='.',
                                save_top_k=1, verbose=False)
    weight_cb = WeightCheckpoint(tc.checkpoint)

    trainer = L.Trainer(
        max_epochs=tc.epochs,
        accelerator=accel,
        callbacks=[ckpt_cb, weight_cb],
        log_every_n_steps=1,
        enable_model_summary=False,
    )

    ckpt_path = tc.get('ckpt_path', None)
    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)

    print(f"\nBest val loss: {weight_cb.best_val:.4f}")
    print(f"Weights saved to {tc.checkpoint}")
    print(f"Lightning checkpoint: {ckpt_cb.best_model_path}")


if __name__ == '__main__':
    main()
