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
        if key in ('dataset_idx', 'ring_count_bucket'):
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('dataset_idx', 'ring_count_bucket'):
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
    bond_length_loss, angle_constraint_loss, ring_closure_loss,
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

    # Ring count bucket: 0=null(CFG), 1=0rings, 2=1ring, 3=2rings, 4=3+rings
    n_rings = rdmol.GetRingInfo().NumRings() if rdmol is not None else 0
    ring_bucket = min(n_rings, 3) + 1  # maps 0→1, 1→2, 2→3, 3+→4

    return MolData(
        pos=coords, atom_types=atom_types,
        edge_index=edge_index, edge_attr=edge_attr,
        n_atoms=torch.tensor(mol.n_atoms),
        angle_triples=angle_triples,
        tree_src=torch.tensor([e[0] for e in tree_edges], dtype=torch.long),
        tree_dst=torch.tensor([e[1] for e in tree_edges], dtype=torch.long),
        dataset_idx=torch.tensor([-1], dtype=torch.long),
        ring_count_bucket=torch.tensor([ring_bucket], dtype=torch.long),
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
    Unconditional EGNN denoiser. Base class for all diffusion modules.

    Subclass and override _cond_emb() to add conditioning.
    All training/validation logic, optimizer, and scheduler are inherited.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        tc, mc = cfg.training, cfg.model

        self.model = GNNDenoiser(hidden_dim=mc.hidden_dim,
                                 n_layers=mc.n_layers, t_dim=mc.t_dim,
                                 self_cond=mc.get('self_cond', False))

        schedule = cosine_schedule(tc.T)
        for k, v in schedule.items():
            self.register_buffer(k, v)

        if tc.resume:
            self.model.load_state_dict(torch.load(tc.resume, map_location='cpu'))
            print(f"Loaded weights from {tc.resume}")

        print(f"Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def on_fit_start(self):
        self.train_ds = self.trainer.datamodule.train_ds
        self.val_ds   = self.trainer.datamodule.val_ds

    def _cond_emb(self, batch):
        """Returns per-atom condition embedding (N, hidden_dim) or None.
        Override in subclasses to add conditioning."""
        return None

    def _run_denoiser(self, noisy_pos, batch, t_dev, cond):
        """
        Run the EGNN to predict clean coordinates x0.

        With self-conditioning disabled this is a single plain forward pass.

        With it enabled, we use the two-pass + stop-gradient scheme from
        Chen et al. 2022 ("Analog Bits"). The model can optionally be told its
        OWN previous x0 prediction; at inference the sampler naturally has last
        step's prediction to feed in, but training jumps to a random noise level
        so there is no previous step — we manufacture one:

          - Flip one coin per training step. ~50% of the time leave x0_prev = 0,
            which is exactly the cold-start the sampler hits on its first step, so
            the model must still work with no hint.
          - The other ~50%: do one extra forward under no_grad to get a realistic
            previous prediction, DETACH it (stop-gradient), then forward again
            conditioned on it. The loss is computed on this second pass only.

        The detach is what keeps it cheap and well-posed: we train the model to
        *use* a hint, not to backprop through the act of producing the hint, so
        there is no second backward pass (~1.5x forward cost, 1x backward).
        """
        def fwd(x0_prev):
            return self.model(noisy_pos, batch.atom_types, batch.edge_index,
                              batch.edge_attr, t_dev, batch=batch.batch,
                              cond_emb=cond, x0_prev=x0_prev)

        if not self.cfg.model.get('self_cond', False):
            return fwd(None)

        x0_prev = torch.zeros_like(noisy_pos)  # cold-start hint
        if torch.rand(()) < 0.5:
            with torch.no_grad():
                x0_prev = fwd(x0_prev)
            x0_prev = x0_prev.detach()         # stop-gradient: hint is a fixed input
        return fwd(x0_prev)

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
            cond   = self._cond_emb(batch)

            x0_pred = self._run_denoiser(noisy_pos, batch, t_dev, cond)
            loss = F.mse_loss(x0_pred, x0_target)

            if tc.lambda_constraint > 0:
                c = bond_length_loss(x0_pred, batch.edge_index)
                if batch.angle_triples.shape[0] > 0:
                    c = c + angle_constraint_loss(x0_pred, batch.angle_triples) / (180.0 ** 2)
                loss = loss + tc.lambda_constraint * c

            # Extra, separately-weighted pressure on ring-closure bonds — the
            # bonds the spanning tree only places implicitly, where rings fail.
            if tc.get('lambda_ring_closure', 0.0) > 0:
                rc = ring_closure_loss(x0_pred, batch.edge_index,
                                       batch.tree_src, batch.tree_dst)
                loss = loss + tc.lambda_ring_closure * rc
        else:
            batch      = batch.to(self.device)
            t_dev      = t_step.to(self.device)
            t_per_atom = t_dev[batch.batch]
            noise      = torch.randn_like(batch.pos)
            abar       = self.alpha_bar[t_per_atom].unsqueeze(-1)
            noisy_pos  = batch.pos * abar.sqrt() + noise * (1 - abar).sqrt()
            cond       = self._cond_emb(batch)

            pred_noise = self.model(noisy_pos, batch.atom_types,
                                    batch.edge_index, batch.edge_attr,
                                    t_dev, batch=batch.batch, cond_emb=cond)
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
        if not torch.isfinite(loss):
            return None  # skip batch; Lightning ignores None and doesn't update weights
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


class RingCondDiffusionModule(MolDiffusionModule):
    """
    Ring-count-conditioned EGNN denoiser with classifier-free guidance.

    Extends the unconditional module by injecting a ring count embedding into
    the EGNN feature track. During training, cfg_dropout% of molecules have
    their ring condition zeroed (null token), teaching the model to also work
    unconditionally — enabling CFG guidance at inference.

    Ring count buckets:  0=null(CFG), 1=0rings, 2=1ring, 3=2rings, 4=3+rings
    Saves full module state to ring_cond_model.pt (includes ring_emb weights).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.ring_emb = torch.nn.Embedding(5, cfg.model.hidden_dim)
        print(f"  (+ring_emb: {sum(p.numel() for p in self.ring_emb.parameters())} params)")

    def _cond_emb(self, batch):
        buckets = batch.ring_count_bucket  # (B,)
        if self.training and self.cfg.training.cfg_dropout > 0:
            drop = torch.bernoulli(
                torch.full((buckets.shape[0],), self.cfg.training.cfg_dropout,
                           device=buckets.device)
            ).bool()
            buckets = buckets.masked_fill(drop, 0)
        return self.ring_emb(buckets)[batch.batch]  # (N, hidden_dim)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _save_sampler_weights(pl_module, path, source_state=None):
    """Write weights in the exact format sample.load_model expects.

    Conditional models keep full module-style keys (model.* + ring_emb.*);
    unconditional models save GNNDenoiser-only keys (the 'model.' prefix stripped).
    `source_state` lets the EMA callback save its shadow weights instead of the
    live ones; when None we read the module's current weights.
    """
    if isinstance(pl_module, RingCondDiffusionModule):
        if source_state is None:
            source_state = pl_module.state_dict()      # includes ring_emb + buffers
        torch.save(dict(source_state), path)
    else:
        if source_state is None:
            source_state = pl_module.model.state_dict()  # already 'model.'-free
            torch.save(dict(source_state), path)
        else:
            # EMA shadow keys are module-style ('model.*'); strip to match GNNDenoiser.
            stripped = {k[len('model.'):]: v for k, v in source_state.items()
                        if k.startswith('model.')}
            torch.save(stripped, path)


class WeightCheckpoint(L.Callback):
    """Saves weights on val improvement. Saves full module state for conditional
    models (includes ring_emb etc.) and model-only state for unconditional.
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
            _save_sampler_weights(pl_module, self.path)


class EMACallback(L.Callback):
    """
    Exponential moving average (EMA) of model weights — standard in diffusion
    (DDPM, EDM). Keep a shadow copy of every trainable parameter, nudged toward
    the live weights after each optimizer step:

        shadow ← decay·shadow + (1 - decay)·weight

    Late in training the optimizer doesn't sit at a point — the weights bounce
    around inside the loss basin. The shadow averages over that trajectory and
    lands nearer its center, which reliably gives lower-variance, usually-better
    samples (and removes the "lucky epoch" noise we saw when picking checkpoints
    by a wobbling val loss).

    We swap the shadow weights in for validation (so val_loss reflects what we'll
    actually sample from) and save the best shadow weights to `path`. The live
    weights are restored afterwards so training continues normally; the separate
    full-resume Lightning checkpoint still stores the live weights + optimizer.
    """

    def __init__(self, decay: float, path: str):
        self.decay    = decay
        self.path     = path
        self.shadow   = {}              # param name -> averaged tensor
        self.backup   = {}              # param name -> live tensor (during val swap)
        self.best_val = float('inf')

    def on_fit_start(self, trainer, pl_module):
        # Seed the shadow from the current weights, before the pre-train sanity val.
        self.shadow = {n: p.detach().clone()
                       for n, p in pl_module.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        for n, p in pl_module.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def _swap_in(self, pl_module):
        """Stash live weights, load shadow weights into the module."""
        self.backup = {}
        for n, p in pl_module.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    def _swap_out(self, pl_module):
        """Restore the live weights stashed by _swap_in."""
        for n, p in pl_module.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking or not self.shadow:
            return
        self._swap_in(pl_module)        # validate on EMA weights

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self.backup:
            return
        val = trainer.callback_metrics.get('val_loss', float('inf'))
        if isinstance(val, torch.Tensor):
            val = val.item()
        if val < self.best_val:
            self.best_val = val
            # Save the shadow directly (independent of what's loaded in the module).
            _save_sampler_weights(pl_module, self.path, source_state=self.shadow)
        self._swap_out(pl_module)       # restore live weights for continued training


# ── Entry point ───────────────────────────────────────────────────────────────

_MODULE_CLASSES = {
    'unconditional': MolDiffusionModule,
    'ring_cond':     RingCondDiffusionModule,
}

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    tc  = cfg.training
    cls = _MODULE_CLASSES[cfg.model.get('module', 'unconditional')]

    dm     = MolDataModule(cfg)
    module = cls(cfg)

    accel = 'mps' if torch.backends.mps.is_available() else 'auto'

    ckpt_cb   = ModelCheckpoint(monitor='val_loss', mode='min',
                                filename='lightning_best', dirpath='.',
                                save_top_k=1, save_last=True, verbose=False)
    # When EMA is enabled the sampler weights come from the EMA shadow (averaged
    # over the training trajectory) instead of the raw live weights. EMACallback
    # is a drop-in replacement for WeightCheckpoint: same .path / .best_val API,
    # same "save best-by-val" behaviour, but it evaluates and saves the shadow.
    weight_cb = (EMACallback(tc.get('ema_decay', 0.999), tc.checkpoint)
                 if tc.get('ema', False) else WeightCheckpoint(tc.checkpoint))

    trainer = L.Trainer(
        max_epochs=tc.epochs,
        accelerator=accel,
        callbacks=[ckpt_cb, weight_cb],
        log_every_n_steps=1,
        enable_model_summary=False,
        gradient_clip_val=1.0,
    )

    trainer.fit(module, datamodule=dm, ckpt_path=tc.get('ckpt_path', None))

    print(f"\nBest val loss: {weight_cb.best_val:.4f}")
    print(f"Weights saved to {tc.checkpoint}")
    print(f"Lightning checkpoint: {ckpt_cb.best_model_path}")


if __name__ == '__main__':
    main()
