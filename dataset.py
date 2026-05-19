"""
Generate 2D molecular graphs from SMILES using RDKit.
Extracts atom coordinates, bond connectivity, bond lengths, and bond angles.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from dataclasses import dataclass
from typing import Optional

RDLogger.DisableLog('rdApp.*')

from molecules import SMILES_LIST


@dataclass
class Molecule2D:
    name: str
    coords: np.ndarray        # (N, 2) atom xy positions, normalized
    atom_types: list          # length N, atomic symbols
    bonds: list               # list of (i, j) index pairs
    bond_lengths: np.ndarray  # length E, Euclidean distance for each bond
    bond_angles: np.ndarray   # length A, angle in degrees at each interior atom
    angle_triples: list       # list of (i, j, k) — angle at j between bonds i-j and j-k
    n_atoms: int


def compute_angle(p1, p2, p3):
    """Angle at p2 between vectors p2->p1 and p2->p3, in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def smiles_to_mol2d(name: str, smiles: str) -> Optional[Molecule2D]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  Could not parse: {name}")
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.Compute2DCoords(mol)
    if result != 0:
        print(f"  Could not compute 2D coords: {name}")
        return None
    mol = Chem.RemoveHs(mol)

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    # Raw 2D coords (z is always 0 in 2D layout)
    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y] for i in range(n_atoms)], dtype=np.float32)

    # Normalize: center at origin, scale so mean bond length = 1.0
    coords -= coords.mean(axis=0)

    atom_types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]

    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    # Bond lengths
    bond_lengths = np.array([
        np.linalg.norm(coords[i] - coords[j]) for i, j in bonds
    ], dtype=np.float32)

    # Normalize coords so mean bond length = 1.0
    mean_bl = bond_lengths.mean() if len(bond_lengths) > 0 else 1.0
    coords /= mean_bl
    bond_lengths /= mean_bl

    # Bond angles: for each atom with degree >= 2, enumerate all pairs of neighbors
    adjacency = {i: [] for i in range(n_atoms)}
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

    angle_triples = []
    bond_angles = []
    for center in range(n_atoms):
        neighbors = adjacency[center]
        if len(neighbors) < 2:
            continue
        for idx_a in range(len(neighbors)):
            for idx_b in range(idx_a + 1, len(neighbors)):
                i = neighbors[idx_a]
                k = neighbors[idx_b]
                angle = compute_angle(coords[i], coords[center], coords[k])
                angle_triples.append((i, center, k))
                bond_angles.append(angle)

    bond_angles = np.array(bond_angles, dtype=np.float32)

    return Molecule2D(
        name=name,
        coords=coords,
        atom_types=atom_types,
        bonds=bonds,
        bond_lengths=bond_lengths,
        bond_angles=bond_angles,
        angle_triples=angle_triples,
        n_atoms=n_atoms,
    )


def build_dataset(source: str = 'chembl') -> list[Molecule2D]:
    """
    source='chembl': load from chembl_mols.json (fetched from ChEMBL API)
    source='builtin': load from the hardcoded SMILES_LIST in molecules.py
    """
    if source == 'chembl':
        with open('chembl_mols.json') as f:
            entries = json.load(f)
        pairs = [(e['name'], e['smiles']) for e in entries]
    else:
        pairs = SMILES_LIST

    mols = []
    failed = 0
    for name, smiles in pairs:
        mol = smiles_to_mol2d(name, smiles)
        if mol is not None:
            mols.append(mol)
        else:
            failed += 1

    print(f"Loaded {len(mols)} molecules ({failed} failed 2D layout)")
    return mols


def print_stats(mols: list[Molecule2D]):
    sizes = [m.n_atoms for m in mols]
    all_bl = np.concatenate([m.bond_lengths for m in mols])
    all_ba = np.concatenate([m.bond_angles for m in mols if len(m.bond_angles) > 0])

    print(f"\n--- Dataset stats ---")
    print(f"Molecules:       {len(mols)}")
    print(f"Atom count:      min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    print(f"Bond lengths:    mean={all_bl.mean():.3f}, std={all_bl.std():.3f}  (normalized to ~1.0)")
    print(f"Bond angles:     mean={all_ba.mean():.1f}°, std={all_ba.std():.1f}°")
    print(f"  min={all_ba.min():.1f}°, max={all_ba.max():.1f}°")

    # Angle distribution — what geometries do we see?
    aromatic_range = all_ba[(all_ba > 55) & (all_ba < 65)]
    sp2_range = all_ba[(all_ba > 115) & (all_ba < 125)]
    sp3_range = all_ba[(all_ba > 104) & (all_ba < 115)]
    print(f"\n  ~60° (3-membered ring):  {len(aromatic_range)} angles")
    print(f"  ~109° (sp3 / 5-ring):    {len(sp3_range)} angles")
    print(f"  ~120° (sp2 / aromatic):  {len(sp2_range)} angles")


def plot_molecules(mols: list[Molecule2D], n=12, cols=4):
    """Draw a grid of molecules with their bond angle annotations."""
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    atom_colors = {
        'C': '#404040', 'N': '#3050F8', 'O': '#FF0D0D',
        'S': '#FFFF30', 'P': '#FF8000', 'F': '#90E050',
        'Cl': '#1FF01F', 'Br': '#A62929', 'default': '#808080'
    }

    for ax, mol in zip(axes, mols[:n]):
        coords = mol.coords
        # Draw bonds
        for i, j in mol.bonds:
            ax.plot([coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color='#888888', linewidth=1.5, zorder=1)
        # Draw atoms
        for idx, (sym, pos) in enumerate(zip(mol.atom_types, coords)):
            color = atom_colors.get(sym, atom_colors['default'])
            ax.scatter(*pos, c=color, s=120, zorder=2, edgecolors='white', linewidths=0.5)
            if sym != 'C':
                ax.text(pos[0], pos[1] + 0.15, sym, fontsize=5,
                        ha='center', va='bottom', color=color)

        ax.set_title(f"{mol.name}\n{mol.n_atoms} atoms, "
                     f"{len(mol.bonds)} bonds", fontsize=7)
        ax.set_aspect('equal')
        ax.axis('off')

    for ax in axes[len(mols):]:
        ax.axis('off')

    plt.suptitle("2D Molecular Geometries (RDKit layout, normalized)", fontsize=11)
    plt.tight_layout()
    plt.savefig("molecules_grid.png", dpi=150, bbox_inches='tight')
    print("Saved molecules_grid.png")
    plt.close()


def plot_constraint_distributions(mols: list[Molecule2D]):
    """Visualize the bond length and angle distributions — these are the physics constraints."""
    all_bl = np.concatenate([m.bond_lengths for m in mols])
    all_ba = np.concatenate([m.bond_angles for m in mols if len(m.bond_angles) > 0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(all_bl, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
    axes[0].axvline(1.0, color='red', linestyle='--', label='mean (normalized)')
    axes[0].set_xlabel("Bond length (normalized units)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Bond Length Distribution\n(all molecules, all bonds)")
    axes[0].legend()

    axes[1].hist(all_ba, bins=40, color='coral', edgecolor='white', linewidth=0.5)
    for angle, label in [(60, '60° (3-ring)'), (109.5, '109.5° (sp3)'), (120, '120° (sp2)')]:
        axes[1].axvline(angle, linestyle='--', alpha=0.7, label=label)
    axes[1].set_xlabel("Bond angle (degrees)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Bond Angle Distribution\n(all angles at all atoms with degree ≥ 2)")
    axes[1].legend(fontsize=8)

    plt.suptitle("Physics Constraints in the Dataset", fontsize=11)
    plt.tight_layout()
    plt.savefig("constraint_distributions.png", dpi=150, bbox_inches='tight')
    print("Saved constraint_distributions.png")
    plt.close()


if __name__ == "__main__":
    mols = build_dataset(source='chembl')
    print_stats(mols)
    plot_molecules(mols, n=24, cols=6)
    plot_constraint_distributions(mols)
