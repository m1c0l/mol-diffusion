"""
Fetch approved small molecule drugs from ChEMBL, save as SMILES list.
Filters out molecules that are too large, too small, or have no valid SMILES.
"""

import json
import warnings
warnings.filterwarnings('ignore')

from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def fetch(min_atoms=4, max_atoms=50, target=15000, min_phase=1):
    print(f"Querying ChEMBL for small molecules with max_phase >= {min_phase}...")
    molecule = new_client.molecule
    results = molecule.filter(
        molecule_type='Small molecule',
        max_phase__gte=min_phase,
    ).only(['molecule_chembl_id', 'molecule_structures'])

    print(f"Total returned: {len(results)}")

    kept = []
    skipped_no_smiles = 0
    skipped_invalid = 0
    skipped_size = 0

    for entry in results:
        structs = entry.get('molecule_structures')
        if not structs:
            skipped_no_smiles += 1
            continue

        smiles = structs.get('canonical_smiles')
        if not smiles:
            skipped_no_smiles += 1
            continue

        # Validate with RDKit — also catches mixtures (.) which we skip
        if '.' in smiles:
            skipped_invalid += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped_invalid += 1
            continue

        # Remove Hs and check heavy atom count
        mol = Chem.RemoveHs(mol)
        n = mol.GetNumAtoms()
        if n < min_atoms or n > max_atoms:
            skipped_size += 1
            continue

        kept.append({
            'name': entry['molecule_chembl_id'],
            'smiles': smiles,
            'n_atoms': n,
        })

        if len(kept) >= target:
            break

    print(f"\nKept:              {len(kept)}")
    print(f"Skipped (no SMILES): {skipped_no_smiles}")
    print(f"Skipped (invalid):   {skipped_invalid}")
    print(f"Skipped (size):      {skipped_size}")

    # Atom count distribution summary
    sizes = [m['n_atoms'] for m in kept]
    import numpy as np
    print(f"\nAtom counts: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")

    return kept


if __name__ == '__main__':
    mols = fetch()
    out = 'chembl_mols.json'
    with open(out, 'w') as f:
        json.dump(mols, f)
    print(f"\nSaved {len(mols)} molecules to {out}")
