import os
import pickle
import sys
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdFMCS

@dataclass
class Summary:
    name: str
    mols: list[Chem.Mol]
    mcs: rdFMCS.MCSResult

def sdf_to_mols(file: str) -> list[Chem.Mol]:
    mols = []
    suppl = Chem.SDMolSupplier(file)
    for mol in suppl:
        mols.append(mol)

    return mols

if __name__ == '__main__':
    results: list[Summary] = []
    for root, _, _ in os.walk('../nbse/'):
        set_path = os.path.join(root, 'ligands')
        set_name = os.path.split(root)[-1]

        for _, _, files in os.walk(set_path):
            ligand_mols: list[Chem.Mol] = []
            for f in files:
                ligand_path = os.path.join(set_path, f)
                set_mols = sdf_to_mols(ligand_path)
                ligand_mols.append(*set_mols)

            mcs: rdFMCS.MCSResult = rdFMCS.FindMCS(ligand_mols)

            results.append(Summary(set_name, ligand_mols, mcs))

    with open('results.pkl', 'wb') as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)















