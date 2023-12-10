import os
from rdkit import Chem
from rdkit.Chem import rdFMCS


def sdf_to_mols(file: str) -> list[Chem.Mol]:
    mols = []
    suppl = Chem.SDMolSupplier(file)
    for mol in suppl:
        mols.append(mol)

    return mols


if __name__ == "__main__":
    with open('results_mcs.csv', 'w') as results:
        results.write('name\tmcs_atoms\tmcs_bonds\tmcs_smarts\n')
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

                results.write(
                    '{}\t{}\t{}\t{}\n'.format(
                       set_name, mcs.numAtoms, mcs.numBonds, mcs.smartsString
                    ))











