
from rdkit.Chem import AllChem

def sdf_to_smiles(file: str) -> list[tuple[str, str]]:
    smiles = []
    suppl = AllChem.SDMolSupplier(file, sanitize=False, removeHs=True)
    for mol in suppl:
        mol: AllChem.Mol
        smiles.append((AllChem.MolToSmiles(mol), mol.GetProp("_Name")))

    return smiles

def sdf_to_mol(file: str) -> list[AllChem.Mol]:
    mols: list[AllChem.Mol] = []
    suppl = AllChem.SDMolSupplier(file, sanitize=False, removeHs=True)
    for mol in suppl:
        mol: AllChem.Mol
        mols.append(mol)

    return mols

def substruct(mol: AllChem.Mol, query: AllChem.Mol) -> list[tuple[int, int]]:
    s = mol.GetSubstructMatch(query)
    assert(len(s) > 0)
    return list(zip(range(len(s)), list(s)))
