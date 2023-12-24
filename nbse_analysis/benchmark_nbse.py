#!/usr/bin/env python3
import os
import subprocess
import time
from dataclasses import dataclass

import sys
import tempfile
from os import listdir
from rdkit import Chem
from rdkit.Chem import AllChem

target = 'Release'
verbose = True
num_conformers = 40
threads = 16
pwd = os.getcwd()
nbse_dir = '../nbse'
coaler_bin = '/home/niklas/projects/uni/coaler/build/{}/src/coaler'.format(target)


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_nbse(file: str) -> list[AllChem.Mol]:
    mols: list[AllChem.Mol] = []
    suppl = Chem.SDMolSupplier(file, sanitize=False, removeHs=True)
    for mol in suppl:
        mol: AllChem.Mol
        mols.append(mol)

    return mols


def sdf_to_smiles(file: str) -> list[tuple[str, str]]:
    smiles = []
    suppl = Chem.SDMolSupplier(file, sanitize=False, removeHs=True)
    for mol in suppl:
        mol: AllChem.Mol
        smiles.append((Chem.MolToSmiles(mol), mol.GetProp("_Name")))

    return smiles


def get_nbse_smiles(dir: str) -> list[tuple[str, str]]:
    files = [file for file in listdir(dir) if
             not file.startswith('benchmark_result') and not file.startswith("result") and file.endswith('.sdf')]
    return flatten([sdf_to_smiles(os.path.join(dir, file)) for file in files])


def get_nbse_mols(dir: str) -> list[AllChem.Mol]:
    files = [file for file in listdir(dir) if
             not file.startswith('benchmark_result') and not file.startswith("result") and file.endswith('.sdf')]
    return flatten([get_nbse(os.path.join(dir, file)) for file in files])


@dataclass
class Result:
    name: str
    took: float
    conformers: int
    local_score: float
    siena_score: float
    avg_conformer_score: float

    def header(self):
        return 'name,took,conformers,local_similarity,avg_conformer_rmsd,siena_rmsd'

    def __str__(self):
        return '{},{},{},{},{},{}'.format(self.name, self.took, self.conformers, self.local_score,
                                          self.avg_conformer_score, self.siena_score)


def substruct(mol: AllChem.Mol, query: AllChem.Mol) -> list[tuple[int, int]]:
    s = mol.GetSubstructMatch(query)
    assert(len(s) > 0)
    return list(zip(range(len(s)), list(s)))



def benchmark_nbse_ensemble(name: str) -> Result:
    directory = os.path.join(nbse_dir, name, 'ligands')
    orig: list[Chem.Mol] = get_nbse_mols(directory)

    in_file = os.path.join(directory, 'benchmark_input.smi')
    if not os.path.isfile(in_file):
        smiles = get_nbse_smiles(directory)
        with open(os.path.join(directory, 'benchmark_input.smi'), 'w') as f:
            for (smi, n) in smiles:
                f.write('{}\t{}\n'.format(smi, n))
            f.flush()

    outfile_name = os.path.join(directory, "result_coaler.sdf")
    cmd_args = [coaler_bin,
                '-f', in_file,
                '-o', outfile_name,
                '-j', str(threads),
                '--divide', 'true',
                '--conformers', str(num_conformers)]

    start = time.time()

    coaler = subprocess.Popen(cmd_args, stdout=(sys.stdout if verbose else subprocess.DEVNULL))
    coaler.communicate()

    end = time.time()

    out_mol_suppl = Chem.SDMolSupplier(outfile_name, sanitize=False, removeHs=True)
    out_by_name: dict[str, Chem.Mol] = {}
    mol: Chem.Mol
    for mol in out_mol_suppl:
        out_by_name[mol.GetProp("_Name")] = mol

    atom_map: dict[int, int] = {}

    merged_in: Chem.Mol | None = None
    merged_out: Chem.Mol | None = None

    inp: AllChem.Mol
    out: AllChem.Mol

    for inp in orig:
        out = out_by_name[inp.GetProp("_Name")]

        offset_in = 0
        offset_out = 0
        if merged_in is None:
            merged_in = inp
            merged_out = out
        else:
            offset_in = merged_in.GetNumAtoms()
            offset_out = merged_out.GetNumAtoms()

            merged_in = Chem.CombineMols(merged_in, inp)
            merged_out = Chem.CombineMols(merged_out, out)

        AllChem.MolToSmiles(inp)
        AllChem.MolToSmiles(out)

        match_in_ref = inp.GetSubstructMatch(out)

        for (in_prb, in_ref) in list(zip(range(len(match_in_ref)), list(match_in_ref))):
            atom_map[in_prb + offset_out] = in_ref + offset_in

    result_writer = Chem.SDWriter(os.path.join(directory, 'benchmark_result.sdf'))
    aligned = AllChem.AlignMol(merged_out, merged_in, -1, -1, list(atom_map.items()))
    print("global alignment score: {}".format(aligned))

    merged_out.SetProp("_Name", "coaler_output")
    merged_in.SetProp("_Name", "siena_ligands")

    result_writer.write(merged_out)
    result_writer.write(merged_in)
    result_writer.close()

    res = Result(
        name=name,
        took=end - start,
        conformers=num_conformers,
        siena_score=aligned,
        avg_conformer_score=0,
        local_score=list(out_by_name.values())[0].GetProp("_Score"),
    )

    print(str(res))

    return res


if __name__ == '__main__':
    reinvesitgate = ['2pqk', '1v48', '4ajn', '4hw2', '4ly9']
    done = ['1d0s', '4ajn', '1u0z', '2w0v', '4c4f', '3id8', '4c4f', '3id8', '4nb6', '2zsd']
    working = ['4ajn', '1u0z', '2w0v', '4c4f', '3id8', '4c4f', '3id8', '4nb6', '2zsd']

    results = []
    for name in working:
        print("running: {}".format(name))
        results.append(benchmark_nbse_ensemble(name))

    results_csv = None
    if not os.path.isfile('benchmark_results.csv'):
        results_csv = open('benchmark_results.csv', 'w')
        results_csv.write(Result.header(results[0]) + '\n')
    else:
        results_csv = open('benchmark_results.csv', 'a')

    for result in results:
        results_csv.write(str(result) + '\n')

    results_csv.close()

    print(results)
