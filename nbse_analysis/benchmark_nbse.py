#!/usr/bin/env python3
import os
import subprocess
import time
import sys
import multiprocessing
from zipfile import ZipFile
from dataclasses import dataclass
from copy import copy

import requests

from os import listdir
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdShapeHelpers
from rdkit.Chem import rdmolops

from nbse_analysis.chem import sdf_to_smiles, sdf_to_mol
from nbse_analysis.list import flatten

target = 'Release'
verbose = True
fast = True
num_conformers_multi = ([10, 20, 40] if fast else [40, 80, 120])
num_conformers = 70
threads = multiprocessing.cpu_count()
pwd = os.getcwd()

nbse_dir = '../nbse'


def download_nbse():
    nbse_address = 'https://fiona.uni-hamburg.de/37ebe22f/nbse.zip'
    os.makedirs(nbse_dir, exist_ok=True)

    print("downloading nbse archive. this can take a while ...")

    resp = requests.get(nbse_address)
    zip_filename = os.path.join(nbse_dir, "nbse.zip")

    with open(zip_filename, "wb") as zip_file:
        zip_file.write(resp.content)

    with ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(nbse_dir)

    print("extracted source files here: {}".format(nbse_dir))


if not os.path.isdir(nbse_dir):
    download_nbse()

coaler_bin = '/usr/local/bin/coaler'


def is_ligand_file(file: str) -> bool:
    return not file.startswith('benchmark_result') and not file.startswith("result") and file.endswith('.sdf')


def get_nbse_smiles(dir: str) -> list[tuple[str, str]]:
    files = [file for file in listdir(dir) if is_ligand_file(file)]
    return flatten([sdf_to_smiles(os.path.join(dir, file)) for file in files])


def get_nbse_mols(dir: str) -> list[AllChem.Mol]:
    files = [file for file in listdir(dir) if is_ligand_file(file)]
    return flatten([sdf_to_mol(os.path.join(dir, file)) for file in files])


@dataclass
class Result:
    name: str
    took: float
    conformers: int
    local_similarity: float
    siena_rmsd: float
    global_tanimoto: float
    avg_conformer_tanimoto_dist: float

    def header(self):
        return 'name\ttook\tconformers\tlocal_similarity\tavg_conformer_tanimoto_dist\tglobal_shape_tanimoto_dist\tsiena_rmsd'

    def __str__(self):
        return ('{}\t{}\t{}\t{}\t{}\t{}\t{}'
                .format(self.name, int(self.took), self.conformers, self.local_similarity,
                                          self.avg_conformer_tanimoto_dist, self.global_tanimoto, self.siena_rmsd))


def run_coaler(infile_name: str, outfile_name: str, conformers: int):
    cmd_args = [coaler_bin,
                '--input', infile_name,
                '--out', outfile_name,
                '--verbose', ('true' if verbose else 'false'),
                '--thread', str(threads),
                '--core', 'mcs',
                '--divide', 'true',
                '--conformers', str(num_conformers)]

    coaler = subprocess.Popen(cmd_args, stdout=(sys.stdout if verbose else subprocess.DEVNULL))
    coaler.communicate()


def create_input_smiles_file(dir: str):
    smiles = get_nbse_smiles(dir)
    with open(os.path.join(dir, 'benchmark_input.smi'), 'w') as f:
        for (smi, n) in smiles:
            f.write('{}\t{}\n'.format(smi, n))
        f.flush()


def benchmark_nbse_ensemble(name: str) -> Result:
    directory = os.path.join(nbse_dir, name, 'ligands')
    orig: list[Chem.Mol] = get_nbse_mols(directory)

    infile_name = os.path.join(directory, 'benchmark_input.smi')
    if not os.path.isfile(infile_name):
        create_input_smiles_file(directory)

    outfile_name = os.path.join(directory, "result_coaler.sdf")

    start = time.time()
    run_coaler(infile_name, outfile_name, num_conformers)
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

    local_similarity = 0.0
    avg_conformer_tanimoto_dist = 0.0

    for inp in orig:
        out = out_by_name[inp.GetProp("_Name")]
        inp = rdmolops.RemoveHs(inp, False, True, True)
        rdmolops.SanitizeMol(out)

        local_similarity += float(out.GetProp("_Score"))

        # we need a copy to align it to the original molecule in order to compute tanimoto shape similarity
        out_for_align = copy(out)
        print("aligned ligands with score: {}".format(AllChem.AlignMol(out_for_align, inp)))
        avg_conformer_tanimoto_dist += rdShapeHelpers.ShapeTanimotoDist(inp, out_for_align)

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

        match_in_ref = inp.GetSubstructMatch(out, False, False)
        if len(match_in_ref) == 0:
            print("could not find match between:\ninp: {}\nout: {}\n".format(AllChem.MolToSmiles(inp),
                                                                             AllChem.MolToSmiles(out)))

        for (in_prb, in_ref) in list(zip(range(len(match_in_ref)), list(match_in_ref))):
            atom_map[in_prb + offset_out] = in_ref + offset_in

    local_similarity /= len(out_by_name)
    avg_conformer_tanimoto_dist /= len(out_by_name)

    result_writer = Chem.SDWriter(os.path.join(directory, 'benchmark_result.sdf'))
    aligned = AllChem.AlignMol(merged_out, merged_in, -1, -1, list(atom_map.items()))

    tanimoto = rdShapeHelpers.ShapeTanimotoDist(merged_out, merged_in)

    merged_out.SetProp("_Name", "coaler_output")
    merged_in.SetProp("_Name", "siena_ligands")

    result_writer.write(merged_out)
    result_writer.write(merged_in)
    result_writer.close()

    res = Result(
        name=name,
        took=end - start,
        conformers=num_conformers,
        siena_rmsd=aligned,
        global_tanimoto=tanimoto,
        avg_conformer_tanimoto_dist=avg_conformer_tanimoto_dist,
        local_similarity=local_similarity,
    )

    return res


if __name__ == '__main__':
    reinvesitgate = ['2pqk', '1v48', '4ajn', '4hw2', '4ly9']
    done = ['1d0s', '2vke', '4ajn', '1u0z', '2w0v', '4c4f', '3id8', '4c4f', '3id8', '4nb6', '2zsd']
    working = ['1d0s', '2vke', '4ajn', '1u0z', '2w0v', '4c4f', '3id8', '4c4f', '3id8', '4nb6', '2zsd']

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
