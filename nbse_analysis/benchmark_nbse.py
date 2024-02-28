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

verbose = True
fast = False
threads = multiprocessing.cpu_count()
pwd = os.getcwd()
# timeout in seconds
timeout = 60 * 15

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
    return not file.startswith('benchmark_result') and not file.startswith(
        "result") and file.endswith('.sdf')


def get_nbse_smiles(dir: str) -> list[tuple[str, str]]:
    files = [file for file in listdir(dir) if is_ligand_file(file)]
    return flatten([sdf_to_smiles(os.path.join(dir, file)) for file in files])


def get_nbse_mols(dir: str) -> list[AllChem.Mol]:
    files = [file for file in listdir(dir) if is_ligand_file(file)]
    return flatten([sdf_to_mol(os.path.join(dir, file)) for file in files])


@dataclass
class Config:
    optimizer_coarse: float = 0.5
    optimizer_fine: float = 0.01
    num_conformers: int = 20
    assemblies: int = int(threads / 2)
    divide: bool = True
    core: str = 'mcs'
    optimizer_steps: int = 100

    def to_args(self) -> list[str]:
        return [
            '--assemblies', str(self.assemblies),
            '--core', self.core,
            '--divide', str(self.divide),
            '--conformers', str(self.num_conformers),
            '--optimizer-coarse-threshold', str(self.optimizer_coarse),
            '--optimizer-fine-threshold', str(self.optimizer_fine),
            '--optimizer-step-limit', str(self.optimizer_steps),
        ]

    @staticmethod
    def csv_header() -> str:
        return 'optimizer_coarse\toptimizer_fine\tnum_conformers\tassemblies\tdivide\tcore\toptimizer_steps'

    def __str__(self) -> str:
        return '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(self.optimizer_coarse,
                                                   self.optimizer_fine,
                                                   self.num_conformers,
                                                   self.assemblies, self.divide,
                                                   self.core,
                                                   self.optimizer_steps)


@dataclass
class Result:
    name: str
    took: float
    local_similarity: float
    siena_rmsd: float
    avg_conformer_tanimoto_dist: float
    conf: Config

    @staticmethod
    def header():
        return 'name\ttook\tlocal_similarity\tavg_conformer_tanimoto_dist\tsiena_rmsd\t' + Config.csv_header()

    def __str__(self):
        return ('{}\t{}\t{}\t{}\t{}\t{}'
                .format(self.name, int(self.took), self.local_similarity,
                        self.avg_conformer_tanimoto_dist, self.siena_rmsd,
                        str(self.conf)))


def run_coaler(infile_name: str, outfile_name: str, config: Config):
    cmd_args = [coaler_bin,
                '--input', infile_name,
                '--out', outfile_name,
                '--verbose', 'false',
                '--thread', str(threads),
                *config.to_args()]

    print(cmd_args)

    coaler = subprocess.Popen(cmd_args, stdout=(
        sys.stdout if verbose else subprocess.DEVNULL))
    coaler.communicate(timeout=timeout)


def create_input_smiles_file(dir: str):
    smiles = get_nbse_smiles(dir)
    with open(os.path.join(dir, 'benchmark_input.smi'), 'w') as f:
        for (smi, n) in smiles:
            f.write('{}\t{}\n'.format(smi, n))
        f.flush()


def benchmark_nbse_ensemble(name: str, conf: Config) -> Result:
    directory = os.path.join(nbse_dir, name, 'ligands')
    orig: list[Chem.Mol] = get_nbse_mols(directory)

    infile_name = os.path.join(directory, 'benchmark_input.smi')
    if not os.path.isfile(infile_name):
        create_input_smiles_file(directory)

    outfile_name = os.path.join(directory, "result_coaler.sdf")

    start = time.time()
    run_coaler(infile_name, outfile_name, conf)
    end = time.time()

    out_mol_suppl = Chem.SDMolSupplier(outfile_name, sanitize=False)
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

        local_similarity += float(out.GetProp("_Score"))

        # we need a copy to align it to the original molecule in order to compute tanimoto shape similarity
        out_for_align = copy(out)
        print("aligned ligands with score: {}".format(
            AllChem.AlignMol(out_for_align, inp)))
        avg_conformer_tanimoto_dist += rdShapeHelpers.ShapeTanimotoDist(inp,
                                                                        out_for_align)

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
            print("could not find match between:\ninp: {}\nout: {}\n".format(
                AllChem.MolToSmiles(inp),
                AllChem.MolToSmiles(out)))

        for (in_prb, in_ref) in list(
                zip(range(len(match_in_ref)), list(match_in_ref))):
            atom_map[in_prb + offset_out] = in_ref + offset_in

    local_similarity /= len(out_by_name)
    avg_conformer_tanimoto_dist /= len(out_by_name)

    result_writer = Chem.SDWriter(
        os.path.join(directory, 'benchmark_result.sdf'))
    aligned = AllChem.AlignMol(merged_out, merged_in, -1, -1,
                               list(atom_map.items()))

    merged_out_mirror = copy(merged_out)
    aligned_reflect = AllChem.AlignMol(merged_out_mirror, merged_in, -1, -1,
                                       list(atom_map.items()), [], True)

    if aligned_reflect < aligned:
        aligned = aligned_reflect
        merged_out = merged_out_mirror

    merged_out.SetProp("_Name", "coaler_output")
    merged_in.SetProp("_Name", "siena_ligands")

    result_writer.write(merged_out)
    result_writer.write(merged_in)
    result_writer.close()

    res = Result(
        name=name,
        took=end - start,
        siena_rmsd=aligned,
        avg_conformer_tanimoto_dist=avg_conformer_tanimoto_dist,
        local_similarity=local_similarity,
        conf=conf,
    )

    return res


if __name__ == '__main__':
    first_forty = [  '3ke8', '2vke', '1odn', '4dko', '3qqs', '1qss',
                    '4asj', '2j7d', '4dwb', '3w1t', '2opm', '1aoe',
                    '3eyg', '3zrc', '4ali', '3ik0', '4gfd', '1uk1',
                    '3pci', '4mjp', '1uio', '1ie8', #'2hct',
                    '1of1', '1v48', '3g1o','1m8d', '1vso', '1qkn',
                    '1d0s', '2w0v', '2bzs', '3g5h', '3vt8', '3tfu',
                    '3id8', '2ves', '3sor', '2zi5']

    confs: list[Config] = [
        #Config(num_conformers=10, optimizer_fine=0.4, optimizer_coarse=0.9),
        #Config(num_conformers=20, optimizer_fine=0.5, optimizer_coarse=0.9),
        #Config(num_conformers=40, optimizer_fine=0.5, optimizer_coarse=0.9),

        #Config(num_conformers=10, optimizer_fine=0.1, optimizer_coarse=0.6),
        #Config(num_conformers=20, optimizer_fine=0.1, optimizer_coarse=0.3),
        #Config(num_conformers=40, optimizer_fine=0.1, optimizer_coarse=0.6),

        #Config(num_conformers=10, optimizer_fine=0.01, optimizer_coarse=0.5, assemblies=1),
        #Config(num_conformers=30, optimizer_fine=0.01, optimizer_coarse=0.5, assemblies=1),
        #Config(num_conformers=60, optimizer_fine=0.01, optimizer_coarse=0.5, assemblies=1),

        # Config(num_conformers=10, optimizer_fine=0.01, optimizer_coarse=0.3),
        # Config(num_conformers=20, optimizer_fine=0.01, optimizer_coarse=0.3),
        # Config(num_conformers=40, optimizer_fine=0.01, optimizer_coarse=0.3)

        # Config(num_conformers=20, optimizer_fine=0.05, optimizer_coarse=0.5),
        # Config(num_conformers=20, optimizer_fine=0.10, optimizer_coarse=0.5),
        # Config(num_conformers=20, optimizer_fine=0.20, optimizer_coarse=0.5),

        # Config(num_conformers=20, optimizer_fine=0.1, optimizer_coarse=0.3, core='murcko'),
    ]

    results_csv = None
    if not os.path.isfile('benchmark_results.csv'):
        results_csv = open('benchmark_results.csv', 'w')
        results_csv.write('{}\n'.format(Result.header()))
        results_csv.flush()

    else:
        results_csv = open('benchmark_results.csv', 'a')

    for conf in confs:
        for name in first_forty:
            print("running: {}".format(name))

            try:
                result = benchmark_nbse_ensemble(name, conf)
                results_csv.write(str(result) + '\n')
                print("result: {}".format(str(result)))
            except subprocess.TimeoutExpired:
                result = Result(name, timeout, 0, 0, 0, conf)
                results_csv.write(str(result) + '\n')
            except Exception as e:
                print("error: {}".format(e))
            finally:
                results_csv.flush()

    results_csv.close()
