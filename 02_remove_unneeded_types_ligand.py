import os
import pickle
import pandas as pd
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]

selected_types = pd.read_pickle("atom_type_encoding_dict.pkl").keys()


def process_line(line):
    line = line.strip().split()[:6]
    line = [int(line[0]),
            float(line[2]), float(line[3]), float(line[4]),
            line[5].split(".")[0]]

    if line[-1] not in selected_types:
        return []
    return line


def save_proc_ligand(pdb_id):
    with open(f"../shallowmake/pdb_mol2/{pdb_id}.mol2", "r") as f:
        lines = [line.strip("\n") for line in f.readlines()]
    start = lines.index("@<TRIPOS>ATOM") + 1
    end = lines.index("@<TRIPOS>BOND")

    lines = [process_line(line) for line in lines[start: end]]
    lines = [line for line in lines if line]

    with open(f"proc_ligands/{pdb_id}.pkl", "wb") as f:
        pickle.dump(lines, f)


for pdb_id in progressbar(pdb_ids):
    save_proc_ligands(pdb_id)

