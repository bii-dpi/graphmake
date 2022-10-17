import os
import numpy as np
import pandas as pd
from collections import defaultdict
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


seq_to_id_dict = pd.read_pickle("b_sequence_to_id_map.pkl")
seq_to_id_dict.update(pd.read_pickle("d_sequence_to_id_map.pkl"))
pdb_ids = [pdb_id for pdb_id in seq_to_id_dict.values() if pdb_id != "5YZ0_B"]

atom_encoding_dict = pd.read_pickle("atom_type_encoding_dict.pkl")


def read_smiles(direction, mode):
    with open(f"../get_data/NewData/results/text/{direction}_{mode}", "r") as f:
        lines = [line.split()[:2] for line in f.readlines()]

    curr_smiles_dict = defaultdict(set)
    for line in lines:
        curr_smiles_dict[seq_to_id_dict[line[1]]] |= {line[0]}

    return curr_smiles_dict


def print_missing(pdb_id):
    #processed_smiles = pd.read_pickle(f"proc_ligands/{pdb_id}.pkl").keys()
    processed_smiles = pd.read_pickle(f"ligand_graphs/{pdb_id}.pkl").keys()

    print(pdb_id, len(smiles_dict[pdb_id] - set(processed_smiles)))


smiles_dict = defaultdict(set)
for direction in ["btd", "dtb"]:
    for mode in ["training_normal", "testing"]:
        curr_smiles_dict = read_smiles(direction, mode)
        for pdb_id in pdb_ids:
            smiles_dict[pdb_id] |= curr_smiles_dict[pdb_id]

for pdb_id in pdb_ids:
    print_missing(pdb_id)

