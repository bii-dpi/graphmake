import os
import numpy as np
import pandas as pd
from progressbar import progressbar


NUM_SELECTED = 5

np.random.seed(12345)

seq_to_id_dict = pd.read_pickle("b_sequence_to_id_map.pkl")
seq_to_id_dict.update(pd.read_pickle("d_sequence_to_id_map.pkl"))

b_pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
d_pdb_ids = list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
available_pdb_ids = [pdb_id for pdb_id in b_pdb_ids + d_pdb_ids if
                     os.path.isfile(f"proc_ligands/{pdb_id}_dist_matrices.pkl")]
np.random.shuffle(available_pdb_ids)


def get_selected_pdb_ids(available_pdb_ids, pdb_ids_subset):
    index = 0
    selected_pdb_ids = []
    while len(selected_pdb_ids) < NUM_SELECTED:
        if available_pdb_ids[index] in pdb_ids_subset:
            selected_pdb_ids.append(available_pdb_ids[index])
        index += 1

    return selected_pdb_ids


def save_cleaned_text_(direction, mode, tt):
    path = \
        f"../get_data/NewData/results/text/{direction}_{tt}{mode}"

    with open(path, "r") as f:
        lines = [line.split() for line in f.readlines()]

    cleaned_lines = []
    for line in progressbar(lines):
        curr_pdb_id = seq_to_id_dict[line[1]]
        if (curr_pdb_id not in selected_b_pdb_ids and
            curr_pdb_id not in selected_d_pdb_ids) or \
            line[0] not in available_smiles[curr_pdb_id]:
            continue
        cleaned_lines.append(" ".join(line))

    with open(f"cleaned_text/{direction}_{tt}{mode}", "w") as f:
        f.write("\n".join(cleaned_lines))


def save_cleaned_text(direction, mode="_normal"):
    save_cleaned_text_(direction, mode, "training")
    save_cleaned_text_(direction, "", "testing")


selected_b_pdb_ids = get_selected_pdb_ids(available_pdb_ids, b_pdb_ids)
selected_d_pdb_ids = get_selected_pdb_ids(available_pdb_ids, d_pdb_ids)

selected_b_pdb_ids = ["5RA9_A", "6B1U_D", "6Y1E_D", "4MQY_A", "6KHE_A"]
selected_d_pdb_ids = ["2RL5", "1HWK", "3E2M", "3KG2", "2F4J"]

available_smiles = dict()
for pdb_id in selected_b_pdb_ids + selected_d_pdb_ids:
    curr_smiles = pd.read_pickle(f"indiv_graphs/{pdb_id}.pkl")
    available_smiles[pdb_id] = list(curr_smiles.keys())

for direction in ["btd", "dtb"]:
    save_cleaned_text(direction)

