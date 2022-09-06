import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


seq_to_id_dict = pd.read_pickle("b_sequence_to_id_map.pkl")
seq_to_id_dict.update(pd.read_pickle("d_sequence_to_id_map.pkl"))

b_pdb_ids = [pdb_id for pdb_id in list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
             if pdb_id != "5YZ0_B"]
d_pdb_ids = list(pd.read_pickle("d_sequence_to_id_map.pkl").values())


def save_cleaned_text_(direction, mode, tt):
    path = \
        f"../get_data/NewData/results/text/{direction}_{tt}{mode}"

    with open(path, "r") as f:
        lines = [line.split() for line in f.readlines()]

    cleaned_lines = []
    for line in progressbar(lines):
        curr_pdb_id = seq_to_id_dict[line[1]]
        if (curr_pdb_id not in selected_b_pdb_ids and
            curr_pdb_id not in selected_d_pdb_ids):
            continue
        cleaned_lines.append(" ".join(line))

    with open(f"cleaned_text/{direction}_{tt}{mode}", "w") as f:
        f.write("\n".join(cleaned_lines))


def save_cleaned_text(direction, mode="_normal"):
    save_cleaned_text_(direction, mode, "training")
    save_cleaned_text_(direction, "", "testing")


#selected_b_pdb_ids = get_selected_pdb_ids(available_pdb_ids, b_pdb_ids)
#selected_d_pdb_ids = get_selected_pdb_ids(available_pdb_ids, d_pdb_ids)

#selected_b_pdb_ids = ["5RA9_A", "6B1U_D", "6Y1E_D", "4MQY_A", "6KHE_A"]
#selected_d_pdb_ids = ["2RL5", "1HWK", "3E2M", "3KG2", "2F4J"]

selected_b_pdb_ids = b_pdb_ids
selected_d_pdb_ids = d_pdb_ids

for direction in ["btd", "dtb"]:
    save_cleaned_text(direction)

