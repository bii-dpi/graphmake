import os
import pickle
import numpy as np
import pandas as pd
import torch
from progressbar import progressbar
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor as PPE


seq_to_id_dict = pd.read_pickle("b_sequence_to_id_map.pkl")
seq_to_id_dict.update(pd.read_pickle("d_sequence_to_id_map.pkl"))
seq_to_id_dict = {seq: pdb_id for seq, pdb_id in seq_to_id_dict.items()
                  if pdb_id != "5YZ0_B" and pdb_id != "6WHC_R"}

#selected_pdb_ids = ["5RA9_A", "6B1U_D", "6Y1E_D", "4MQY_A", "6KHE_A"]
#selected_pdb_ids += ["2RL5", "1HWK", "3E2M", "3KG2", "2F4J"]


def get_data_graph(line, indiv_graphs):
    curr_adjacency_list, curr_node_attributes, is_active = \
        indiv_graphs[line[0]]

    curr_adjacency_list = np.array(curr_adjacency_list).T - 1
    curr_node_attributes = np.array(curr_node_attributes)

    return Data(x=Tensor(curr_node_attributes),
                edge_index=LongTensor(curr_adjacency_list),
                y=LongTensor([is_active]),
                num_nodes=len(curr_node_attributes))


def save_line_dict(pair):
    pdb_id, lines = pair
    indiv_graphs = pd.read_pickle(f"indiv_graphs/{pdb_id}.pkl")

    line_dict = dict()
    for line in lines:
        line_dict[" ".join(line)] = get_data_graph(line, indiv_graphs)

    '''
    torch.save(f"line_dicts/{pdb_id}.pt", line_dict)

    '''
    with open(f"line_dicts/{pdb_id}.pkl", "wb") as f:
        pickle.dump(line_dict, f)


def get_pdb_id(line):
    return seq_to_id_dict[line[1]]


all_lines = set()
for direction in ["btd", "dtb"]:
    with open(f"cleaned_text/{direction}_training_normal", "r") as f:
        all_lines |= {line.strip("\n") for line in f.readlines()}
    with open(f"cleaned_text/{direction}_testing", "r") as f:
        all_lines |= {line.strip("\n") for line in f.readlines()}


all_lines = [line.split() for line in all_lines]
all_lines = [(get_pdb_id(line), line) for line in all_lines]
all_lines_dict = dict()
for pdb_id in seq_to_id_dict.values():
    all_lines_dict[pdb_id] = [pair[1] for pair in all_lines
                              if pair[0] == pdb_id]
all_lines = [(pdb_id, lines) for pdb_id, lines in all_lines_dict.items()
             if not os.path.isfile(f"line_dicts/{pdb_id}.pkl")]
print(len(all_lines))

save_line_dict(all_lines[0])
with PPE(max_workers=10) as executor:
    executor.map(save_line_dict, all_lines)

