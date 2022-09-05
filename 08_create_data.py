import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor as PPE

seq_to_id_dict = pd.read_pickle("b_sequence_to_id_map.pkl")
seq_to_id_dict.update(pd.read_pickle("d_sequence_to_id_map.pkl"))

#selected_pdb_ids = ["5RA9_A", "6B1U_D", "6Y1E_D", "4MQY_A", "6KHE_A"]
#selected_pdb_ids += ["2RL5", "1HWK", "3E2M", "3KG2", "2F4J"]


def save_data_graph(line):
    curr_adjacency_list, curr_node_attributes, is_active = \
        indiv_graphs_dict[seq_to_id_dict[line[1]]][line[0]]

    curr_adjacency_list = np.array(curr_adjacency_list).T - 1
    curr_node_attributes = np.array(curr_node_attributes)

    return Data(x=Tensor(curr_node_attributes),
                edge_index=LongTensor(curr_adjacency_list),
                y=LongTensor([is_active]),
                num_nodes=len(curr_node_attributes))


def load_indiv_graphs(pdb_id):
    try:
        return pdb_id, pd.read_pickle(f"indiv_graphs/{pdb_id}.pkl")
    except:
        return pdb_id, None


with PPE() as executor:
    indiv_graphs_dict = executor.map(load_indiv_graphs,
#                                     selected_pdb_ids)
                                     seq_to_id_dict.values())
indiv_graphs_dict = {pdb_id: indiv_graphs for pdb_id, indiv_graphs
                     in indiv_graphs_dict if indiv_graphs}
print("Loaded indiv graphs.")


all_lines = set()
for direction in ["btd", "dtb"]:
    with open(f"cleaned_text/{direction}_training_normal", "r") as f:
        all_lines |= {line.strip("\n") for line in f.readlines()}
    with open(f"cleaned_text/{direction}_testing", "r") as f:
        all_lines |= {line.strip("\n") for line in f.readlines()}


line_dict = dict()
for line in progressbar(all_lines):
    line_dict[line] = save_data_graph(line.split())


with open("line_dict.pkl", "wb") as f:
    pickle.dump(line_dict, f)

