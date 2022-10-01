import os
import torch
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from scipy.spatial import distance_matrix
from concurrent.futures import ProcessPoolExecutor as PPE


CUTOFF = 6

pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]

atom_encoding_dict = pd.read_pickle("atom_type_encoding_dict.pkl")


def convert_to_torch(adjacency_list, node_attributes):
    adjacency_list = np.array(adjacency_list).T - 1
    node_attributes = np.array(node_attributes)

    return Data(x=Tensor(node_attributes),
                edge_index=LongTensor(adjacency_list),
                num_nodes=len(node_attributes))


def get_indicator(element):
    base = [0 for _ in range(len(atom_encoding_dict))]
    base[atom_encoding_dict[element]] = 1

    return base


def get_graph(dist_matrix, elements_list):
    node_attributes = [get_indicator(element) for element in elements_list]

    dist_matrix = dist_matrix <= CUTOFF

    adjacency_list = []
    for index_1 in range(dist_matrix.shape[0]):
        for index_2 in range(dist_matrix.shape[1]):
            if dist_matrix[index_1][index_2]:
                index_1_ = index_1 + 1
                index_2_ = index_2 + dist_matrix.shape[0] + 1
                adjacency_list.append([index_1_, index_2_])
                adjacency_list.append([index_2_, index_1_])

    return convert_to_torch(adjacency_list, node_attributes)


def save_graphs(pdb_id):
    try:
        protein_pocket = pd.read_pickle(f"proc_proteins/{pdb_id}.pkl")

        coordinates_list = [row[:-1] for row in protein_pocket]
        elements_list = [row[-1] for row in protein_pocket]

        dist_matrix = distance_matrix(coordinates_list, coordinates_list)

        with open(f"protein_graphs/{pdb_id}.pkl", "wb") as f:
            pickle.dump(get_graph(dist_matrix, elements_list), f)

    except Exception as e:
        print(e)
        print(pdb_id)


np.random.shuffle(pdb_ids)

if __name__ == "__main__":
    '''
    for pdb_id in progressbar(pdb_ids):
        save_graphs(pdb_id)
    '''
    with PPE() as executor:
        executor.map(save_graphs, pdb_ids)

