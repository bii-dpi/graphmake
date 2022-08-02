import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from sklearn.metrics.pairwise import euclidean_distances
from concurrent.futures import ProcessPoolExecutor as PPE


pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]

atom_encoding_dict = pd.read_pickle("atom_type_encoding_dict.pkl")


def get_relevant_protein_atoms(dist_matrix, protein_pairs):
    min_dists = np.apply_along_axis(np.min, 1, dist_matrix)
    relevant_indices = np.where(min_dists <= 6)[0]

    return dist_matrix[relevant_indices, :], \
            [protein_pairs[i] for i in relevant_indices]


def get_indicator(element):
    base = [0 for _ in range(len(atom_encoding_dict))]
    base[atom_encoding_dict[element]] = 1

    return base


def get_graph(dist_matrix, protein_elements, ligand_elements, is_active):
    protein_pairs = list(zip(protein_elements,
                                range(1, len(protein_elements) + 1)))
    ligand_pairs = list(zip(ligand_elements,
                               range(len(protein_elements),
                                     len(protein_elements) + \
                                        len(ligand_elements) + 1)))

    dist_matrix, protein_pairs = \
        get_relevant_protein_atoms(dist_matrix, protein_pairs)


    node_attributes = [get_indicator(element) for element in
                       [pair[0] for pair in protein_pairs] +
                       [pair[0] for pair in ligand_pairs]]

    dist_matrix = dist_matrix <= 6

    adjacency_list = []
    for protein_index in range(dist_matrix.shape[0]):
        for ligand_index in range(dist_matrix.shape[1]):
            if dist_matrix[protein_index][ligand_index]:
                adjacency_list.append([protein_index, ligand_index])
                adjacency_list.append([ligand_index, protein_index])

    return adjacency_list, node_attributes, is_active


def save_graphs(pdb_id):
    protein_elements = pd.read_pickle(f"proc_proteins/{pdb_id}_pocket.pkl")
    protein_elements = [quartet[-1] for quartet in protein_elements]
    ligand_elements_dict = pd.read_pickle(f"proc_ligands/{pdb_id}.pkl")
    for smiles, (ligand_elements, is_active) in ligand_elements_dict.items():
        ligand_elements_dict[smiles] = \
            ([quartet[-1] for quartet in ligand_elements], is_active)

    ligand_dist_matrices_dict = \
        pd.read_pickle(f"proc_ligands/{pdb_id}_dist_matrices.pkl")
    ligand_graphs_dict = dict()
    for smiles in progressbar(ligand_dist_matrices_dict):
        ligand_graphs_dict[smiles] = \
            get_graph(ligand_dist_matrices_dict[smiles],
                      protein_elements,
                      *ligand_elements_dict[smiles])

    with open(f"indiv_graphs/{pdb_id}.pkl", "wb") as f:
        pickle.dump(ligand_graphs_dict, f)


if __name__ == "__main__":
    with PPE() as executor:
        executor.map(save_graphs, pdb_ids)

