import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from sklearn.metrics.pairwise import euclidean_distances
from concurrent.futures import ProcessPoolExecutor as PPE


CUTOFF = 6

pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]
#and not os.path.isfile(f"missing/{pdb_id}_{CUTOFF}.pkl")]

atom_encoding_dict = pd.read_pickle("atom_type_encoding_dict.pkl")


def get_indicator(element):
    base = [0 for _ in range(len(atom_encoding_dict))]
    base[atom_encoding_dict[element]] = 1

    return base


def get_relevant_data(dist_matrix, protein_elements, ligand_elements):
    min_dists = np.apply_along_axis(np.min, 1, dist_matrix)
    relevant_proteins = np.where(min_dists <= CUTOFF)[0]

    min_dists = np.apply_along_axis(np.min, 0, dist_matrix)
    relevant_ligands = np.where(min_dists <= CUTOFF)[0]

    return dist_matrix[np.ix_(relevant_proteins, relevant_ligands)], \
            [protein_elements[i] for i in relevant_proteins], \
            [ligand_elements[i] for i in relevant_ligands]


def get_graph(dist_matrix, protein_elements, ligand_elements, is_active):
    dist_matrix, protein_elements, ligand_elements = \
        get_relevant_data(dist_matrix, protein_elements, ligand_elements)

    if len(protein_elements) == 0:
        return

    node_attributes = [get_indicator(element) for element in
                       protein_elements + ligand_elements]

    dist_matrix = dist_matrix <= CUTOFF

    adjacency_list = []
    for protein_index in range(dist_matrix.shape[0]):
        for ligand_index in range(dist_matrix.shape[1]):
            if dist_matrix[protein_index][ligand_index]:
                protein_index_ = protein_index + 1
                ligand_index_ = ligand_index + dist_matrix.shape[0] + 1
                adjacency_list.append([protein_index_, ligand_index_])
                adjacency_list.append([ligand_index_, protein_index_])

    return adjacency_list, node_attributes, is_active


def save_graphs(pdb_id):
    try:
        if os.path.isfile(f"indiv_graphs/{pdb_id}.pkl"):
            return

        missing = [[], []]

        protein_elements = pd.read_pickle(f"proc_proteins/{pdb_id}_pocket.pkl")

        if not len(protein_elements):
            return

        protein_elements = [quartet[-1] for quartet in protein_elements]
        ligand_elements_dict = pd.read_pickle(f"proc_ligands/{pdb_id}.pkl")
        for smiles, (ligand_elements, is_active) in ligand_elements_dict.items():
            ligand_elements_dict[smiles] = \
                ([quartet[-1] for quartet in ligand_elements], is_active)

        ligand_dist_matrices_dict = \
            pd.read_pickle(f"proc_ligands/{pdb_id}_dist_matrices.pkl")
        ligand_graphs_dict = dict()
        for smiles in ligand_dist_matrices_dict:
            indiv_graph = get_graph(ligand_dist_matrices_dict[smiles],
                                    protein_elements,
                                    *ligand_elements_dict[smiles])
            if indiv_graph is not None:
                ligand_graphs_dict[smiles] = indiv_graph
                missing[ligand_elements_dict[smiles][1]].append(0)
            else:
                missing[ligand_elements_dict[smiles][1]].append(1)

        with open(f"indiv_graphs/{pdb_id}.pkl", "wb") as f:
            pickle.dump(ligand_graphs_dict, f)

        missing[0] = np.mean(missing[0])
        missing[1] = np.mean(missing[1])
        with open(f"missing/{pdb_id}_{CUTOFF}.pkl", "wb") as f:
            pickle.dump(missing, f)
    except Exception as e:
        print(e)
        print(pdb_id)


np.random.shuffle(pdb_ids)

if __name__ == "__main__":
    for pdb_id in progressbar(pdb_ids):
        save_graphs(pdb_id)
    '''
    with PPE(max_workers=28) as executor:
        executor.map(save_graphs, pdb_ids)
    '''
