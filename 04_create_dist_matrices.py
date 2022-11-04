import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from sklearn.metrics.pairwise import euclidean_distances
from concurrent.futures import ProcessPoolExecutor as PPE


pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"][::-1]
pdb_ids = [pdb_id for pdb_id in pdb_ids if not
           os.path.isfile(f"proc_ligands/{pdb_id}_dist_matrices.pkl")]
print(len(pdb_ids))


def save_dist_matrices(pdb_id):
    protein_coords = pd.read_pickle(f"proc_proteins/{pdb_id}_pocket.pkl")
    try:
        protein_coords = [coord[:-1] for coord in protein_coords]

        ligand_coords_dict = pd.read_pickle(f"proc_ligands/{pdb_id}.pkl")
        ligand_coords_dict = {smiles: np.array([coord[:-1] for coord in pair[0]])
                              for smiles, pair in ligand_coords_dict.items()}

        ligand_distance_matrices = dict()
        for smiles in progressbar(ligand_coords_dict):
            ligand_distance_matrices[smiles] = \
                euclidean_distances(protein_coords, ligand_coords_dict[smiles])

        with open(f"proc_ligands/{pdb_id}_dist_matrices.pkl", "wb") as f:
            pickle.dump(ligand_distance_matrices, f)
    except Exception as e:
        print(e)
        return

if __name__ == "__main__":
    with PPE(max_workers=10) as executor:
        executor.map(save_dist_matrices, pdb_ids)

