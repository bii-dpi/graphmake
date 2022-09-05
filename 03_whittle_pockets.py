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
#pdb_ids = [pdb_id for pdb_id in pdb_ids if not
#           os.path.isfile(f"proc_proteins/{pdb_id}_pocket.pkl")]


def save_protein_pocket(pdb_id):
    protein_coords = pd.read_pickle(f"proc_proteins/{pdb_id}.pkl")
    protein_coords = [coord[:-1] for coord in protein_coords]

    ligand_coords = pd.read_pickle(f"proc_ligands/{pdb_id}.pkl").values()
    ligand_coords = [pair[0] for pair in ligand_coords]
    ligand_coords = [coord[:-1] for sublist in ligand_coords
                     for coord in sublist]
    ligand_coords = np.array(ligand_coords)

    dist_matrix = euclidean_distances(protein_coords, ligand_coords)
    min_dists = np.apply_along_axis(np.min, 1, dist_matrix)
    pocket_indices = np.where(min_dists <= CUTOFF)[0]

    if not len(pocket_indices):
        return

    with open(f"proc_proteins/{pdb_id}_pocket.pkl", "wb") as f:
        protein_coords = pd.read_pickle(f"proc_proteins/{pdb_id}.pkl")
        protein_pocket_coords = [protein_coords[i] for i in pocket_indices]
        pickle.dump(protein_pocket_coords, f)


if __name__ == "__main__":
    with PPE(max_workers=40) as executor:
        executor.map(save_protein_pocket, pdb_ids)

