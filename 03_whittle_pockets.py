import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]
pdb_ids = [pdb_id for pdb_id in pdb_ids
           if not os.path.isfile(f"proc_proteins/{pdb_id}_pocket.pkl")]
print(len(pdb_ids))



def get_dist_fn(protein_coord):
    protein_coord = np.array(protein_coord)

    return lambda row: np.linalg.norm(row - protein_coord)


def save_protein_pocket(pdb_id, cutoff=6):
    protein_coords = pd.read_pickle(f"proc_proteins/{pdb_id}.pkl")
    protein_coords = [coord[:-1] for coord in protein_coords]

    ligand_coords = pd.read_pickle(f"proc_ligands/{pdb_id}.pkl").values()
    ligand_coords = [pair[0] for pair in ligand_coords]
    ligand_coords = [coord[:-1] for sublist in ligand_coords
                     for coord in sublist]
    ligand_coords = np.array(ligand_coords)

    pocket_indices = []
    for i, protein_coord in list(enumerate(protein_coords)):
        all_dists = np.apply_along_axis(get_dist_fn(protein_coord),
                                        1,
                                        ligand_coords)
        if np.min(all_dists) <= cutoff:
            pocket_indices.append(i)

    with open(f"proc_proteins/{pdb_id}_pocket.pkl", "wb") as f:
        protein_coords = pd.read_pickle(f"proc_proteins/{pdb_id}.pkl")
        protein_pocket_coords = [protein_coords[i] for i in pocket_indices]
        pickle.dump(protein_pocket_coords, f)


if __name__ == "__main__":
    with PPE() as executor:
        executor.map(save_protein_pocket, pdb_ids)

