import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]


def get_dist_fn(protein_coord):
    protein_coord = np.array(protein_coord)

    return lambda row: np.linalg.norm(row - protein_coord)


def is_pocket_index(pair):
    protein_coord, ligand_coords = pair
    all_dists = np.apply_along_axis(get_dist_fn(protein_coord),
                                    1,
                                    ligand_coords)

    return np.min(all_dists) <= 6


def is_pocket_index_batch(pair_batch):

    return [is_pocket_index(pair) for pair in pair_batch]


def save_protein_pocket(pdb_id, cutoff=6):
    protein_coords = pd.read_pickle(f"proc_proteins/{pdb_id}.pkl")
    protein_coords = [coord[:-1] for coord in protein_coords]

    ligand_coords = pd.read_pickle(f"proc_ligands/{pdb_id}.pkl").values()
    ligand_coords = [pair[0] for pair in ligand_coords]
    ligand_coords = [coord[:-1] for sublist in ligand_coords
                     for coord in sublist]
    ligand_coords = np.array(ligand_coords)

    protein_batches = np.array_split(protein_coords, 75)
    pair_batches = []
    for protein_batch in protein_batches:
        pair_batches.append([(protein_coord, ligand_coords)
                             for protein_coord in protein_batch])

    with PPE() as executor:
        are_pocket_indices_batched = \
            executor.map(is_pocket_index_batch, pair_batches)
    are_pocket_indices = [is_pocket_index
                          for batch in are_pocket_indices_batched
                          for is_pocket_index in batch]
    pocket_indices = np.where(are_pocket_indices)[0]

    with open(f"proc_proteins/{pdb_id}_pocket.pkl", "wb") as f:
        protein_coords = pd.read_pickle(f"proc_proteins/{pdb_id}.pkl")
        protein_pocket_coords = [protein_coords[i] for i in pocket_indices]
        pickle.dump(protein_pocket_coords, f)


if __name__ == "__main__":
    for pdb_id in progressbar(pdb_ids):
        save_protein_pocket(pdb_id)

