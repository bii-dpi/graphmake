import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from sklearn.metrics.pairwise import euclidean_distances
from concurrent.futures import ProcessPoolExecutor as PPE


pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]


def save_protein_pocket(pdb_id):
    protein_pocket_coords = pd.read_pickle(f"proc_proteins/{pdb_id}_pocket.pkl")
    print({l[-1] for l in protein_pocket_coords})


if __name__ == "__main__":
    with PPE(max_workers=10) as executor:
        executor.map(save_protein_pocket, pdb_ids)

