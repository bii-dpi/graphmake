import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from sklearn.metrics.pairwise import euclidean_distances
from concurrent.futures import ProcessPoolExecutor as PPE


CUTOFF = 4

pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]


def get_row(pdb_id):
    try:
        missing = pd.read_pickle(f"missing/{pdb_id}_{CUTOFF}.pkl")
    except:
        missing = [1, 1]

    missing = [f"{prop * 100:.2f}" for prop in missing]

    return f"{pdb_id},{missing[1]},{missing[0]}"


rows = ["PDB ID,Actives missing (%),Decoys missing (%)"]
for pdb_id in pdb_ids:
    rows.append(get_row(pdb_id))

with open(f"missing_summary_{CUTOFF}.csv", "w") as f:
    f.write("\n".join(rows))

