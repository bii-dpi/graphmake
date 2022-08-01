import pickle
import warnings
import pandas as pd
from concurrent.futures import ProcessPoolExecutor as PPE


warnings.filterwarnings('ignore')

pdb_ids = list(pd.read_pickle("b_sequence_to_id_map.pkl").values())
pdb_ids += list(pd.read_pickle("d_sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]

SELECTED_TYPES = pd.read_pickle("atom_type_encoding_dict.pkl").keys()


def load_ids(pdb_id):
    scores = pd.read_csv(f"../shallowmake/scores/dock_{pdb_id}_lib.csv")
    return scores.loc[:, ["s_sm_number", "s_m_title"]].dropna()


def get_ligand_dict(pdb_id):
    ids = pd.read_csv(f"../shallowmake/ligand_ids/{pdb_id}.csv")

    return dict(zip(ids.iloc[:, 1].tolist(), ids.iloc[:, 0].tolist()))


def get_id_dict(pdb_id):
    ids = load_ids(pdb_id)
    ligand_dict = get_ligand_dict(pdb_id)
    casual_ids = [id_.split(":")[1] for id_ in ids.iloc[:, 1].tolist()]
    ligands = [(ligand_dict[id_], int(id_.startswith("A")))
               for id_ in ids.iloc[:, 0].tolist()]

    return dict(zip(casual_ids, ligands))


def process_line(line):
    line = line.strip().split()[:6]
    line = [float(line[2]), float(line[3]), float(line[4]),
            line[5].split(".")[0]]

    return line


def process_indiv(indiv, id_dict):
    try:
        indiv = indiv.split("\n")[1:]
        ligand_name, is_active = id_dict[indiv[0].split(":")[1]]
        start = indiv.index("@<TRIPOS>ATOM") + 1
        end = indiv.index("@<TRIPOS>BOND")
        indiv = [process_line(line) for line in indiv[start: end]]
        indiv = [line for line in indiv if line[-1] in SELECTED_TYPES]

        return ligand_name, (indiv, is_active)
    except Exception as e:
        print(e)
#        print(1)
        return None


def save_proc_ligands(pdb_id):
    id_dict = get_id_dict(pdb_id)

    with open(f"../shallowmake/ligand_mol2/dock_{pdb_id}.mol2", "r") as f:
        split_mol2 = f.read().split("@<TRIPOS>MOLECULE")[1:]

    with open(f"proc_ligands/{pdb_id}.pkl", "wb") as f:
        results = []
        for indiv in split_mol2:
            curr_results = process_indiv(indiv, id_dict)
            if curr_results is not None:
                results.append(curr_results)
        pickle.dump(dict(results), f)


with PPE() as executor:
    executor.map(save_proc_ligands, pdb_ids)

