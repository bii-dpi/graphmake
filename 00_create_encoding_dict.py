import pickle
import pandas as pd


with open("ligand_types.csv", "r") as f:
    atom_types = {line.split(",")[0].split(".")[0]
                  for line in f.readlines()[1:]}

with open("protein_types.csv", "r") as f:
    atom_types |= {line.split(",")[0].split(".")[0]
                   for line in f.readlines()[1:]}

ligand_totals = {atom_type: 0 for atom_type in atom_types}
ligand_counts = pd.read_csv("ligand_types.csv")
for i in range(len(ligand_counts)):
    atom_type = ligand_counts.iloc[i, 0].split(".")[0]
    ligand_totals[atom_type] += ligand_counts.iloc[i, 1]

ligand_totals = [[atom_type, total]
                 for atom_type, total in ligand_totals.items()]
ligand_totals = sorted(ligand_totals, key=lambda pair: -pair[1])

protein_totals = {atom_type: 0 for atom_type in atom_types}
protein_counts = pd.read_csv("protein_types.csv")
for i in range(len(protein_counts)):
    atom_type = protein_counts.iloc[i, 0].split(".")[0]
    protein_totals[atom_type] += protein_counts.iloc[i, 1]

protein_totals = [[atom_type, total]
                 for atom_type, total in protein_totals.items()]
protein_totals = sorted(protein_totals, key=lambda pair: -pair[1])

selected_ligand_types = {"H", "C", "N", "O", "S", "F", "Cl", "Br"}
selected_protein_types = {"C", "O", "N", "S", "H"}

selected_atom_types = selected_ligand_types | selected_protein_types
selected_atom_types = sorted(list(selected_atom_types))

encoding_dict = dict(zip(selected_atom_types, range(len(selected_atom_types))))

print(encoding_dict)

with open("atom_type_encoding_dict.pkl", "wb") as f:
    pickle.dump(encoding_dict, f)

