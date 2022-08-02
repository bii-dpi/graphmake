import pandas as pd


class Graph:
    def __init__(self, dist_matrix, protein_elements, ligand_elements):
        self.protein_elements = list(zip(protein_elements,
                                    range(1, len(protein_elements + 1)))
        self.ligand_elements = list(zip(ligand_elements,
                                   range(len(protein_elements),
                                         len(protein_elements) + \
                                            len(ligand_elements) + 1))
        self.set_node_attributes()
        self.set_adjacency_matrix(


        pass

