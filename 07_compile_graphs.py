import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


seq_to_id_dict = pd.read_pickle("b_sequence_to_id_map.pkl")
seq_to_id_dict.update(pd.read_pickle("d_sequence_to_id_map.pkl"))

selected_pdb_ids = ["5RA9_A", "6B1U_D", "6Y1E_D", "4MQY_A", "6KHE_A"]
selected_pdb_ids += ["2RL5", "1HWK", "3E2M", "3KG2", "2F4J"]


def convert_to_rows(list_):
    list_ = [str(element) for sublist in list_ for element in sublist]
    return "\n".join(list_)


def update_adjacency_list(adjacency_list, curr_adjacency_list, node_counter):
    curr_adjacency_list = np.array(curr_adjacency_list) + node_counter
    curr_adjacency_list = [", ".join(str(node) for node in sublist)
                           for sublist in curr_adjacency_list]

    adjacency_list.append(curr_adjacency_list)


def update_graph_indicators(graph_indicators, graph_counter, num_nodes):
    graph_indicators.append([graph_counter for _ in range(num_nodes)])


def update_node_attributes(node_attributes, curr_node_attributes):
    node_attributes.append([",".join(str(i) for i in line)
                            for line in curr_node_attributes])


def update_node_labels(node_labels, num_nodes, is_active):
    node_labels.append([is_active for _ in range(num_nodes)])


def update_graph_labels(graph_labels, is_active):
    graph_labels.append([is_active])


def save_compiled_graphs_loop(lines,
                              graph_counter,
                              node_counter,
                              graph_labels,
                              node_labels,
                              node_attributes,
                              graph_indicators,
                              adjacency_list):
    for line in progressbar(lines):
        curr_adjacency_list, curr_node_attributes, is_active = \
            indiv_graphs_dict[seq_to_id_dict[line[1]]][line[0]]

        graph_counter += 1
        update_graph_labels(graph_labels, is_active)

        num_nodes = len(curr_node_attributes)
        if len(curr_adjacency_list) == 0:
            print(graph_counter)

        update_node_labels(node_labels, num_nodes, is_active)
        update_node_attributes(node_attributes, curr_node_attributes)

        update_graph_indicators(graph_indicators, graph_counter, num_nodes)

        update_adjacency_list(adjacency_list, curr_adjacency_list, node_counter)

        node_counter += num_nodes

    return (graph_counter,
            node_counter,
            graph_labels,
            node_labels,
            node_attributes,
            graph_indicators,
            adjacency_list)


def save_compiled_graphs(direction):
    with open(f"cleaned_text/{direction}_training_normal", "r") as f:
        lines = [line.split() for line in f.readlines()]

    (graph_counter, node_counter,
     graph_labels, node_labels, node_attributes,
     graph_indicators, adjacency_list) = \
        save_compiled_graphs_loop(lines, 0, 0, [], [], [], [], [])

    num_training_graphs = graph_counter
    num_training_nodes = node_counter

    ###
    if direction == "dtb":
        direction = "btd"
    else:
        direction = "dtb"

    with open(f"cleaned_text/{direction}_training_normal", "r") as f:
        lines = [line.split() for line in f.readlines()]
    ###

    (graph_counter, node_counter,
     graph_labels, node_labels, node_attributes,
     graph_indicators, adjacency_list) = \
        save_compiled_graphs_loop(lines,
                                  graph_counter,
                                  node_counter,
                                  graph_labels,
                                  node_labels,
                                  node_attributes,
                                  graph_indicators,
                                  adjacency_list)

    graph_labels = convert_to_rows(graph_labels)
    node_labels = convert_to_rows(node_labels)
    node_attributes = convert_to_rows(node_attributes)
    graph_indicators = convert_to_rows(graph_indicators)
    adjacency_list = convert_to_rows(adjacency_list)

    n = len(graph_indicators.split("\n"))
    assert n == len(node_labels.split("\n"))
    assert n == len(node_attributes.split("\n"))

    with open(f"compiled_graphs/{direction}/raw/{direction}_A.txt", "w") as f:
        f.write(adjacency_list)

    with open(f"compiled_graphs/{direction}/raw/{direction}_graph_indicator.txt", "w") as f:
        f.write(graph_indicators)

    with open(f"compiled_graphs/{direction}/raw/{direction}_graph_labels.txt", "w") as f:
        f.write(graph_labels)

    with open(f"compiled_graphs/{direction}/raw/{direction}_node_labels.txt", "w") as f:
        f.write(node_labels)

    with open(f"compiled_graphs/{direction}/raw/{direction}_node_attributes.txt", "w") as f:
        f.write(node_attributes)

    with open(f"compiled_graphs/{direction}/raw/{direction}_training_marker.pkl", "wb") as f:
        pickle.dump((num_training_graphs, num_training_nodes), f)


def load_indiv_graphs(pdb_id):
    try:
        return pdb_id, dict(pd.read_pickle(f"indiv_graphs/{pdb_id}.pkl"))
    except:
        return pdb_id, None


with PPE() as executor:
    indiv_graphs_dict = executor.map(load_indiv_graphs,
                                     selected_pdb_ids)
                                     #seq_to_id_dict.values())
indiv_graphs_dict = {pdb_id: indiv_graphs for pdb_id, indiv_graphs
                     in indiv_graphs_dict if indiv_graphs}
print("Loaded indiv graphs.")

for direction in ["btd", "dtb"]:
    save_compiled_graphs(direction)

