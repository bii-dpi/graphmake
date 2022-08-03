import os
import numpy as np
import pandas as pd
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


seq_to_id_dict = pd.read_pickle("b_sequence_to_id_map.pkl")
seq_to_id_dict.update(pd.read_pickle("d_sequence_to_id_map.pkl"))

selected_pdb_ids = ['1W6J_A', '3UNI_B', '6ZWO_B', '6OCU_A', '4YC6_G',
                    '2RGP', '3FRJ', '1B9V', '3H0B', '2PRH']


def convert_to_rows(list_):
    return "\n".join(list_)


def update_adjacency_list(adjacency_list, curr_adjacency_list, node_counter):
    curr_adjacency_list = np.array(curr_adjacency_list) + node_counter
    curr_adjacency_list = [", ".join(str(node) for node in sublist)
                           for sublist in curr_adjacency_list]

    return adjacency_list + curr_adjacency_list


def update_graph_indicators(graph_indicators, graph_counter, num_nodes):
    return graph_indicators + [graph_counter for _ in range(num_nodes)]


def update_node_attributes(node_attributes, curr_node_attributes):
    return node_attributes + [",".join(str(i) for i in line)
                              for line in curr_node_attributes]


def update_node_labels(node_labels, num_nodes):
    return node_labels + [0 for _ in range(num_nodes)]


def update_graph_labels(graph_labels, is_active):
    return graph_labels + [is_active]


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
        graph_labels = update_graph_labels(graph_labels, is_active)

        num_nodes = len(curr_node_attributes)
        node_labels = update_node_labels(node_labels, num_nodes)
        node_attributes = update_node_attributes(node_attributes,
                                                 curr_node_attributes)

        graph_indicators = update_graph_indicators(graph_indicators,
                                                   graph_counter,
                                                   num_nodes)

        adjacency_list = update_adjacency_list(adjacency_list,
                                               curr_adjacency_list,
                                               node_counter)

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

    with open(f"cleaned_text/{direction}_testing", "r") as f:
        lines = [line.split() for line in f.readlines()]

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
    adjancency_list = convert_to_rows(adjancency_list)

    with open(f"compiled_graphs/{direction}_A.txt", "w"):
        f.write(adjacency_list)

    with open(f"compiled_graphs/{direction}_graph_indicator.txt", "w"):
        f.write(graph_indicators)

    with open(f"compiled_graphs/{direction}_graph_labels.txt", "w"):
        f.write(graph_labels)

    with open(f"compiled_graphs/{direction}_node_labels.txt", "w"):
        f.write(node_labels)

    with open(f"compiled_graphs/{direction}_node_attributes.txt", "w"):
        f.write(node_attributes)

    with open(f"compiled_graphs/{direction}_training_marker.pkl", "wb") as f:
        pickle.dump((num_training_graphs, num_training_nodes), f)


def load_indiv_graphs(pdb_id):
    try:
        return pdb_id, pd.read_pickle(f"indiv_graphs/{pdb_id}.pkl")
    except:
        return


with PPE() as executor:
    indiv_graphs_dict = executor.map(load_indiv_graphs,
                                     selected_pdb_ids)
                                     #seq_to_id_dict.values())
indiv_graphs_dict = {pdb_id: indiv_graphs for pdb_id, indiv_graphs
                     in indiv_graphs_dict if indiv_graphs}

for direction in ["btd", "dtb"]:
    save_compiled_graphs(direction)

