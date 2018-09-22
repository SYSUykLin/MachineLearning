import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataTool import loadData

def get_max_community_label(vector_dict, adjacency_node_list):
    label_dict = {}
    for node in adjacency_node_list:
        node_id_weight = node.strip().split(":")
        node_id = node_id_weight[0]
        node_weight = int(node_id_weight[1])

        if vector_dict[node_id] not in label_dict:
            label_dict[vector_dict[node_id]] = node_weight
        else:
            label_dict[vector_dict[node_id]] += node_weight

    sort_list = sorted(label_dict.items(), key=lambda d: d[1], reverse=True)
    return sort_list[0][0]

def check(vector_dict, edge_dict):
    for node in vector_dict.keys():
        adjacency_node_list = edge_dict[node]
        node_label = vector_dict[node]
        label = get_max_community_label(vector_dict, adjacency_node_list)
        if node_label >= label:
            continue
        else:
            return 0
    return 1

def label_propagation(vector_dict, edge_dict):
    t = 0
    print('First Label: ')
    while True:
        if (check(vector_dict, edge_dict) == 0):
            t = t + 1
            print('iteration: ', t)
            for node in vector_dict.keys():
                adjacency_node_list = edge_dict[node]
                vector_dict[node] = get_max_community_label(vector_dict, adjacency_node_list)
        else:
            break
    return vector_dict

if __name__ == '__main__':
    filePath = '../Data/label_data.txt'
    vector, edge = loadData(filePath)
    print(edge)
    vector_dict = label_propagation(vector, edge)
    print(vector_dict)