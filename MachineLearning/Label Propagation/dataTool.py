import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string

def loadData(filePath):
    f = open(filePath)
    vector_dict = {}
    edge_dict = {}
    for line in f.readlines():
        lines = line.strip().split("\t")
        for i in range(2):
            if lines[i] not in vector_dict:
                vector_dict[lines[i]] = int(lines[i])
                edge_list = []
                if len(lines) == 3:
                    edge_list.append(lines[1 - i] + ":" + lines[2])
                else:
                    edge_list.append(lines[1 - i] + ":" + "1")
                edge_dict[lines[i]] = edge_list
            else:
                edge_list = edge_dict[lines[i]]
                if len(lines) == 3:
                    edge_list.append(lines[1 - i] + ":" + lines[2])
                else:
                    edge_list.append(lines[1 - i] + ":" + "1")
                edge_dict[lines[i]] = edge_list
    return vector_dict, edge_dict

if __name__ == '__main__':
    filePath = '../Data/label_data.txt'
    vector, edge = loadData(filePath)
    print(vector)
    print(edge)

