import networkx as nx
import numpy as np
import os
import torch
from tqdm import tqdm

def gen_graph(data, weights):
    edges = data
    weights = weights
    g1 = []
    for i in range(len(edges)):
        l = list(edges[i])
        l.append(weights[i])
        g1.append(tuple(l))

    g = nx.Graph()
    g.add_weighted_edges_from(g1)
    return g

def read_tu_data(file_path, dataname):
    with open('{}/{}/A.txt'.format(file_path, dataname), 'r') as f:
        edges = f.read().splitlines()
    edges = [tuple(map(int, e.replace(" ", "").split(","))) for e in edges]

    with open('{}/{}/graph_indicator.txt'.format(file_path, dataname), 'r') as f:
        g = f.readlines()
    g = [int(i) for i in g]
    slice_x = torch.cumsum(torch.bincount(torch.LongTensor(g) - 1), 0)
    slice_x = torch.cat([torch.LongTensor([0]), slice_x])

    weights = []
    if os.path.exists('{}/{}/edge_labels.txt'.format(file_path, dataname)):
        with open('{}/{}/edge_labels.txt'.format(file_path, dataname), 'r') as f:
            w = f.readlines()
        weights = [int(i) for i in w]

    with open('{}/{}/graph_labels.txt'.format(file_path, dataname), 'r') as f:
        l = f.readlines()
    graph_labels = [int(i) for i in l]

    with open('{}/{}/node_labels.txt'.format(file_path, dataname), 'r') as f:
        nl = f.readlines()
    node_labels = [int(i[-2]) for i in nl]

    G_edges = []
    G_weight = []

    if os.path.exists('{}/{}/edge_labels.txt'.format(file_path, dataname)):
        for i in tqdm(range(len(graph_labels)), desc="Create edges", unit='graphs'):
            edge = []
            for e in range(len(edges)):
                if g[edges[e][0] - 1] == i + 1:
                    edge.append(edges[e])

                elif g[edges[e][0] - 1] == i + 2:
                    break
            G_edges.append(edge)
        G_weight = []
        for i in tqdm(range(len(graph_labels)), desc="Create weights", unit='graphs'):
            weight = []
            for w in range(len(weights)):
                if g[edges[w][0] - 1] == i + 1:
                    weight.append(weights[w])
                elif g[edges[w][0] - 1] == i + 2:
                    break
            G_weight.append(weight)
    else:
        for i in tqdm(range(len(graph_labels)), desc="Create edges", unit='graphs'):
            edge = []
            weight = []
            for e in range(len(edges)):
                if g[edges[e][0] - 1] == i + 1:
                    edge.append(edges[e])
                    weight.append(1)
                elif g[edges[e][0] - 1] == i + 2:
                    break
            G_edges.append(edge)
            G_weight.append(weight)
    g_list = []
    for i in tqdm(range(len(G_edges)), desc="Create original graph", unit='graphs'):
        g_list.append(gen_graph(G_edges[i], G_weight[i]))

    return g_list, node_labels, slice_x


class S2VGraph(object):
    def __init__(self, edge_index, label, x):
        self.edge_index = edge_index
        self.y = label
        self.x = x

def read_tu_data_from_graph(file_path, dataname, degree_as_tag):
    with open('{}/{}/A.txt'.format(file_path, dataname), 'r') as f:
        edges = f.read().splitlines()
    edges = [tuple(map(int, e.replace(" ", "").split(","))) for e in edges]

    with open('{}/{}/graph_labels.txt'.format(file_path, dataname), 'r') as f:
        l = f.readlines()
    labels = torch.LongTensor([int(i) for i in l])

    with open('{}/{}/graph_indicator.txt'.format(file_path, dataname), 'r') as f:
        g = f.readlines()
    g = [int(i) for i in g]
    ptr = torch.cumsum(torch.bincount(torch.LongTensor(g) - 1), 0)
    ptr = torch.cat([torch.LongTensor([0]), ptr])

    with open('{}/{}/node_labels.txt'.format(file_path, dataname), 'r') as f:
        nl = f.readlines()
    one_hot_x = torch.LongTensor([int(i[-2]) for i in nl])

    edge_index, one_hot_deg = [], []
    slices = {'edge_index': [0], 'x': [0]}

    for i in tqdm(range(labels.size(0)), desc="Create edges", unit='graphs'):
        for e in range(len(edges)):
            if g[edges[e][0] - 1] == i + 1:
                edge_index.append([edges[e][0] - ptr[i] - 1, edges[e][1] - ptr[i] - 1])
            elif g[edges[e][0] - 1] == i + 2:
                break
        slices['edge_index'].append(len(edge_index))

    edge_index = torch.LongTensor(edge_index).t()
    # one_hot_x = torch.LongTensor(one_hot_deg if degree_as_tag else one_hot_x)
    feat_dim = one_hot_x.max() + 1
    one_hot_x = torch.nn.functional.one_hot(one_hot_x, feat_dim).to(torch.float)
    slices['edge_index'] = torch.LongTensor(slices['edge_index'])
    slices['x'] = ptr

    data = S2VGraph(edge_index, labels, one_hot_x)
    return data, slices