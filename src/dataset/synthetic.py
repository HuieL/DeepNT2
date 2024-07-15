import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import os
import random
import heapq
from tqdm import tqdm
import ipdb
from src.utils.split_with_monitors import generate_tomography_dataset, graph_to_pyg

def generate_connected_erdos_renyi_network(num_nodes, probability, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    return nx.erdos_renyi_graph(num_nodes, probability, seed=seed)


def initialize_edge_metrics(G):
    for u, v in G.edges():
        G[u][v]['delay'] = np.random.uniform(1, 10)
        G[u][v]['cost'] = np.random.uniform(1, 100)
        G[u][v]['reliability'] = np.random.uniform(0.9, 0.999)
        G[u][v]['bandwidth'] = np.random.uniform(10, 1000)
        G[u][v]['is_secure'] = np.random.choice([0, 1])



# Usage example:
num_nodes = 200
num_monitors = int(0.2 * num_nodes)
G = generate_connected_erdos_renyi_network(num_nodes, probability=0.02, seed=42)
print(G)
initialize_edge_metrics(G)

metrics = ['delay', 'cost', 'reliability', 'bandwidth', 'is_secure']
monitors = random.sample(list(G.nodes()), num_monitors)

measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown = generate_tomography_dataset(G, monitors, metrics)
data = graph_to_pyg(G, monitors, measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown)



