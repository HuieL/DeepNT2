import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import os
import random
import heapq
from tqdm import tqdm

def maximum_bandwidth_path(G, source, target):
    def dijkstra_max_bandwidth(G, source, target):
        bandwidth = {source: float('inf')}
        heap = [(-float('inf'), source)]
        path = {source: [source]}
        
        while heap:
            (neg_bw, node) = heapq.heappop(heap)
            bw = -neg_bw
            
            if node == target:
                return path[node], bandwidth[node]
            
            for neighbor in G.neighbors(node):
                edge_bw = G[node][neighbor]['bandwidth']
                new_bw = min(bw, edge_bw)
                if new_bw > bandwidth.get(neighbor, 0):
                    bandwidth[neighbor] = new_bw
                    heapq.heappush(heap, (-new_bw, neighbor))
                    path[neighbor] = path[node] + [neighbor]
        
        return None, 0  # No path found

    path, max_bw = dijkstra_max_bandwidth(G, source, target)
    return path

def calculate_path_performance(G, path, metric):
    if path is None:
        return None
    if metric in ['delay', 'cost']:
        return sum(G[u][v][metric] for u, v in zip(path[:-1], path[1:]))
    elif metric == 'reliability':
        return np.prod([G[u][v][metric] for u, v in zip(path[:-1], path[1:])])
    elif metric == 'bandwidth':
        return min(G[u][v][metric] for u, v in zip(path[:-1], path[1:]))
    elif metric == 'is_secure':
        return all(G[u][v][metric] for u, v in zip(path[:-1], path[1:]))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
def best_performance_routing(G, source, target, metric):
    if metric in ['delay', 'cost']:
        return nx.shortest_path(G, source, target, weight=metric)
    elif metric == 'reliability':
        return nx.shortest_path(G, source, target, weight=lambda u, v, d: -np.log(d['reliability']))
    elif metric == 'bandwidth':
        return maximum_bandwidth_path(G, source, target)
    elif metric == 'is_secure':
        secure_graph = nx.Graph((u, v, d) for (u, v, d) in G.edges(data=True) if d['is_secure'])
        if source in secure_graph and target in secure_graph:
            if nx.has_path(secure_graph, source, target):
                return nx.shortest_path(secure_graph, source, target)
        return None  # No secure path exists
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def generate_tomography_dataset(G, monitors, metrics):
    measurements_monitor = {}
    measurements_unknown = {}
    edge_index_monitor = []
    edge_index_unknown = []
    edge_attr_monitor = {metric: [] for metric in metrics}
    edge_attr_unknown = {metric: [] for metric in metrics}
    
    nodes = list(G.nodes())
    
    for i, source in enumerate(tqdm(monitors, desc="Processing monitors")):
        for j, target in enumerate(nodes):
            if source != target:
                for metric in metrics:
                    try:
                        path = best_performance_routing(G, source, target, metric)
                        if path is not None:
                            performance = calculate_path_performance(G, path, metric)
                            if target in monitors:
                                measurements_monitor[(source, target, metric)] = performance
                                edge_index_monitor.append([i, j])
                                edge_attr_monitor[metric].append(performance)
                            else:
                                measurements_unknown[(source, target, metric)] = performance
                                edge_index_unknown.append([i, j])
                                edge_attr_unknown[metric].append(performance)
                        else:
                            if target in monitors:
                                measurements_monitor[(source, target, metric)] = np.nan
                            else:
                                measurements_unknown[(source, target, metric)] = np.nan
                    except nx.NetworkXNoPath:
                        if target in monitors:
                            measurements_monitor[(source, target, metric)] = np.nan
                        else:
                            measurements_unknown[(source, target, metric)] = np.nan

    return measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown


# Use_example
# metrics = ['delay', 'cost', 'reliability', 'bandwidth', 'is_secure']
# monitors = random.sample(list(G.nodes()), num_monitors)

# measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown = generate_tomography_dataset(G, monitors, metrics)
