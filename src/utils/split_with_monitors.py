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
    measurements_monitor = {metric: {} for metric in metrics}
    measurements_unknown = {metric: {} for metric in metrics}
    edge_index_monitor = {metric: set() for metric in metrics}
    edge_index_unknown = {metric: set() for metric in metrics}
    edge_attr_monitor = {metric: [] for metric in metrics}
    edge_attr_unknown = {metric: [] for metric in metrics}
    
    nodes = list(G.nodes())
    monitor_set = set(monitors)
    monitor_to_index = {monitor: idx for idx, monitor in enumerate(monitors)}
    
    for source in tqdm(monitors, desc="Processing monitors"):
        for target in nodes:
            if source != target:
                for metric in metrics:
                    try:
                        path = best_performance_routing(G, source, target, metric)
                        if path is not None:
                            performance = calculate_path_performance(G, path, metric)
                            if target in monitor_set:
                                measurements_monitor[metric][(source, target)] = performance
                                edge_index_monitor[metric].add((monitor_to_index[source], monitor_to_index[target]))
                                edge_attr_monitor[metric].append(performance)
                            else:
                                measurements_unknown[metric][(source, target)] = performance
                                edge_index_unknown[metric].add((monitor_to_index[source], nodes.index(target)))
                                edge_attr_unknown[metric].append(performance)
                    except nx.NetworkXNoPath:
                        if target in monitor_set:
                            measurements_monitor[metric][(source, target)] = np.nan
                        else:
                            measurements_unknown[metric][(source, target)] = np.nan

    return (measurements_monitor, measurements_unknown, 
            {m: list(e) for m, e in edge_index_monitor.items()}, 
            {m: list(e) for m, e in edge_index_unknown.items()}, 
            edge_attr_monitor, edge_attr_unknown)

def graph_to_pyg(G, monitors, measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown):
    num_nodes = G.number_of_nodes()
    num_monitors = len(monitors)
    
    data = Data()
    data.num_nodes = num_nodes
    data.num_monitors = num_monitors
    
    # Original graph structure
    data.edge_index = torch.tensor(list(G.edges())).t().contiguous()
    
    # Monitor paths (training data)
    data.edge_index_monitor = {
        metric: torch.tensor(edges).t().contiguous()
        for metric, edges in edge_index_monitor.items()}
    
    # Unknown paths (test data)
    data.edge_index_unknown = {
        metric: torch.tensor(edges).t().contiguous()
        for metric, edges in edge_index_unknown.items()}
    
    # Metrics
    data.metrics_monitor = {
        metric: torch.tensor(values, dtype=torch.float) 
        for metric, values in edge_attr_monitor.items()}
    
    data.metrics_monitor = {
        metric: torch.tensor(values, dtype=torch.float) 
        for metric, values in edge_attr_unknown.items()}
    
    # Store measurements
    data.measurements_monitor = measurements_monitor
    data.measurements_unknown = measurements_unknown
    
    return data


# Usage example:
# metrics = ['delay', 'cost', 'reliability', 'bandwidth', 'is_secure']
# monitors = random.sample(list(G.nodes()), num_monitors)

# measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown = generate_tomography_dataset(G, monitors, metrics)
# data = graph_to_pyg(G, monitors, measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown)

