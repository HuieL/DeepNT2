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
                edge_bw = G[node][neighbor].get('bandwidth', G[node][neighbor].get('capacity', 0))
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
    if metric in ['delay', 'cost', 'rtt', 'interaction_frequency', 'social_distance', 'length', 'flow_time']:
        return sum(G[u][v].get(metric, 0) for u, v in zip(path[:-1], path[1:]))
    elif metric in ['reliability', 'trust_decay', 'information_fidelity']:
        return np.prod([G[u][v].get(metric, 1) for u, v in zip(path[:-1], path[1:])])
    elif metric in ['bandwidth', 'capacity']:
        return min(G[u][v].get(metric, 0) for u, v in zip(path[:-1], path[1:]))
    elif metric == 'is_secure':
        return all(G[u][v].get(metric, False) for u, v in zip(path[:-1], path[1:]))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def best_performance_routing(G, source, target, metric):
    if metric in ['delay', 'cost', 'rtt', 'interaction_frequency', 'social_distance', 'length', 'flow_time']:
        return nx.shortest_path(G, source, target, weight=metric)
    elif metric in ['reliability', 'trust_decay', 'information_fidelity']:
        return nx.shortest_path(G, source, target, weight=lambda u, v, d: -np.log(d.get(metric, 1)))
    elif metric in ['bandwidth', 'capacity']:
        return maximum_bandwidth_path(G, source, target)
    elif metric == 'is_secure':
        secure_graph = nx.Graph((u, v, d) for (u, v, d) in G.edges(data=True) if d.get('is_secure', False))
        if source in secure_graph and target in secure_graph:
            if nx.has_path(secure_graph, source, target):
                return nx.shortest_path(secure_graph, source, target)
        return None  # No secure path exists
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def generate_tomography_dataset(G, sampling_rate, metrics, seed):
    random.seed(seed)
    num_nodes = G.number_of_nodes()
    total_pairs = num_nodes * (num_nodes - 1)  # Total number of possible node pairs
    target_samples = int(total_pairs * sampling_rate)
    
    # Calculate number of monitors
    num_monitors = 1
    while num_monitors * (num_nodes - 1) < target_samples:
        num_monitors += 1
    
    # Randomly select monitors
    monitors = random.sample(list(G.nodes()), num_monitors)
    monitor_set = set(monitors)
    
    measurements_monitor = {metric: {} for metric in metrics}
    measurements_unknown = {metric: {} for metric in metrics}
    node_pair_monitor = {metric: set() for metric in metrics}
    node_pair_unknown = {metric: set() for metric in metrics}
    path_attr_monitor = {metric: [] for metric in metrics}
    path_attr_unknown = {metric: [] for metric in metrics}
    
    nodes = list(G.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    
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
                                node_pair_monitor[metric].add((node_to_index[source], node_to_index[target]))
                                path_attr_monitor[metric].append(performance)
                            else:
                                measurements_unknown[metric][(source, target)] = performance
                                node_pair_unknown[metric].add((node_to_index[source], node_to_index[target]))
                                path_attr_unknown[metric].append(performance)
                    except nx.NetworkXNoPath:
                        if target in monitor_set:
                            measurements_monitor[metric][(source, target)] = np.nan
                        else:
                            measurements_unknown[metric][(source, target)] = np.nan

    return (measurements_monitor, measurements_unknown, 
            {m: list(e) for m, e in node_pair_monitor.items()}, 
            {m: list(e) for m, e in node_pair_unknown.items()}, 
            path_attr_monitor, path_attr_unknown)

def graph_to_pyg(G, measurements_monitor, measurements_unknown, node_pair_monitor, node_pair_unknown, path_attr_monitor, path_attr_unknown):
    num_nodes = G.number_of_nodes()
    
    data = Data()
    data.num_nodes = num_nodes
    
    # Original graph structure
    data.edge_index = torch.tensor(list(G.edges())).t().contiguous()
    
    # Monitor paths (training data)
    data.node_pair_monitor = {
        metric: torch.tensor(edges).t().contiguous()
        for metric, edges in node_pair_monitor.items()}
    
    # Unknown paths (test data)
    data.node_pair_unknown = {
        metric: torch.tensor(edges).t().contiguous()
        for metric, edges in node_pair_unknown.items()}
    
    # Metrics
    data.metrics_monitor = {
        metric: torch.tensor(values, dtype=torch.float) 
        for metric, values in path_attr_monitor.items()}
    
    data.metrics_unknown = {
        metric: torch.tensor(values, dtype=torch.float) 
        for metric, values in path_attr_unknown.items()}
    
    # Store measurements
    data.measurements_monitor = measurements_monitor
    data.measurements_unknown = measurements_unknown
    
    return data
