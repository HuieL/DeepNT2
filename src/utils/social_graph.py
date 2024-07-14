import torch
from torch_geometric.data import Data
import numpy as np
import tarfile
import os
import re
import networkx as nx
import argparse
import random


def read_edges(file_content):
    edges = []
    for line in file_content.decode().split('\n'):
        if line.strip():
            source, target = map(int, line.strip().split())
            edges.append([source, target])
    return edges

def read_circles(file_content):
    circles = {}
    for line in file_content.decode().split('\n'):
        if line.strip():
            parts = line.strip().split()
            circle_name = parts[0]
            members = list(map(int, parts[1:]))
            circles[circle_name] = members
    return circles

def read_features(file_content):
    features = {}
    for line in file_content.decode().split('\n'):
        if line.strip():
            values = list(map(int, line.strip().split()))
            node_id = values[0]
            feature_vector = values[1:]
            features[node_id] = feature_vector
    return features

def read_featnames(file_content):
    featnames = []
    for line in file_content.decode().split('\n'):
        if line.strip():
            parts = line.strip().split()
            feature_id = int(parts[0])
            feature_name = ' '.join(parts[1:])
            featnames.append((feature_id, feature_name))
    return featnames

def read_ego_features(file_content):
    return list(map(int, file_content.decode().strip().split()))

def compute_path_metrics(G, seed):
    """
    Compute path-based metrics that satisfy additive or multiplicative properties.
    """
    np.random.seed(seed)
    metrics = {
        'propagation_delay': {},
        'trust_decay': {},
        'inverse_bandwidth': {},
        'information_fidelity': {},
        'social_distance': {},
        'inverse_interaction_frequency': {}
    }
    
    # Compute node centralities for social distance
    centrality = nx.eigenvector_centrality(G)
    
    # Assign base values to edges
    for u, v in G.edges():
        metrics['propagation_delay'][(u, v)] = np.random.uniform(0.5, 1.5)
        metrics['trust_decay'][(u, v)] = np.random.uniform(0.8, 1.0)
        metrics['inverse_bandwidth'][(u, v)] = np.random.uniform(0.1, 1.0)
        metrics['information_fidelity'][(u, v)] = np.random.uniform(0.9, 1.0)
        metrics['social_distance'][(u, v)] = abs(centrality[u] - centrality[v])
        metrics['inverse_interaction_frequency'][(u, v)] = np.random.uniform(0.1, 1.0)
    
    # Ensure symmetry for undirected graph
    for metric in metrics.values():
        for (u, v) in list(metric.keys()):
            metric[(v, u)] = metric[(u, v)]
    
    # Verify and adjust additive metrics
    additive_metrics = ['propagation_delay', 'inverse_bandwidth', 'social_distance', 'inverse_interaction_frequency']
    for metric_name in additive_metrics:
        for path in nx.all_pairs_dijkstra_path(G):
            for i in range(len(path) - 2):
                for j in range(i + 2, len(path)):
                    direct = metrics[metric_name].get((path[i], path[j]), float('inf'))
                    indirect = sum(metrics[metric_name][(path[k], path[k+1])] for k in range(i, j))
                    metrics[metric_name][(path[i], path[j])] = metrics[metric_name][(path[j], path[i])] = min(direct, indirect)
    
    # Verify and adjust multiplicative metrics
    multiplicative_metrics = ['trust_decay', 'information_fidelity']
    for metric_name in multiplicative_metrics:
        for path in nx.all_pairs_dijkstra_path(G):
            for i in range(len(path) - 2):
                for j in range(i + 2, len(path)):
                    direct = metrics[metric_name].get((path[i], path[j]), 0)
                    indirect = np.prod([metrics[metric_name][(path[k], path[k+1])] for k in range(i, j)])
                    metrics[metric_name][(path[i], path[j])] = metrics[metric_name][(path[j], path[i])] = max(direct, indirect)
    
    return metrics

def process_ego_network(tar, ego_id, dataset_name, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        edges = read_edges(tar.extractfile(f"{dataset_name}/{ego_id}.edges").read())
        circles = read_circles(tar.extractfile(f"{dataset_name}/{ego_id}.circles").read())
        node_features = read_features(tar.extractfile(f"{dataset_name}/{ego_id}.feat").read())
        ego_features = read_ego_features(tar.extractfile(f"{dataset_name}/{ego_id}.egofeat").read())
        featnames = read_featnames(tar.extractfile(f"{dataset_name}/{ego_id}.featnames").read())
    except KeyError as e:
        print(f"Error processing ego network {ego_id}: {e}")
        return None
    
    # Create a networkx graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Add ego node and its connections
    ego_node = int(ego_id)
    G.add_node(ego_node)
    G.add_edges_from([(ego_node, node) for node in G.nodes() if node != ego_node])
    
    # Create a mapping from original node IDs to zero-indexed IDs
    original_nodes = sorted(G.nodes())
    node_mapping = {node: i for i, node in enumerate(original_nodes)}
    reverse_mapping = {i: node for node, i in node_mapping.items()}
    
    # Compute path metrics
    edge_metrics = compute_path_metrics(G, seed)
    
    # Prepare node features
    num_nodes = len(G.nodes())
    num_features = len(featnames)
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    
    for node, features in node_features.items():
        if node in node_mapping:
            x[node_mapping[node]] = torch.tensor(features, dtype=torch.float)
    
    # Add ego features
    x[node_mapping[ego_node]] = torch.tensor(ego_features, dtype=torch.float)
    
    # Prepare edge index and edge attributes
    edge_index = []
    edge_attr = {metric: [] for metric in edge_metrics.keys()}
    
    for u, v in G.edges():
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_index.append([node_mapping[v], node_mapping[u]])  # Add reverse edge for undirected graph
        for metric, values in edge_metrics.items():
            edge_attr[metric].append(values[(u, v)])
            edge_attr[metric].append(values[(v, u)])  # Add value for reverse edge
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    data.ego_node = node_mapping[ego_node]
    
    # Add each metric as a separate edge attribute
    for metric_name, metric_values in edge_attr.items():
        setattr(data, f'edge_{metric_name}', torch.tensor(metric_values, dtype=torch.float))
    
    return data, node_mapping, featnames

def process_tar_file(tar_path, output_dir, dataset_name, seed):
    with tarfile.open(tar_path, 'r:gz') as tar:
        ego_ids = set()
        for member in tar.getmembers():
            if member.name.startswith(f'{dataset_name}/'):
                match = re.match(rf'{dataset_name}/(\d+)\..*', member.name)
                if match:
                    ego_ids.add(match.group(1))
        
        print(f"Found {len(ego_ids)} ego networks in the {dataset_name} dataset")
        
        for ego_id in ego_ids:
            print(f"Processing ego network {ego_id}")
            result = process_ego_network(tar, ego_id, dataset_name, seed)
            if result is not None:
                data, _, _ = result
                output_path = os.path.join(output_dir, f"{dataset_name}_ego_net_{ego_id}.pt")
                torch.save(data, output_path)
            else:
                print(f"Skipping ego network {ego_id} due to processing error")


def main():
    parser = argparse.ArgumentParser(description="Process social network data and compute path metrics")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset ('facebook' or 'twitter")
    parser.add_argument("--input", type=str, default="./dataset/raw_data/social_network", help="Directory containing the input tar.gz file")
    parser.add_argument("--output", type=str, default="./dataset/processed_data/social_network", help="Directory to save processed data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    tar_path = os.path.join(args.input, f"{args.dataset_name}.tar.gz")
    output_dir = os.path.join(args.output, f"{args.dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    process_tar_file(tar_path, output_dir, args.dataset_name, args.seed)

if __name__ == "__main__":
    main()

