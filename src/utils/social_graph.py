import torch
from torch_geometric.data import Data
import numpy as np
import tarfile
import os
import re
import networkx as nx
import argparse
import random
from tqdm import tqdm
from src.utils.split_with_monitors import generate_tomography_dataset, graph_to_pyg

def initialize_edge_metrics(G, seed):
    np.random.seed(seed)
    metrics = ['delay', 'interaction_frequency', 'social_distance', 'trust_decay', 'information_fidelity']
    
    # Compute node centralities for social distance
    centrality = nx.eigenvector_centrality(G)
    
    for u, v in G.edges():
        G[u][v]['delay'] = np.random.uniform(0.1, 1.0)
        G[u][v]['interaction_frequency'] = np.random.uniform(0.1, 1.0)
        G[u][v]['social_distance'] = abs(centrality[u] - centrality[v]) # Given by raw_data;
        G[u][v]['trust_decay'] = np.random.uniform(0.8, 1.0)
        G[u][v]['information_fidelity'] = np.random.uniform(0.9, 1.0)
    
    # Ensure symmetry for undirected graph
    for u, v in G.edges():
        for metric in metrics:
            G[v][u][metric] = G[u][v][metric]
    
    return metrics

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

def process_ego_network(tar, ego_id, dataset_name, seed, monitor_rate):
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
    
    # Initialize edge metrics
    metrics = initialize_edge_metrics(G, seed)
    
    # Select monitors
    num_monitors = int(G.number_of_nodes() * monitor_rate)
    monitors = random.sample(list(G.nodes()), num_monitors)
    
    # Generate tomography dataset
    measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown = generate_tomography_dataset(G, monitors, metrics)
    
    # Prepare node features
    num_nodes = len(G.nodes())
    num_features = len(featnames)
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    
    for node, features in node_features.items():
        if node in node_mapping:
            x[node_mapping[node]] = torch.tensor(features, dtype=torch.float)
    
    # Add ego features
    x[node_mapping[ego_node]] = torch.tensor(ego_features, dtype=torch.float)
    
    # Use graph_to_pyg to convert the graph to PyG format
    data = graph_to_pyg(G, monitors, measurements_monitor, measurements_unknown, 
                        edge_index_monitor, edge_index_unknown, 
                        edge_attr_monitor, edge_attr_unknown)
    
    # Add node features and ego node information
    data.x = x
    data.ego_node = node_mapping[ego_node]
    
    return data, node_mapping, featnames

def process_tar_file(tar_path, output_dir, dataset_name, seed, monitor_rate):
    with tarfile.open(tar_path, 'r:gz') as tar:
        ego_ids = set()
        for member in tar.getmembers():
            if member.name.startswith(f'{dataset_name}/'):
                match = re.match(rf'{dataset_name}/(\d+)\..*', member.name)
                if match:
                    ego_ids.add(match.group(1))
        
        print(f"Found {len(ego_ids)} ego networks in the {dataset_name} dataset")
        
        for ego_id in tqdm(ego_ids, desc="Processing ego networks"):
            result = process_ego_network(tar, ego_id, dataset_name, seed, monitor_rate)
            if result is not None:
                data, _, _ = result
                output_path = os.path.join(output_dir, f"{dataset_name}_ego_net_{ego_id}.pt")
                torch.save(data, output_path)
            else:
                print(f"Skipping ego network {ego_id} due to processing error")

def main():
    parser = argparse.ArgumentParser(description="Process social network data and compute path metrics")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset ('facebook' or 'twitter')")
    parser.add_argument("--input", type=str, default="./dataset/raw_data/social_network", help="Directory containing the input tar.gz file")
    parser.add_argument("--output", type=str, default="./dataset/processed_data/social_network", help="Directory to save processed data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--monitor_rate", type=float, default=0.1, help="Ratio of nodes to be selected as monitors")
    
    args = parser.parse_args()
    
    tar_path = os.path.join(args.input, f"{args.dataset_name}.tar.gz")
    output_dir = os.path.join(args.output, f"{args.dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    process_tar_file(tar_path, output_dir, args.dataset_name, args.seed, args.monitor_rate)

if __name__ == "__main__":
    main()
