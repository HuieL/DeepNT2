import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import os
import random
import heapq
from tqdm import tqdm
import ipdb
import argparse
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

def generate_and_save_graphs(num_graphs, num_nodes, probability, monitor_rate, output_dir, start_seed):
    os.makedirs(output_dir, exist_ok=True)
    
    for seed in tqdm(range(start_seed, start_seed + num_graphs), desc="Generating graphs"):
        # Set the seed for this graph
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate the graph
        G = generate_connected_erdos_renyi_network(num_nodes, probability, seed=seed)
        
        # Initialize edge metrics
        initialize_edge_metrics(G)
        
        # Select monitors
        num_monitors = max(1, int(num_nodes * monitor_rate))
        monitors = random.sample(list(G.nodes()), num_monitors)
        
        # Generate tomography dataset
        metrics = ['delay', 'cost', 'reliability', 'bandwidth', 'is_secure']
        measurements_monitor, measurements_unknown, edge_index_monitor, edge_index_unknown, edge_attr_monitor, edge_attr_unknown = generate_tomography_dataset(G, monitors, metrics)
        
        # Convert to PyG format
        data = graph_to_pyg(G, monitors, measurements_monitor, measurements_unknown, 
                            edge_index_monitor, edge_index_unknown, 
                            edge_attr_monitor, edge_attr_unknown)
        
        # Add original edge attributes
        for attr in ['delay', 'cost', 'reliability', 'bandwidth', 'is_secure']:
            edge_attr = [G[u][v][attr] for u, v in G.edges()]
            setattr(data, f'edge_{attr}', torch.tensor(edge_attr, dtype=torch.float))
        
        # Save the graph
        output_file = os.path.join(output_dir, f"graph_seed_{seed}.pt")
        torch.save(data, output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate multiple Erdos-Renyi graphs with metrics")
    parser.add_argument("--num_graphs", type=int, default=10, help="Number of graphs to generate")
    parser.add_argument("--num_nodes", type=int, default=1000, help="Number of nodes in each graph")
    parser.add_argument("--probability", type=float, default=0.01, help="Edge probability for Erdos-Renyi model")
    parser.add_argument("--monitor_rate", type=float, default=0.1, help="Ratio of nodes to be selected as monitors")
    parser.add_argument("--output", type=str, default="./dataset/processed_data/synthenic", help="Output directory for generated graphs")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed for graph generation")
    
    args = parser.parse_args()
    
    generate_and_save_graphs(args.num_graphs, args.num_nodes, args.probability, 
                             args.monitor_rate, args.output, args.start_seed)

if __name__ == "__main__":
    main()
