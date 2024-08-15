import torch
import numpy as np
import networkx as nx
import random
import torch.nn as nn
import torch.optim as optim
import os
import glob
import argparse
from src.utils.split_with_monitors import generate_tomography_dataset, graph_to_pyg


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_tntp_file(file_path):
    edges = []
    edge_attrs = {
        'capacity': [], 'length': [], 'flow_time': [],
        'b': [], 'power': [], 'speed': [], 'toll': [], 'link_type': []
    }
    nodes = set()
    
    with open(file_path, 'r') as file:
        data_section = False
        for line in file:
            line = line.strip()
            if line.startswith('~'):
                data_section = True
                continue
            if data_section and line and not line.startswith(('~', '<')):
                parts = line.rstrip(';').split()
                if len(parts) == 10:
                    init_node = int(parts[0])
                    term_node = int(parts[1])
                    nodes.add(init_node)
                    nodes.add(term_node)
                    edges.append([init_node, term_node])
                    
                    # Parse edge attributes
                    for i, attr in enumerate(['capacity', 'length', 'flow_time', 'b', 'power', 'speed', 'toll', 'link_type']):
                        edge_attrs[attr].append(float(parts[i+2]))
    
    num_nodes = max(nodes)
    return edges, edge_attrs, num_nodes, nodes

def create_binary_embeddings(nodes, embedding_dim=12):
    embeddings = {}
    for i, node in enumerate(sorted(nodes)):
        binary = format(i % (2**embedding_dim), f'0{embedding_dim}b')
        embedding = np.array([int(b) for b in binary])
        embeddings[node] = embedding
    return embeddings

def hope_embeddings(G, dimensions):
    A = nx.adjacency_matrix(G)
    n = A.shape[0]
    
    beta = 0.1  # decay factor
    S = np.linalg.inv(np.eye(n) - beta * A.toarray()) - np.eye(n)
    
    u, s, vt = np.linalg.svd(S)
    sqrt_s = np.sqrt(s[:dimensions])
    X = u[:, :dimensions] * sqrt_s
    Y = vt[:dimensions, :].T * sqrt_s
    
    embeddings = np.hstack([X, Y])
    return {node: embeddings[i] for i, node in enumerate(G.nodes())}

def random_walk(G, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        curr = walk[-1]
        neighbors = list(G.neighbors(curr))
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    return walk

def generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, node, walk_length))
    return walks

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, target, context):
        target_emb = self.embeddings(target)
        context_emb = self.output(context)
        return torch.sum(target_emb * context_emb, dim=1)

def walk_embeddings(walks, embedding_dim, num_epochs=5, batch_size=128, window_size=5, num_negative=5):
    vocab = list(set(node for walk in walks for node in walk))
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    
    def get_context(walk, idx, window_size):
        start = max(0, idx - window_size)
        end = min(len(walk), idx + window_size + 1)
        context = [word_to_ix[walk[i]] for i in range(start, end) if i != idx]
        return context

    dataset = []
    for walk in walks:
        for i, word in enumerate(walk):
            context = get_context(walk, i, window_size)
            word_ix = word_to_ix[word]
            dataset.extend([(word_ix, ctx) for ctx in context])

    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        random.shuffle(dataset)
        total_loss = 0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            target = torch.tensor([pair[0] for pair in batch])
            context = torch.tensor([pair[1] for pair in batch])
            
            neg_context = torch.randint(0, vocab_size, (len(batch), num_negative))
            
            pos_score = model(target, context)
            neg_score = model(target.repeat_interleave(num_negative), neg_context.view(-1))
            
            pos_loss = torch.mean(-torch.log(torch.sigmoid(pos_score) + 1e-10))
            neg_loss = torch.mean(-torch.log(1 - torch.sigmoid(neg_score) + 1e-10))
            loss = pos_loss + neg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    embeddings = {vocab[i]: model.embeddings.weight[i].detach().numpy() for i in range(vocab_size)}
    return embeddings

def create_embeddings(edges, nodes, embedding_type, embedding_dim, num_walks, walk_length, seed):
    set_random_seed(seed)
    if embedding_type == 'binary':
        return create_binary_embeddings(nodes, embedding_dim)
    elif embedding_type == 'hope':
        G = nx.Graph(edges)
        return hope_embeddings(G, embedding_dim // 2)  # HOPE returns double the dimension
    elif embedding_type in ['randomwalk', 'deepwalk']:
        G = nx.Graph(edges)
        walks = generate_walks(G, num_walks, walk_length)
        return walk_embeddings(walks, embedding_dim)
    else:
        raise ValueError("Invalid embedding type. Choose 'binary', 'hope', 'randomwalk', or 'deepwalk'.")

def update_graph(graph):
    graph.edge_index = graph.edge_index - 1
    for attr in ['node_pair_monitor', 'node_pair_unknown']:
        if hasattr(graph, attr):
            setattr(graph, attr, {k: v - 1 for k, v in getattr(graph, attr).items()})

    graph.original_node_ids = list(range(1, graph.num_nodes + 1))
    return graph

def create_pyg_graph(edges, edge_attrs, num_nodes, node_embeddings, sampling_rate, seed):
    set_random_seed(seed)
    node_mapping = {node: idx for idx, node in enumerate(range(1, num_nodes + 1))}
    
    x = torch.zeros((num_nodes, len(next(iter(node_embeddings.values())))), dtype=torch.float)
    for node, embedding in node_embeddings.items():
        if node in node_mapping:
            x[node_mapping[node]] = torch.tensor(embedding, dtype=torch.float)
    
    # Create a NetworkX graph for tomography dataset generation
    G = nx.Graph()
    for edge, capacity, length, flow_time in zip(edges, edge_attrs['capacity'], edge_attrs['length'], edge_attrs['flow_time']):
        G.add_edge(edge[0], edge[1], capacity=capacity, length=length, flow_time=flow_time)
    
    # Generate tomography dataset
    metrics = ['capacity', 'length', 'flow_time']
    measurements_monitor, measurements_unknown, node_pair_monitor, node_pair_unknown, path_attr_monitor, path_attr_unknown = generate_tomography_dataset(G, sampling_rate, metrics, seed)
    
    # Use graph_to_pyg to convert the graph to PyG format
    data = graph_to_pyg(G, measurements_monitor, measurements_unknown, 
                        node_pair_monitor, node_pair_unknown, 
                        path_attr_monitor, path_attr_unknown)
    
    # Add node features
    data.x = x
    
    # Add original edge attributes
    for attr_name, attr_values in edge_attrs.items():
        setattr(data, f'edge_{attr_name}', torch.tensor(attr_values, dtype=torch.float))
    
    data.original_node_ids = list(range(1, num_nodes + 1))
    
    return data, node_mapping

def tntp_to_pyg(file_path, embedding_type, embedding_dim, num_walks, walk_length, sampling_rate, seed):
    set_random_seed(seed)
    edges, edge_attrs, num_nodes, nodes = parse_tntp_file(file_path)
    node_embeddings = create_embeddings(edges, nodes, embedding_type, embedding_dim, num_walks, walk_length, seed)
    graph, node_mapping = create_pyg_graph(edges, edge_attrs, num_nodes, node_embeddings, sampling_rate, seed)
    return graph, node_mapping

def process_tntp_files(input_folder, output_folder, embedding_type, embedding_dim, num_walks, walk_length, sampling_rate, seed):
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all .tntp files in the input folder
    tntp_files = glob.glob(os.path.join(input_folder, '*.tntp'))
    
    for file_path in tntp_files:
        # Get the filename without extension
        file_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(file_name)[0]

        output_file = os.path.join(output_folder, f"{name_without_ext}_{sampling_rate}_{embedding_type}.pt")
        if os.path.exists(output_file):
            print(f"Cached: {output_file}")
            continue
        graph, _ = tntp_to_pyg(file_path, embedding_type, embedding_dim, num_walks, walk_length, sampling_rate, seed)
        
        torch.save(update_graph(graph), output_file)
        print(f"Processed and saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert TNTP files to PyG graphs")
    parser.add_argument("--input", type=str, default="./dataset/raw_data/transportation_network",
                        help="Input folder containing TNTP files")
    parser.add_argument("--output", type=str, default="./dataset/processed_data/transportation_network",
                        help="Output folder for processed PyG graphs")
    parser.add_argument("--embedding_type", type=str, default="binary",
                        choices=["binary", "hope", "randomwalk", "deepwalk"],
                        help="Type of node embedding to use")
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="Dimension of node embeddings")
    parser.add_argument("--num_walks", type=int, default=10,
                        help="Number of random walks per node (for randomwalk and deepwalk)")
    parser.add_argument("--walk_length", type=int, default=80,
                        help="Length of each random walk (for randomwalk and deepwalk)")
    parser.add_argument("--sampling_rate", type=float, default=0.1,
                        help="Sampling rate for observed node pairs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()

    process_tntp_files(args.input, args.output, args.embedding_type, 
                        args.embedding_dim, args.num_walks, args.walk_length, 
                        args.sampling_rate, args.seed)

if __name__ == "__main__":
    main()
