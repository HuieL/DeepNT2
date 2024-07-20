import networkx as nx
import numpy as np
import torch
import os
import random
from tqdm import tqdm
import argparse
from src.utils.split_with_monitors import generate_tomography_dataset, graph_to_pyg
import torch.nn as nn
import torch.optim as optim


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

def random_walk(G, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        curr = walk[-1]
        neighbors = list(G.neighbors(curr))
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    return walk

def deepwalk(G, start_node, walk_length, p=1, q=1):
    walk = [start_node]
    for _ in range(walk_length - 1):
        curr = walk[-1]
        neighbors = list(G.neighbors(curr))
        if len(neighbors) == 0:
            break
        if len(walk) == 1:
            walk.append(random.choice(neighbors))
        else:
            prev = walk[-2]
            next_node = deepwalk_next(G, prev, curr, neighbors, p, q)
            walk.append(next_node)
    return walk

def deepwalk_next(G, prev, curr, neighbors, p, q):
    probs = []
    for neighbor in neighbors:
        if neighbor == prev:
            probs.append(1/p)
        elif G.has_edge(neighbor, prev):
            probs.append(1.0)
        else:
            probs.append(1/q)
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()
    return np.random.choice(neighbors, p=probs)

def generate_walks(G, num_walks, walk_length, walk_type='random', p=1, q=1):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            if walk_type == 'random':
                walks.append(random_walk(G, node, walk_length))
            elif walk_type == 'deepwalk':
                walks.append(deepwalk(G, node, walk_length, p, q))
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
    
    embeddings = {vocab[i]: model.embeddings.weight[i].detach().numpy() for i in range(vocab_size)}
    return embeddings

def hope_embeddings(G, dimensions):
    A = nx.adjacency_matrix(G)
    n = A.shape[0]
    
    beta = 0.1
    S = np.linalg.inv(np.eye(n) - beta * A.toarray()) - np.eye(n)
    
    u, s, vt = np.linalg.svd(S)
    sqrt_s = np.sqrt(s[:dimensions])
    X = u[:, :dimensions] * sqrt_s
    Y = vt[:dimensions, :].T * sqrt_s
    
    embeddings = np.hstack([X, Y])
    return {node: embeddings[i] for i, node in enumerate(G.nodes())}

def binary_coding_embeddings(G):
    embeddings = {}
    for i, node in enumerate(G.nodes()):
        binary = format(i % (2**16), '016b')
        embedding = np.array([int(b) for b in binary])
        embeddings[node] = embedding
    
    return embeddings

def generate_and_save_graphs(num_graphs, num_nodes, probability, sampling_rate, output_dir, start_seed, embedding_type, num_walks, walk_length, p, q, embedding_dim):
    os.makedirs(output_dir, exist_ok=True)
    
    for seed in tqdm(range(start_seed, start_seed + num_graphs), desc="Generating graphs"):
        np.random.seed(seed)
        random.seed(seed)
        
        G = generate_connected_erdos_renyi_network(num_nodes, probability, seed)
        initialize_edge_metrics(G)
        
        metrics = ['delay', 'cost', 'reliability', 'bandwidth', 'is_secure']
        measurements_monitor, measurements_unknown, node_pair_monitor, node_pair_unknown, path_attr_monitor, path_attr_unknown = generate_tomography_dataset(G, sampling_rate, metrics, seed)
        
        data = graph_to_pyg(G, measurements_monitor, measurements_unknown, 
                            node_pair_monitor, node_pair_unknown, 
                            path_attr_monitor, path_attr_unknown)
        
        for attr in ['delay', 'cost', 'reliability', 'bandwidth', 'is_secure']:
            edge_attr = [G[u][v][attr] for u, v in G.edges()]
            setattr(data, f'edge_{attr}', torch.tensor(edge_attr, dtype=torch.float))
        
        # Generate node embeddings
        if embedding_type in ['randomwalk', 'deepwalk']:
            walks = generate_walks(G, num_walks, walk_length, embedding_type, p, q)
            embeddings = walk_embeddings(walks, embedding_dim)
        elif embedding_type == 'binary':
            embeddings = binary_coding_embeddings(G)
        elif embedding_type == 'hope':
            embeddings = hope_embeddings(G, embedding_dim)
        else:
            raise ValueError("Invalid embedding type. Choose 'randomwalk', 'deepwalk', 'hope', or 'binary'.")

        # Add embeddings to the data object
        data.x = torch.tensor(list(embeddings.values()), dtype=torch.float)
        
        output_file = os.path.join(output_dir, f"graph_seed_{seed}.pt")
        torch.save(data, output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate multiple Erdos-Renyi graphs with metrics and node embeddings")
    parser.add_argument("--num_graphs", type=int, default=1, help="Number of graphs to generate")
    parser.add_argument("--num_nodes", type=int, default=500, help="Number of nodes in each graph")
    parser.add_argument("--probability", type=float, default=0.01, help="Edge probability for Erdos-Renyi model")
    parser.add_argument("--sampling_rate", type=float, default=0.1, help="Sampling rate for observed node pairs")
    parser.add_argument("--output", type=str, default="./dataset/processed_data/synthetic", help="Output directory for generated graphs")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed for graph generation")
    parser.add_argument("--embedding_type", type=str, default="binary", choices=["randomwalk", "deepwalk", "hope", "binary"], help="Type of node embedding to use")
    parser.add_argument("--num_walks", type=int, default=10, help="Number of random walks per node")
    parser.add_argument("--walk_length", type=int, default=512, help="Length of each random walk")
    parser.add_argument("--p", type=float, default=1, help="Return parameter for deepwalk")
    parser.add_argument("--q", type=float, default=1, help="In-out parameter for deepwalk")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of node embeddings")
    
    args = parser.parse_args()
    
    generate_and_save_graphs(args.num_graphs, args.num_nodes, args.probability, 
                             args.sampling_rate, args.output, args.start_seed,
                             args.embedding_type, args.num_walks, args.walk_length,
                             args.p, args.q, args.embedding_dim)

if __name__ == "__main__":
    main()
