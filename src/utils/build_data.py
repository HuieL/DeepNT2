import warts
from warts.traceroute import Traceroute
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

def random_walk(G, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        curr = walk[-1]
        neighbors = list(G.neighbors(curr))
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    return walk

def node2vec_walk(G, start_node, walk_length, p=1, q=1):
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
            next_node = node2vec_next(G, prev, curr, neighbors, p, q)
            walk.append(next_node)
    return walk

def node2vec_next(G, prev, curr, neighbors, p, q):
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
                walks.append(node2vec_walk(G, node, walk_length, p, q))
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
    # Create vocabulary
    vocab = list(set(node for walk in walks for node in walk))
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    
    # Create dataset
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

    # Initialize model
    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        random.shuffle(dataset)
        total_loss = 0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            target = torch.tensor([pair[0] for pair in batch])
            context = torch.tensor([pair[1] for pair in batch])
            
            # Negative sampling
            neg_context = torch.randint(0, vocab_size, (len(batch), num_negative))
            
            # Forward pass
            pos_score = model(target, context)
            neg_score = model(target.repeat_interleave(num_negative), neg_context.view(-1))
            
            # Manual implementation of log sigmoid
            pos_loss = torch.mean(-torch.log(torch.sigmoid(pos_score) + 1e-10))
            neg_loss = torch.mean(-torch.log(1 - torch.sigmoid(neg_score) + 1e-10))
            loss = pos_loss + neg_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # Extract embeddings
    embeddings = {vocab[i]: model.embeddings.weight[i].detach().numpy() for i in range(vocab_size)}
    return embeddings
    
def binary_coding_embeddings(G):
     # Fixed 16-bit binary code
    embeddings = {}
    for i, node in enumerate(G.nodes()):
        # Convert node index to 16-bit binary
        binary = format(i % (2**16), '016b')
        embedding = np.array([int(b) for b in binary])
        embeddings[node] = embedding
    
    return embeddings

def warts_to_graph(warts_file, max_records=None, num_walks=10, walk_length=80, embedding_type='randomwalk', p=1, q=1, embedding_dim=64):
    G = nx.DiGraph()
    node_mapping = {}  # Map IP addresses to integer indices
    node_counter = 0

    with open(warts_file, 'rb') as f:
        record_count = 0
        while max_records is None or record_count < max_records:
            try:
                record = warts.parse_record(f)
                if isinstance(record, Traceroute):
                    src = record.src_address
                    if src not in node_mapping:
                        node_mapping[src] = node_counter
                        node_counter += 1
                    
                    prev_node = src
                    for hop in record.hops:
                        if hop.address:
                            if hop.address not in node_mapping:
                                node_mapping[hop.address] = node_counter
                                node_counter += 1
                            
                            G.add_edge(node_mapping[prev_node], node_mapping[hop.address], rtt=hop.rtt)
                            prev_node = hop.address
                    
                    record_count += 1
            except EOFError:
                break
            except Exception as e:
                print(f"Error processing record: {e}")
                continue

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Generate embeddings based on the selected type
    if embedding_type in ['randomwalk', 'deepwalk']:
        walks = generate_walks(G, num_walks, walk_length, embedding_type, p, q)
        embeddings = walk_embeddings(walks, embedding_dim)
    elif embedding_type == 'binary':
        embeddings = binary_coding_embeddings(G)
    else:
        raise ValueError("Invalid embedding type. Choose 'randomwalk', 'deepwalk', or 'binary'.")

    print(f"Generated embeddings for {len(embeddings)} nodes")

    # Convert NetworkX graph to PyG Data
    pyg_graph = from_networkx(G)
    
    # Add node embeddings as features
    node_features = torch.FloatTensor([embeddings[node] for node in G.nodes()])
    pyg_graph.x = node_features

    # Add node IDs (IP addresses) as a node attribute
    node_ids = [None] * len(node_mapping)
    for ip, idx in node_mapping.items():
        node_ids[idx] = ip
    pyg_graph.node_ids = node_ids

    # Convert edge attributes (RTT) to a tensor
    edge_attr = torch.tensor([[e['rtt']] for e in G.edges.values()], dtype=torch.float)
    pyg_graph.edge_attr = edge_attr

    return pyg_graph

# Usage
# warts_file = './topo-v6.l8.20240101.1704072990.hlz2-nz.warts'
# pyg_data = warts_to_graph(warts_file, max_records=1000, num_walks=10, walk_length=80, embedding_type='deepwalk', embedding_dim=32)
