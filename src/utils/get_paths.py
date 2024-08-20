from collections import deque
import random
import torch
import networkx as nx


# Build adjacent matrix with edge_index
def edge_index_to_adj(edge_index, num_nodes):
      adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
      adj[edge_index[0], edge_index[1]] = 1
      adj[edge_index[1], edge_index[0]] = 1
      adj += torch.eye(num_nodes)
      return adj

def edge_masked(edge_index, num_nodes, topology_error_rate, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    num_edges = edge_index.shape[1]
    num_errors = int(num_edges * topology_error_rate)
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    edges_to_remove = random.sample(list(G.edges()), num_errors)
    G.remove_edges_from(edges_to_remove)
    
    for _ in range(num_errors):
        components = list(nx.connected_components(G))
        if len(components) == 1:
            while True:
                new_source = random.randint(0, num_nodes - 1)
                new_target = random.randint(0, num_nodes - 1)
                if new_source != new_target and not G.has_edge(new_source, new_target):
                    G.add_edge(new_source, new_target)
                    break
        else:
            comp1, comp2 = random.sample(components, 2)
            new_source = random.choice(list(comp1))
            new_target = random.choice(list(comp2))
            G.add_edge(new_source, new_target)
    
    new_edge_list = list(G.edges())
    new_edge_index = torch.tensor(new_edge_list, dtype=torch.long).t()
    
    return new_edge_index

# Randomly sample paths
def random_sample_paths(G, start, goal, num_paths, max_depth):
    all_paths = []
    for path in nx.shortest_simple_paths(G, start, goal):
        if len(path) > max_depth:
            break
        all_paths.append(path)
        if len(all_paths) >= num_paths:
            break

    if len(all_paths) < num_paths:
        return all_paths
    return random.sample(all_paths, num_paths)

def batch_random_sample_paths(adj, starts, goals, num_paths, max_depth=10):
    batch_size = starts.shape[0]
    G = nx.from_numpy_array(adj.cpu().numpy())
    
    all_paths = []
    path_lengths = []
    
    for i in range(batch_size):
        start, goal = starts[i].item(), goals[i].item()
        paths = random_sample_paths(G, start, goal, num_paths, max_depth)

        padded_paths = []
        lengths = []

        while len(paths) < num_paths:
            paths.append([start, goal])  

        for path in paths:
            if len(path) < max_depth:
                padded_path = path + [path[-1]] * (max_depth - len(path))
                length = len(path)
            else:
                padded_path = path[:max_depth]
                length = max_depth
            padded_paths.append(padded_path)
            lengths.append(length)
        
        while len(padded_paths) < num_paths:
            padded_paths.append(padded_paths[-1] if padded_paths else [start] * max_depth)
        
        all_paths.append(padded_paths)
        path_lengths.append(lengths)
    
    return torch.LongTensor(all_paths).to(starts.device), torch.LongTensor(path_lengths).to(starts.device)
