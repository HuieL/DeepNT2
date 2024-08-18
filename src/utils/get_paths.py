from collections import deque
import random
import torch


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
    
    masked_edge_index = edge_index.clone()
    edges_to_replace = random.sample(range(num_edges), num_errors)
    
    for edge_idx in edges_to_replace:
        while True:
            new_source = random.randint(0, num_nodes - 1)
            new_target = random.randint(0, num_nodes - 1)
            
            if new_source != new_target and not edge_exists(masked_edge_index, new_source, new_target):
                masked_edge_index[:, edge_idx] = torch.tensor([new_source, new_target])
                break
    
    return masked_edge_index

def edge_exists(edge_index, source, target):
    return ((edge_index[0] == source) & (edge_index[1] == target)).any() or \
           ((edge_index[0] == target) & (edge_index[1] == source)).any()

# Randomly sample paths
def random_sample_paths(adj, start, goal, num_paths=2, max_attempts=200, max_depth=10):
    all_paths = set()

    def bfs_paths(adj, start, goal, max_attempts, max_depth):
        queue = deque([(start, [start])])
        attempts = 0

        while queue and attempts < max_attempts:
            node, path = queue.popleft()

            if node == goal:
                all_paths.add(tuple(path))
                attempts += 1
                if len(all_paths) >= num_paths:
                    break
                continue

            if len(path) >= max_depth:
                continue

            neighbors = torch.nonzero(adj[node]).squeeze().tolist()
            if isinstance(neighbors, int):
                neighbors = [neighbors]
            for neighbor in neighbors:
                if neighbor not in path:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        return list(all_paths)

    generated_paths = bfs_paths(adj, start, goal, max_attempts, max_depth)
    if len(generated_paths) < num_paths:
        num_paths = len(generated_paths)
    return random.sample(generated_paths, num_paths)
