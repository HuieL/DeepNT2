from collections import deque
import random
import torch


# Build adjacent matrix with edge_index
def edge_index_to_adj(edge_index, num_nodes):
      adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
      adj[edge_index[0], edge_index[1]] = 1
      adj[edge_index[1], edge_index[0]] = 1
      adj += torch.eye(num_nodes)
      # Normalize adjacency matrix
      deg = adj.sum(dim=1)
      deg_inv_sqrt = deg.pow(-0.5)
      deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
      adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
      return adj

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
