import torch
import torch.nn as nn
from src.model.gnn import GCN, GAT
from src.model.edge_att import AttentionLayer
from src.utils.get_paths import random_sample_paths


class DeepNT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_paths=1):
        super(DeepNT, self).__init__()
        self.num_paths = num_paths
        self.gnn = GCN(input_dim, hidden_dim, output_dim, num_layers)
        self.attention_layer = AttentionLayer(output_dim)
        self.fc = nn.Linear(2 * output_dim, 1)

    def forward(self, x, u, v, adj):
        node_embeddings = self.gnn(x, adj)
        batch_size = u.size(0)
        
        hu_updated = node_embeddings[u]
        hv_updated = node_embeddings[v]

        for i in range(batch_size):
            paths = random_sample_paths(adj, u[i].item(), v[i].item(), self.num_paths, max_attempts=200, max_depth=10)
            
            for path in paths:
                path_embeddings = torch.stack([node_embeddings[node] for node in path])
                hu_updated[i] = self.attention_layer(hu_updated[i], path_embeddings)
                hv_updated[i] = self.attention_layer(hv_updated[i], path_embeddings)

        concat = torch.cat((hu_updated, hv_updated), dim=-1)
        output = self.fc(concat)
        return output.squeeze(-1) 
