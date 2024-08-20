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
        
        # Batch sample paths
        paths, path_lengths = batch_random_sample_paths(adj, u, v, self.num_paths)  # Shape: [batch_size, num_paths, path_length]
        path_embeddings = node_embeddings[paths]  # Shape: [batch_size, num_paths, path_length, embedding_dim]
        
        hu_updated = self.attention_layer(node_embeddings[u], path_embeddings, path_lengths)
        hv_updated = self.attention_layer(node_embeddings[v], path_embeddings, path_lengths)
        
        concat = torch.cat((hu_updated, hv_updated), dim=-1)
        output = self.fc(concat)
        return output.squeeze(-1)
