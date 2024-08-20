import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Parameter(torch.empty(size=(2 * input_dim, 1)))
        nn.init.xavier_uniform_(self.attention.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.elu = nn.ELU()

    def forward(self, hu, hz, path_lengths):
        batch_size, num_paths, max_path_length, input_dim = hz.shape
        hu_expanded = hu.unsqueeze(1).unsqueeze(2).expand(-1, num_paths, max_path_length, -1)
        concat = torch.cat((hu_expanded, hz), dim=-1)

        e_ik = self.leaky_relu(torch.matmul(concat, self.attention).squeeze(-1))
        mask = torch.arange(max_path_length, device=path_lengths.device).expand(batch_size, num_paths, -1) < path_lengths.unsqueeze(-1)
        e_ik = e_ik.masked_fill(~mask, float('-inf'))
        alpha_ik = F.softmax(e_ik, dim=-1)

        weighted_hz = torch.sum(alpha_ik.unsqueeze(-1) * hz, dim=2)
        aggregated_hz = torch.sum(weighted_hz, dim=1)

        hu_updated = hu + self.elu(aggregated_hz)

        return hu_updated
