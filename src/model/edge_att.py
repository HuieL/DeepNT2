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

    def forward(self, hu, hz):
        if hu.dim() == 1:
            hu = hu.unsqueeze(0)
        if hz.dim() == 1:
            hz = hz.unsqueeze(0)

        concat = torch.cat((hu.expand(hz.size(0), -1), hz), dim=1)
        e_ik = self.leaky_relu(torch.matmul(concat, self.attention).squeeze(1))
        alpha_ik = F.softmax(e_ik, dim=0)
        weighted_hz = torch.sum(alpha_ik.unsqueeze(1) * hz, dim=0)
        hu_updated = hu.squeeze(0) + self.elu(weighted_hz)

        return hu_updated
