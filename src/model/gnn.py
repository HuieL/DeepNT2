import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.sparse.linalg import eigsh
from torch_geometric.nn import GCNConv
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops as add_self_loops_fn, scatter


# GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout = 0.0):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNLayer(input_dim, hidden_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(GCNLayer(hidden_dim, output_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x

class GCNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__(aggr='add')
        self.lin = Linear(input_dim, output_dim, bias=False, weight_initializer='glorot')
        self.bias = Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)

        row_sum = adj.sum(dim=1, keepdim=True)
        adj_norm = adj / row_sum

        x = self.lin(x)
        x = torch.matmul(adj_norm, x)
        if self.bias is not None:
            x = x + self.bias
        return x


# GAT model
class GAT(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
    super(GAT, self).__init__()
    self.num_layers = num_layers
    self.gat_layers = nn.ModuleList()

    self.gat_layers.append(GATLayer(input_dim, hidden_dim, num_heads))
    for _ in range(num_layers - 2):
      self.gat_layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads))
      self.gat_layers.append(GATLayer(hidden_dim * num_heads, output_dim, 1))

  def forward(self, x, adj):
      for i in range(self.num_layers):
          x = self.gat_layers[i](x, adj)
          if i < self.num_layers - 1:
              x = F.elu(x)
      return x

class GATLayer(nn.Module):
  def __init__(self, input_dim, output_dim, num_heads, dropout=0.6, alpha=0.2):
    super(GATLayer, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.alpha = alpha

    # Define the weight matrix for each head
    self.W = nn.ParameterList([nn.Parameter(torch.Tensor(input_dim, output_dim)) for _ in range(num_heads)])
    for w in self.W:
        nn.init.xavier_uniform_(w, gain=1.414)

    # Define the attention mechanism
    self.a = nn.ParameterList([nn.Parameter(torch.Tensor(2 * output_dim, 1)) for _ in range(num_heads)])
    for a in self.a:
        nn.init.xavier_uniform_(a, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, x, adj):
      N = x.size(0)
      h_prime = []

      for i in range(self.num_heads):
          h = torch.matmul(x, self.W[i])

          # Compute attention coefficients
          a_input = self._prepare_attentional_mechanism_input(h)
          e = self.leakyrelu(torch.matmul(a_input, self.a[i]).squeeze(2))

          # Apply masking to attention coefficients based on adjacency matrix
          zero_vec = -9e15 * torch.ones_like(e)
          attention = torch.where(adj > 0, e, zero_vec)
          attention = F.softmax(attention, dim=1)
          attention = F.dropout(attention, self.dropout, training=self.training)

          # Apply attention weights to node features
          h_prime.append(torch.matmul(attention, h))

      # Concatenate the output of each head
      h_prime = torch.cat(h_prime, dim=1)

      return h_prime

  def _prepare_attentional_mechanism_input(self, h):
      N = h.size(0)
      h_repeat = h.repeat_interleave(N, dim=0)
      h_tiled = h.repeat(N, 1)
      a_input = torch.cat([h_repeat, h_tiled], dim=1).view(N, N, -1)
      return a_input
