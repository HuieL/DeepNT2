import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh


# Loss
def proximal_operator(A):
    L_complex, V_complex = torch.linalg.eig(A)
    L = L_complex.real
    V = V_complex.real
    L = torch.clamp(L, min=0)
    return (V @ torch.diag(L) @ V.T).real

def apply_constraints(A, d, V):
    A = torch.clamp(A, min=0)
    row_sum = torch.sum(A, dim=1).unsqueeze(1)
    A = A - torch.diag(row_sum.squeeze() / V) + torch.ones_like(A)
    A = proximal_operator(A)
    mask = torch.topk(A.view(-1), d, largest=False).indices
    A.view(-1)[mask] = 0
    return A

def diameter_constraint(A, Q, K):
    A_np = A.detach().cpu().numpy()
    L_complex, V_complex = eigsh(A_np, k=min(A_np.shape[0] - 1, 10))
    L = torch.from_numpy(L_complex).float().to(A.device)
    V = torch.from_numpy(V_complex).float().to(A.device)
    A_k = torch.zeros_like(A)

    for k in range(1, K+1):
        L_k = torch.pow(L, k)
        A_k += V @ torch.diag(L_k) @ V.T

    return torch.norm(torch.clamp(Q - A_k, min=0), p='fro')

# Simplified loss function with only MSE loss
def simplified_loss_function(output, target):
    mse_loss = F.mse_loss(output, target)
    return mse_loss

def contrained_loss_function(output, target, A, Q, K, lambda1, lambda2, lambda3):
    mse_loss = F.mse_loss(output, target)
    sparsity_loss = lambda1 * torch.norm(A, p=0)
    diameter_loss = lambda2 * diameter_constraint(A, Q, K)
    metric_specific_loss = lambda3 * sum(torch.clamp(output - min(yuz + yzv for yuz, yzv in zip(output, output)), min=0))
    return mse_loss + sparsity_loss + diameter_loss + metric_specific_loss
