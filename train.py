import torch
from torch.utils.data import TensorDataset, DataLoader
from src.model.loss import simplified_loss_function, contrained_loss_function, apply_constraints
from src.utils.metrics import mape_calculation, mse_calculation
from src.dataset.loader import load_data
from src.model.deepnt import DeepNT
from src.utils.get_paths import edge_index_to_adj, edge_masked
from tqdm import tqdm
import argparse
import numpy as np


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def prepare_batches(data, batch_size, mask, shuffle=True):
    node_pairs = data.node_pair[:, mask].t()
    performance_values = data.performance_values[mask]
    dataset = TensorDataset(node_pairs, performance_values)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_wo_constraints(model, data, adj, optimizer, device, num_epochs, batch_size, patience):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    dataloader = prepare_batches(data, batch_size, data.train_mask, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mape = 0
        total_mse = 0
        total_samples = 0

        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = simplified_loss_function(batch_outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_mse += mse_calculation(batch_outputs, batch_labels).item() * len(batch_labels)
            total_samples += len(batch_labels)

        avg_loss = total_loss / total_samples
        avg_mape = total_mape / total_samples
        avg_mse = total_mse / total_samples
        val_loss, val_mape, val_mse = validate_wo_constraints(model, data, adj, device, batch_size)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train MAPE: {avg_mape:.4f}, Train MSE: {avg_mse:.4f}, Val Loss: {val_loss:.4f}, Val MAPE: {val_mape:.4f}, Val MSE: {val_mse:.4f}')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model

def validate_wo_constraints(model, data, adj, device, batch_size):
    model.eval()
    dataloader = prepare_batches(data, batch_size, data.val_mask, shuffle=False)
    
    total_loss = 0
    total_mape = 0
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Running validation"):
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)

            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = simplified_loss_function(batch_outputs, batch_labels)
            
            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_mse += mse_calculation(batch_outputs, batch_labels).item() * len(batch_labels)
            total_samples += len(batch_labels)

    return total_loss / total_samples, total_mape / total_samples, total_mse / total_samples

def test_wo_constraints(model, data, adj, device, batch_size):
    model.eval()
    dataloader = prepare_batches(data, batch_size, data.test_mask, shuffle=False)

    total_loss = 0
    total_mape = 0
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Running test"):
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = simplified_loss_function(batch_outputs, batch_labels)
            
            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_mse += mse_calculation(batch_outputs, batch_labels).item() * len(batch_labels)
            total_samples += len(batch_labels)

    return total_loss / total_samples, total_mape / total_samples, total_mse / total_samples


def train_with_constraints(model, data, adj, optimizer, Q, K, lambda1, lambda2, lambda3, device, num_epochs, d, V, batch_size, patience):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    dataloader = prepare_batches(data, batch_size, data.train_mask, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        adj = apply_constraints(adj, d, V).to(device)
        total_loss = 0
        total_mape = 0
        total_mse = 0
        total_samples = 0

        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = contrained_loss_function(batch_outputs, batch_labels, adj, Q, K, lambda1, lambda2, lambda3)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_mse += mse_calculation(batch_outputs, batch_labels).item() * len(batch_labels)
            total_samples += len(batch_labels)

        avg_loss = total_loss / total_samples
        avg_mape = total_mape / total_samples
        avg_mse = total_mse / total_samples
        val_loss, val_mape, val_mse = validate_with_constraints(model, data, adj, Q, K, lambda1, lambda2, lambda3, device, batch_size)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train MAPE: {avg_mape:.4f}, Train MSE: {avg_mse:.4f}, Val Loss: {val_loss:.4f}, Val MAPE: {val_mape:.4f}, Val MSE: {val_mse:.4f}')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, adj

def validate_with_constraints(model, data, adj, Q, K, lambda1, lambda2, lambda3, device, batch_size):
    model.eval()
    dataloader = prepare_batches(data, batch_size, data.val_mask, shuffle=False)

    total_loss = 0
    total_mape = 0
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Running validation"):
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = contrained_loss_function(batch_outputs, batch_labels, adj, Q, K, lambda1, lambda2, lambda3)

            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_mse += mse_calculation(batch_outputs, batch_labels).item() * len(batch_labels)
            total_samples += len(batch_labels)

    return total_loss / total_samples, total_mape / total_samples, total_mse / total_samples

def test_with_constraints(model, data, adj, Q, K, lambda1, lambda2, lambda3, device, batch_size):
    model.eval()
    dataloader = prepare_batches(data, batch_size, data.test_mask, shuffle=False)
    
    total_loss = 0
    total_mape = 0
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Running test"):
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = contrained_loss_function(batch_outputs, batch_labels, adj, Q, K, lambda1, lambda2, lambda3)
            
            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_mse += mse_calculation(batch_outputs, batch_labels).item() * len(batch_labels)
            total_samples += len(batch_labels)

    return total_loss / total_samples, total_mape / total_samples, total_mse / total_samples


def main():
    parser = argparse.ArgumentParser(description="Training DeepNT model ...")
    parser.add_argument("--use_constraints", action="store_true", help="Use constraints in training")
    parser.add_argument("--input_dim", type=int, default=None, help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--output_dim", type=int, default=128, help="Output dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--num_paths", type=int, default=1, help="Number of paths")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--lambda1", type=float, default=1e-3, help="Lambda1 for constraints")
    parser.add_argument("--lambda2", type=float, default=1e-3, help="Lambda2 for constraints")
    parser.add_argument("--lambda3", type=float, default=1e-1, help="Lambda3 for constraints")
    parser.add_argument("--K", type=int, default=3, help="K value for constraints")
    parser.add_argument("--d", type=int, default=100, help="Sparsity threshold")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training and validation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--metric", type=str, default="delay", help="Metric to use")
    parser.add_argument("--patience", type=int, default=7, help="Patience for early stopping")
    parser.add_argument("--topology_error_rate", type=float, default=0.005, help="Topology error")
    parser.add_argument("--seed", type=float, default=42, help="Random seed")
 
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    raw_data = torch.load(args.data_path)
    data = load_data(raw_data, args.metric)['data'].to(device)
    
    if args.input_dim is None:
        args.input_dim = data.x.size(1)
 
    # Initialize the model, criterion, and optimizer
    model = DeepNT(args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.num_paths).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    masked_edge_index = edge_masked(data.edge_index, data.x.shape[0], args.topology_error_rate, args.seed).to(device)   
    adj = edge_index_to_adj(masked_edge_index, data.x.shape[0]).to(device) 

    # Define the constraints
    V = data.x.shape[0]
    Q = torch.zeros_like(adj).to(device)
    K = torch.tensor(args.K, device=device)

    if args.use_constraints:
        model, adj = train_with_constraints(model, data, adj, optimizer, Q, K, args.lambda1, args.lambda2, args.lambda3, device, args.num_epochs, args.d, V, args.batch_size, args.patience)
        test_loss, test_mape, test_mse = test_with_constraints(model, data, adj, Q, args.K, args.lambda1, args.lambda2, args.lambda3, device, args.batch_size)
    else:
        model = train_wo_constraints(model, data, adj, optimizer, device, args.num_epochs, args.batch_size, args.patience)
        test_loss, test_mape, test_mse = test_wo_constraints(model, data, adj, device, args.batch_size)

    print(f'Test Loss: {test_loss:.4f}, Test MAPE: {test_mape:.4f}, Test MSE: {test_mse:.4f}')

if __name__ == "__main__":
    main()
