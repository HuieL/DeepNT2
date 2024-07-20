import torch
from torch.utils.data import TensorDataset, DataLoader
from src.model.loss import simplified_loss_function, contrained_loss_function, apply_constraints
from src.utils.metrics import mape_calculation
from src.dataset.loader import load_data
from src.model.deepnt import DeepNT
from src.utils.get_paths import edge_index_to_adj
from tqdm import tqdm
import argparse


def prepare_batches(data, batch_size, mask, shuffle=True):
    node_pairs = data.node_pair[:, mask].t()
    performance_values = data.performance_values[mask]
    dataset = TensorDataset(node_pairs, performance_values)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_wo_constraints(model, data, adj, optimizer, device, num_epochs, batch_size):
    model.train()
    dataloader = prepare_batches(data, batch_size, data.train_mask, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        total_mape = 0
        total_samples = 0

        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = simplified_loss_function(batch_outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_samples += len(batch_labels)

        avg_loss = total_loss / total_samples
        avg_mape = total_mape / total_samples
        val_loss, val_mape = validate_wo_constraints(model, data, adj, device, batch_size)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train MAPE: {avg_mape:.4f}, Val Loss: {val_loss:.4f}, Val MAPE: {val_mape:.4f}')

def validate_wo_constraints(model, data, adj, device, batch_size):
    model.eval()
    dataloader = prepare_batches(data, batch_size, data.val_mask, shuffle=False)
    
    total_loss = 0
    total_mape = 0
    total_samples = 0

    with torch.no_grad():
        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Running validation"):
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = simplified_loss_function(batch_outputs, batch_labels)
            
            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_samples += len(batch_labels)

    return total_loss / total_samples, total_mape / total_samples

def test_wo_constraints(model, data, adj, device, batch_size):
    model.eval()
    dataloader = prepare_batches(data, batch_size, data.test_mask, shuffle=False)
    
    total_loss = 0
    total_mape = 0
    total_samples = 0

    with torch.no_grad():
        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Running test"):
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = simplified_loss_function(batch_outputs, batch_labels)
            
            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_samples += len(batch_labels)

    return total_loss / total_samples, total_mape / total_samples


def train_with_constraints(model, data, adj, optimizer, Q, K, lambda1, lambda2, lambda3, device, num_epochs, d, V, batch_size):
    model.train()
    dataloader = prepare_batches(data, batch_size, data.train_mask, shuffle=True)

    for epoch in range(num_epochs):
        adj = apply_constraints(adj, d, V)
        total_loss = 0
        total_samples = 0

        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = contrained_loss_function(batch_outputs, batch_labels, adj, Q, K, lambda1, lambda2, lambda3)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            total_samples += len(batch_labels)

        avg_loss = total_loss / total_samples
        val_loss, val_mape = validate_with_constraints(model, data, adj, Q, K, lambda1, lambda2, lambda3, device, batch_size)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAPE: {val_mape:.4f}')
    
    return adj

def validate_with_constraints(model, data, adj, Q, K, lambda1, lambda2, lambda3, device, batch_size):
    model.eval()
    dataloader = prepare_batches(data, batch_size, data.val_mask, shuffle=False)
    
    total_loss = 0
    total_mape = 0
    total_samples = 0

    with torch.no_grad():
        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Running validation"):
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = contrained_loss_function(batch_outputs, batch_labels, adj, Q, K, lambda1, lambda2, lambda3)
            
            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_samples += len(batch_labels)

    return total_loss / total_samples, total_mape / total_samples

def test_with_constraints(model, data, adj, Q, K, lambda1, lambda2, lambda3, device, batch_size):
    model.eval()
    dataloader = prepare_batches(data, batch_size, data.test_mask, shuffle=False)
    
    total_loss = 0
    total_mape = 0
    total_samples = 0

    with torch.no_grad():
        for batch_node_pairs, batch_labels in tqdm(dataloader, desc=f"Running test"):
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_outputs = model(data.x, batch_node_pairs[:, 0], batch_node_pairs[:, 1], adj)
            
            loss = contrained_loss_function(batch_outputs, batch_labels, adj, Q, K, lambda1, lambda2, lambda3)
            
            total_loss += loss.item() * len(batch_labels)
            total_mape += mape_calculation(batch_outputs, batch_labels).sum().item()
            total_samples += len(batch_labels)

    return total_loss / total_samples, total_mape / total_samples

def main():
    parser = argparse.ArgumentParser(description="Training DeepNT model ...")
    parser.add_argument("--use_constraints", action="store_true", help="Use constraints in training")
    parser.add_argument("--input_dim", type=int, default=None, help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--output_dim", type=int, default=64, help="Output dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_paths", type=int, default=2, help="Number of paths")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--lambda1", type=float, default=1e-3, help="Lambda1 for constraints")
    parser.add_argument("--lambda2", type=float, default=1e-3, help="Lambda2 for constraints")
    parser.add_argument("--lambda3", type=float, default=1e-1, help="Lambda3 for constraints")
    parser.add_argument("--K", type=int, default=3, help="K value for constraints")
    parser.add_argument("--d", type=int, default=100, help="Sparsity threshold")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and validation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--metric", type=str, default="delay", help="Metric to use")

    args = parser.parse_args()

    # Load data
    raw_data = torch.load(args.data_path)
    data = load_data(raw_data, args.metric)['data']
    
    if args.input_dim is None:
        args.input_dim = data.x.size(1)

    # Initialize the model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepNT(args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.num_paths).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    data = data.to(device)
    adj = edge_index_to_adj(data.edge_index, data.x.shape[0]).to(device)

    # Define the constraints
    V = data.x.shape[0]
    Q = torch.zeros_like(adj) 

    if args.use_constraints:
        adj = train_with_constraints(model, data, adj, optimizer, Q, args.K, args.lambda1, args.lambda2, args.lambda3, device, args.num_epochs, args.d, V, args.batch_size)
        test_loss, test_mape = test_with_constraints(model, data, adj, Q, args.K, args.lambda1, args.lambda2, args.lambda3, device, args.batch_size)
    else:
        train_wo_constraints(model, data, adj, optimizer, device, args.num_epochs, args.batch_size)
        test_loss, test_mape = test_wo_constraints(model, data, adj, device, args.batch_size)

    print(f'Test Loss: {test_loss:.4f}, Test MAPE: {test_mape:.4f}')

if __name__ == "__main__":
    main()
