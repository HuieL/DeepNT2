import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data


class NTDataset(Dataset):
    def __init__(self, data, metric, val_ratio=0.5, seed=42):
        self.data = data
        self.metric = metric
        self.x = data.x
        self.num_nodes = data.num_nodes
        
        # Get the specific node pairs and path attributes for the metric
        self.node_pair_monitor = data.node_pair_monitor[metric]
        self.node_pair_unknown = data.node_pair_unknown[metric]
        self.path_attr_monitor = data.metrics_monitor[metric]
        self.path_attr_unknown = data.metrics_unknown[metric]
        
        # Combine monitored and unknown edges
        self.edge_index = data.edge_index
        self.node_pair = torch.cat([self.node_pair_monitor, self.node_pair_unknown], dim=1)
        self.performance_values = torch.cat([self.path_attr_monitor, self.path_attr_unknown])
        
        # Generate splits
        self.train_mask, self.val_mask, self.test_mask = self.generate_splits(val_ratio, seed)
    
    def generate_splits(self, val_ratio, seed):
        num_monitor_edges = self.node_pair_monitor.shape[1]
        num_unknown_edges = self.node_pair_unknown.shape[1]
        total_edges = num_monitor_edges + num_unknown_edges

        # All monitored edges are for training
        train_mask = torch.zeros(total_edges, dtype=torch.bool)
        train_mask[:num_monitor_edges] = True

        # Split unknown edges into validation and test
        unknown_indices = np.arange(num_unknown_edges)
        val_indices, test_indices = train_test_split(
            unknown_indices, test_size=1-val_ratio, random_state=seed
        )
        
        val_mask = torch.zeros(total_edges, dtype=torch.bool)
        val_mask[num_monitor_edges + val_indices] = True
        
        test_mask = torch.zeros(total_edges, dtype=torch.bool)
        test_mask[num_monitor_edges + test_indices] = True
        
        return train_mask, val_mask, test_mask
    
    def __len__(self):
        return 1  # We only have one graph
    
    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset contains only one graph")
        
        graph = Data(
            x=self.x,
            edge_index=self.edge_index,
            node_pair=self.node_pair,
            performance_values=self.performance_values,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask,
            num_nodes=self.num_nodes,
            node_pair_monitor=self.node_pair_monitor,
            node_pair_unknown=self.node_pair_unknown
        )
        
        return {"data": graph}

def load_data(data, metric, val_ratio=0.5, seed=42):
    dataset = NTDataset(data, metric, val_ratio, seed)
    return dataset[0]  # Return the single graph data
