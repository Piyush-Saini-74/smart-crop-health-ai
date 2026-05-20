import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
except ImportError:
    print("Warning: torch_geometric is not installed. GNN modules will not work.")

class GNNFusion(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='gcn'):
        super(GNNFusion, self).__init__()
        self.model_type = model_type
        
        # Environmental features: N, P, K, Temp, Humidity, pH, Rainfall (7 nodes) + 1 Image node
        # In a real setup, nodes could be: 1 image node + 7 env nodes.
        # Here we just set up a generic GCN/GAT that takes node features and an adjacency matrix.
        
        if model_type == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'gat':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        else:
            raise ValueError("Unsupported model_type. Choose 'gcn' or 'gat'.")
            
        # Final classification head
        self.fc = nn.Linear(hidden_channels, out_channels)
        
        # Environmental stress score (e.g., 0-1)
        self.stress_fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges]
        batch: Batch vector [num_nodes] mapping nodes to graphs
        """
        if self.model_type == 'gcn':
            x = self.conv1(x, edge_index, edge_weight)
        else:
            # GAT usually doesn't take edge_weight in its standard simple form without modifications
            x = self.conv1(x, edge_index)
            
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        if self.model_type == 'gcn':
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv2(x, edge_index)
            
        x = F.relu(x)
        
        # Pool graph representations
        if batch is None:
            # Default to single graph (all nodes belong to batch 0)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        pooled_embedding = global_mean_pool(x, batch) # [batch_size, hidden_channels]

        disease_logits = self.fc(pooled_embedding)
        stress_score = torch.sigmoid(self.stress_fc(pooled_embedding))
        
        return disease_logits, stress_score
