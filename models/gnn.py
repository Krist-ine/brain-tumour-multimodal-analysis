import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_classes=4, num_layers=2):
        super(GNNClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    from torch_geometric.data import Data
    # Dummy graph: 10 nodes, 3 features, fully connected
    x = torch.randn(10, 3)
    edge_index = torch.combinations(torch.arange(10), r=2).t().contiguous()
    batch = torch.zeros(10, dtype=torch.long)
    
    model = GNNClassifier(in_features=3, num_classes=4)
    out = model(x, edge_index, batch)
    print("Output shape:", out.shape)  # [num_graphs, num_classes]
