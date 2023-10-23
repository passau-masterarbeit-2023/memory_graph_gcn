import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class VerySimplifiedGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(VerySimplifiedGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.fc1 = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Single Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Output Layer
        x = self.fc1(x)
        
        return x  # Removed sigmoid activation to use with BCEWithLogitsLoss
