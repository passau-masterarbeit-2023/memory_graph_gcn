import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LessSimplifiedGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(LessSimplifiedGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 12)
        self.conv2 = GCNConv(12, 24)
        self.fc1 = torch.nn.Linear(24, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second Graph Convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Fully Connected Layer 1
        x = self.fc1(x)
        x = F.relu(x)

        # Output Layer
        x = self.fc2(x)
        
        return x  # Removed sigmoid activation to use with BCEWithLogitsLoss
