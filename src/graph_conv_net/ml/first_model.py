import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from graph_conv_net.utils.cpu_gpu_torch import DEVICE

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16).to(DEVICE)
        self.conv2 = GCNConv(16, 32).to(DEVICE)
        self.fc1 = torch.nn.Linear(32, 64).to(DEVICE)
        self.fc2 = torch.nn.Linear(64, 32).to(DEVICE)
        self.fc3 = torch.nn.Linear(32, num_classes).to(DEVICE)

    def forward(self, data):
        x, edge_index = data.x.to(DEVICE), data.edge_index.to(DEVICE)

        # First Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second Graph Convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Fully Connected Layer 1
        x = self.fc1(x)
        x = F.relu(x)

        # Fully Connected Layer 2
        x = self.fc2(x)
        x = F.relu(x)

        # Output Layer
        x = self.fc3(x)
        
        return x  # Removed sigmoid activation to use with BCEWithLogitsLoss
