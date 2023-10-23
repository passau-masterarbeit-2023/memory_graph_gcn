import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d

class AdvancedGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(AdvancedGCN, self).__init__()
        
        # Graph Convolution Layers
        self.conv1 = GCNConv(num_features, 32)
        self.bn1 = BatchNorm1d(32)
        
        self.conv2 = GCNConv(32, 64)
        self.bn2 = BatchNorm1d(64)

        self.conv3 = GCNConv(64, 128)
        self.bn3 = BatchNorm1d(128)
        
        # Fully Connected Layers
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, num_classes)
        
        # Dropout Layer
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Graph Convolution
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        # Second Graph Convolution
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)

        # Third Graph Convolution
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)

        # Fully Connected Layer 1
        x = self.fc1(x3)
        x = F.relu(x)
        x = self.dropout(x)

        # Fully Connected Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fully Connected Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        
        # Output Layer
        x = self.fc4(x)
        
        return x
