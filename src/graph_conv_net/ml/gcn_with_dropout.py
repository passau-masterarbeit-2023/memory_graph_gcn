import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d

class ImprovedGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(ImprovedGNN, self).__init__()
        
        self.conv1 = GCNConv(num_features, 16)
        self.bn1 = BatchNorm1d(16)
        
        self.conv2 = GCNConv(16, 32)
        self.bn2 = BatchNorm1d(32)
        
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, num_classes)
        
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x