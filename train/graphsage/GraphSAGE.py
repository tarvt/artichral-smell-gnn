import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        # Initialize GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, 16, edge_dim=3)
        self.conv2 = SAGEConv(16, out_channels, edge_dim=3)

    def forward(self, x, edge_index , edge_attr):
        # First GraphSAGE layer
        x = F.relu(self.conv1(x, edge_index , edge_attr))
        x = F.dropout(x, training=self.training)
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index , edge_attr)
        return x
