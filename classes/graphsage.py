import torch
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels=14, out_channels=16)
        self.conv2 = SAGEConv(16, out_channels)

    def forward(self, x, edge_index , edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x