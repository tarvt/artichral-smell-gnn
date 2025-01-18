import torch
from torch_geometric.data import Data

# Node features (assuming hypothetical feature vectors for each microservice)
x = torch.tensor([
    [1, 0, 0, 0, 1],  # User Service
    [0, 1, 0, 2, 0],  # Order Service
    [0, 0, 1, 0, 1],  # Inventory Service
    [0, 0, 0, 0, 2]   # Database Service
], dtype=torch.float)

# Edge index, adjusted for new interactions
edge_index = torch.tensor([
    [1, 1, 2, 0],  # Source nodes
    [0, 2, 3, 1]   # Target nodes
], dtype=torch.long)

# Edge attributes for each edge defined in edge_index
edge_attr = torch.tensor([
    [10, 1000, 2000],  # Latency, Bytes Transmitted, Bytes Received for edge 1-0
    [15, 1500, 2500],  # Latency, Bytes Transmitted, Bytes Received for edge 1-2
    [5, 500, 1000],    # Latency, Bytes Transmitted, Bytes Received for edge 2-3
    [20, 2000, 3000]   # Latency, Bytes Transmitted, Bytes Received for edge 0-1
], dtype=torch.float)

# Labels for nodes (hypothetical)
y = torch.tensor([0, 1, 2, 0], dtype=torch.long)

# Creating the data object including labels and edge attributes
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

print(data)