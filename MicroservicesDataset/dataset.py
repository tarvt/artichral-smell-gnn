import os
import torch
import json
from torch_geometric.data import InMemoryDataset, Data
import numpy as np

torch.manual_seed(42)  # For reproducibility
np.random.seed(42)
dataset_name = "graph_dataset.jsonl"
class MicroservicesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MicroservicesDataset, self).__init__(root, transform, pre_transform)
        self.process_data()

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_file_names(self):
        return [dataset_name]

    def process_data(self):
        processed_path = self.processed_paths[0]
        raw_path = os.path.join(self.root, self.raw_file_names[0])

        # Check if we need to reprocess the data
        if not os.path.exists(processed_path) or os.path.getmtime(raw_path) > os.path.getmtime(processed_path):
            print("Reprocessing dataset...")
            self.process()
        else:
            print("Loading processed dataset...")
            self.data, self.slices = torch.load(processed_path)

    def normalize_features(self, features):
        # Normalize using Min-Max scaling
        min_vals = torch.min(features, dim=0)[0]
        max_vals = torch.max(features, dim=0)[0]
        return (features - min_vals) / (max_vals - min_vals + 1e-5)

    def process(self):
        data_list = []
        path = os.path.join(self.root, self.raw_file_names[0])
        with open(path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    graph = json.loads(line.strip())
                    x = torch.tensor(graph['Nodes'], dtype=torch.float)
                    x = self.normalize_features(x)  # Normalize features here
                    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
                    edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float)
                    # Ensure they match in number 
                    if edge_index.shape[1] != edge_attr.shape[0]:
                        print(f"Mismatch between edge_index ({edge_index.shape[1]}) and edge_attr ({edge_attr.shape[0]})")
                    y = torch.tensor(graph['labels'], dtype=torch.float)
                    x = torch.cat((x, y), dim=1)
                    x = self.normalize_features(x)  # Normalize features
                    


                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"JSON decoding failed at line {line_number}: {e}")
                except Exception as e:
                    print(f"An error occurred at line {line_number}: {str(e)}")

        if data_list:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            self.data, self.slices = data, slices
        else:
            print("No valid data was loaded.")


