import torch
from torch_geometric.data import Data, InMemoryDataset
import json
import logging
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_curve
import numpy as np
import os 
import uuid

dataset_name ='handgenerated3.jsonl' #'graph_dataset.jsonl'
torch.manual_seed(42)  # For reproducibility
np.random.seed(42)

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

    def process(self):
        print("here1")
        data_list = []
        path = os.path.join(self.root, self.raw_file_names[0])
        print("here2")
        with open(path, 'r') as f:
            print("here3")
            for line_number, line in enumerate(f, 1):
                try:
                    graph = json.loads(line.strip().replace("'", '"'))
                    x = torch.tensor(graph['Nodes'], dtype=torch.float)
                    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
                    edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float)
                    y = torch.tensor(graph['labels'], dtype=torch.float)

                    # Ensure edge_index is correctly shaped
                    if edge_index.size(0) != 2:
                        if edge_index.size(1) == 2:
                            edge_index = edge_index.t()
                        else:
                            logging.error(f"edge_index at line {line_number} is malformed: {edge_index.size()}")
                            continue

                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding failed at line {line_number}: {e}")
                except Exception as e:
                    logging.error(f"An error occurred at line {line_number}: {str(e)}")

        if data_list:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            self.data, self.slices = data, slices
        else:
            logging.error("No valid data was loaded.")

# Setup basic configuration for logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
dataset = MicroservicesDataset(root='E:\Masters_degree\project code\MicroservicesDataset')
print("dataset done ----------------------------------------------")

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16, normalize=True, aggr='mean')  # 'mean' aggregation by default
        self.conv2 = SAGEConv(16, out_channels, normalize=True, aggr='mean')
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Apply dropout
        x = self.conv2(x, edge_index)
        return x


# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGENet(10, 4).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Prepare data loaders
train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[int(len(dataset) * 0.8):], batch_size=64, shuffle=False)

# Training loop
#train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
def train():
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Validation function
def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def run_gnn():
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement before stopping
    patience_counter = 0  # Counter for tracking the number of epochs without improvement
    
    for epoch in range(500):  # You might not need 500 epochs; adjust as early stopping might kick in
        train_loss = train()
        val_loss = validate()
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Check if the current validation loss improved
        if val_loss < best_val_loss - 0.0001:  # min_delta = 0.0001, adjust as needed
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter
            # Save the best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        # Early stopping condition
        if patience_counter >= patience:
            print(f'Stopping early after {epoch+1} epochs.')
            break

    # Load the best model state if needed
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f'Best Validation Loss: {best_val_loss:.4f}')
        torch.save(model.state_dict(), f'saved_models/graphsage_{uuid.uuid4()}.pth')

def get_predictions(loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # Ensure to pass the required arguments as expected by the model's forward method
            logits = model(data.x, data.edge_index)  # Adjusted to pass all required arguments
            probs = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
            all_preds.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    return np.vstack(all_preds), np.vstack(all_labels)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(preds, labels, threshold=0.5518115758):
    preds = (preds > threshold).astype(int)  # apply a threshold to convert probabilities to binary outputs
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='micro', zero_division=1)
    recall = recall_score(labels, preds, average='micro', zero_division=1)
    f1 = f1_score(labels, preds, average='micro', zero_division=1)
    roc_auc = roc_auc_score(labels, preds, average='micro')  # ROC AUC can be useful for imbalanced datasets

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def get_raw_predictions(model, data_loader):
    model.eval()
    all_logits = []
    all_probabilities = []
    with torch.no_grad():
        for data in data_loader:
            logits = model(data.x, data.edge_index)  # Assuming outputs are logits
            probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
            all_logits.append(logits.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_probabilities, axis=0)

def optimize_threshold(probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels.flatten(), probs.flatten())
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    return thresholds[ix], fscore[ix]

run_gnn()
# Evaluate model
#model.load_state_dict(torch.load('saved_models/gnn_model.pth'))
preds, labels = get_predictions(val_loader)
metrics = calculate_metrics(preds, labels)
print(metrics)
# Load the model
#
model.eval()

logits, probabilities = get_raw_predictions(model, val_loader)
# Assuming logits and probabilities are already obtained from get_raw_predictions
optimal_threshold, optimal_fscore = optimize_threshold(probabilities, labels)
print(f'Optimal Threshold: {optimal_threshold}, F-Score: {optimal_fscore}')

# Assuming you have access to the dataset labels
print("Distribution of Labels Across Dataset:", np.mean(labels, axis=0))  # This will show the average per label