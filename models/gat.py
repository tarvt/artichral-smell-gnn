import torch
from torch_geometric.data import Data, InMemoryDataset
import json
import logging
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.nn import GATConv
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

class MicroservicesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MicroservicesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        path = self.raw_paths[0]  # Assuming the file is in the 'raw' directory under 'root'

        with open(path, 'r') as f:
            for line_number, line in enumerate(f):
                try:
                    graph = json.loads(line.strip())
                    x = torch.tensor(graph['Nodes'], dtype=torch.float)
                    
                    # Remove outliers from each feature in x
                    x = self.remove_outliers(x)

                    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
                    edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float)
                    y = torch.tensor(graph['labels'], dtype=torch.long)  # Assuming labels are to be used as class indices

                    # Validate edge_index shape
                    if edge_index.size(0) != 2:
                        if edge_index.size(1) == 2:
                            edge_index = edge_index.t()
                        else:
                            raise ValueError("Edge index must be of shape [2, num_edges]")

                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding failed at line {line_number}: {e}")
                except Exception as e:
                    logging.error(f"An error occurred at line {line_number}: {str(e)}")

        if data_list:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            logging.error("No valid data was loaded.")

    def remove_outliers(self, x):
        # Convert tensor to numpy for easier manipulation
        x_np = x.numpy()
        Q1 = np.percentile(x_np, 25, axis=0)
        Q3 = np.percentile(x_np, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Use broadcasting to mask values outside the bounds
        mask = (x_np >= lower_bound) & (x_np <= upper_bound)
        # Filter outliers
        x_filtered = x_np[mask].reshape(-1, x_np.shape[1])
        return torch.tensor(x_filtered, dtype=torch.float)

# Setup basic configuration for logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
dataset = MicroservicesDataset(root='/tmp/MicroservicesDataset')
print("Dataset loaded and processed.")

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=4, dropout=0.3)  # Reduced dropout and heads
        self.conv2 = GATConv(8 * 4, num_classes, heads=1, concat=False, dropout=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.3, training=self.training)  # Reduced dropout in training
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Setup the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)  # Set seed for reproducibility
np.random.seed(0)

# Assuming `dataset` is your dataset instance loaded with your custom dataset class
model = GAT(dataset.num_node_features, dataset.num_classes).to(device)
optimizer = Adam(model.parameters(), lr=0.0009, weight_decay=9e-4)
class_counts = [sum(dataset.y == i) for i in range(dataset.num_classes)]
#class_weights = [sum(class_counts) / c for c in class_counts]
#weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = BCEWithLogitsLoss()


data_loader = DataLoader(dataset, batch_size=512, shuffle=True)
train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=1024, shuffle=True)
val_loader = DataLoader(dataset[int(len(dataset) * 0.8):], batch_size=1024, shuffle=True)

def train():
    model.train()
    total_loss = 0
    for data in data_loader:  # Assuming dataset is iterable and returns Data objects
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.to(device))  # Ensure labels are also moved to device
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.to(device))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def run_gat():
    # Train and validate the model
    for epoch in range(20):
        train_loss = train()
        val_loss = validate()
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        torch.save(model.state_dict(), 'saved_models/gat_model_02.pth')



"""def get_predictions(loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)  # get logits from the model
            probs = torch.sigmoid(logits)  # apply sigmoid to convert logits to probabilities
            all_preds.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    return np.vstack(all_preds), np.vstack(all_labels)"""

def get_predictions(loader):
    model.eval()
    predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds = torch.sigmoid(out)  # Apply sigmoid to output if using BCEWithLogitsLoss to get probabilities
            predictions.append(preds.cpu())
            all_labels.append(data.y.cpu())

    # Concatenate all batches into single tensors
    predictions = torch.cat(predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return predictions, all_labels

def calculate_metrics(predictions, labels):
    # Assuming predictions are sigmoid outputs that need to be thresholded to obtain binary outcomes
    binary_predictions = (predictions > 0.5).float()  # Threshold probabilities at 0.5
    
    # Calculate metrics
    accuracy = accuracy_score(labels.numpy(), binary_predictions.numpy())
    precision = precision_score(labels.numpy(), binary_predictions.numpy(), average='macro', zero_division=0)
    recall = recall_score(labels.numpy(), binary_predictions.numpy(), average='macro')
    f1 = f1_score(labels.numpy(), binary_predictions.numpy(), average='macro')
    
    # ROC AUC might need special handling if labels are not binary
    try:
        roc_auc = roc_auc_score(labels.numpy(), predictions.numpy(), average='macro', multi_class='ovr')
    except ValueError as e:
        print(f"ROC AUC calculation error: {e}")
        roc_auc = None  # In case ROC AUC cannot be calculated

    # Aggregate metrics into a dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc  # Note this might be None if there was an error
    }
    return metrics


"""def calculate_metrics(preds, labels, threshold=0.55):
    preds = (preds > threshold).astype(int)  # apply a threshold to convert probabilities to binary outputs
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')
    roc_auc = roc_auc_score(labels, preds, average='micro')  # ROC AUC can be useful for imbalanced datasets

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }"""

def debug_model_output(data_loader):
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            print("Predictions (Probabilities):", probs)
            print("True Labels:", data.y)
            break  # Just print for the first batch

def find_best_threshold(preds, labels):
    best_threshold = 0.5
    best_f1 = 0
    thresholds = np.linspace(0.1, 0.9, 17)
    for thresh in thresholds:
        preds_bin = (preds > thresh).astype(int)
        f1 = f1_score(labels, preds_bin, average='micro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold, best_f1


# Evaluate model
#run_gat()
preds, labels = get_predictions(val_loader)
metrics = calculate_metrics(preds, labels)
print(metrics)
# Load the model
#model.load_state_dict(torch.load('saved_models/gat_model.pth', weights_only=True))
model.eval()
#debug_model_output(train_loader)

#best_thresh, best_f1 = find_best_threshold(preds,labels )
#print("Best Threshold:", best_thresh, "with F1 Score:", best_f1)