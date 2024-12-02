import torch
from torch_geometric.data import Data, DataLoader
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve

def load_test_data(test_path, transform=None):
    test_data_list = []
    with open(test_path, 'r') as f:
        for line in f:
            graph = json.loads(line.strip())
            x = torch.tensor(graph['Nodes'], dtype=torch.float)
            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float)
            y = torch.tensor(graph['labels'], dtype=torch.float)

            if transform:
                x = transform(x)  # Apply the same normalization as used during training

            test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            test_data_list.append(test_data)

    return DataLoader(test_data_list, batch_size=32, shuffle=False)


def normalize_features(features):
    min_vals = torch.min(features, dim=0)[0]
    max_vals = torch.max(features, dim=0)[0]
    return (features - min_vals) / (max_vals - min_vals + 1e-5)


def calculate_optimal_thresholds(labels, predictions):
    thresholds = []
    for i in range(labels.shape[1]):  # Assuming labels.shape[1] is the number of labels
        precision, recall, threshold = precision_recall_curve(labels[:, i], predictions[:, i])
        F1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # avoid division by zero
        optimal_idx = np.argmax(F1_scores)
        optimal_threshold = threshold[optimal_idx] if optimal_idx < len(threshold) else 1.0
        thresholds.append(optimal_threshold)
    return thresholds

def evaluate_model(model, loader, device):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, batch=data.batch)  # adjust model inputs as necessary
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    # Calculate metrics for each label
    metrics = {}
    for i in range(labels.shape[1]):
        label_preds = (preds[:, i] > 0.5).astype(int)
        label_true = labels[:, i]
        metrics[f'Label {i}'] = {
            'accuracy': accuracy_score(label_true, label_preds),
            'precision': precision_score(label_true, label_preds, zero_division=0),
            'recall': recall_score(label_true, label_preds, zero_division=0),
            'f1_score': f1_score(label_true, label_preds, zero_division=0)
        }

    return metrics

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)  # Optional: Dropout for regularization
        x = self.conv2(x, edge_index)
        return x

def evaluate_model_with_thresholds(model, loader, device, thresholds):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, batch=data.batch)  # adjust model inputs as necessary
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    # Apply calculated thresholds to each label for prediction
    adjusted_preds = np.array([[1 if preds[i, j] >= thresholds[j] else 0 for j in range(preds.shape[1])]
                               for i in range(preds.shape[0])])

    # Calculate metrics for each label
    metrics = {}
    for i in range(labels.shape[1]):
        label_preds = adjusted_preds[:, i]
        label_true = labels[:, i]
        metrics[f'Label {i}'] = {
            'accuracy': accuracy_score(label_true, label_preds),
            'precision': precision_score(label_true, label_preds, zero_division=0),
            'recall': recall_score(label_true, label_preds, zero_division=0),
            'f1_score': f1_score(label_true, label_preds, zero_division=0)
        }

    return metrics



optimal_thresholds = [0.09957521, 0.4234537, 0.09757432, 0.14775226]

# Assuming your model is defined and loaded correctly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(10, 4).to(device)
model_path = 'saved_models/gnn_model_bddfb03c-679b-4021-9c0f-9787cd96d2bc.pth'
model.load_state_dict(torch.load(model_path))
test_loader = load_test_data('MicroservicesDataset/test.jsonl', transform=normalize_features)
test_metrics = evaluate_model_with_thresholds(model, test_loader, device, optimal_thresholds)
print(test_metrics)