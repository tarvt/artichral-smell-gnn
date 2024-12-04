import torch
from torch_geometric.data import Data, InMemoryDataset
import json
import logging
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import precision_recall_curve
import numpy as np
import os 
import uuid
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv, GATConv, NNConv , GraphConv
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

dataset_name = 'handgenerated1.jsonl'#'handgenerated.jsonl' #'graph_dataset.jsonl'
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
                    y = torch.tensor(graph['labels'], dtype=torch.float)
                    x = torch.cat((x, y), dim=1)
                    x = self.normalize_features(x)  # Normalize features
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



logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
dataset = MicroservicesDataset(root='E:\Masters_degree\project code\MicroservicesDataset')
testdaset = MicroservicesDataset(root='E:\Masters_degree\project code\MicroservicesTestset')
print("dataset done ----------------------------------------------")

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(GNN, self).__init__()

        # Define GCN layers (same as your original setup)
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

        # Edge feature transformation (for GCNConv)
        self.edge_fc = torch.nn.Linear(edge_attr_dim, 16)

    def forward(self, x, edge_index, edge_attr):
        # Transform edge_attr
        edge_attr_transformed = self.edge_fc(edge_attr)

        # Combine edge_attr with node features before passing through GCNConv
        x = F.relu(self.conv1(x, edge_index, edge_attr_transformed))  # Pass edge_attr to conv1
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr_transformed)  # Pass edge_attr to conv2
        return x



        #print(f"edge_index.shape() : {edge_index.shape}")
        #print(f"edge_attr.shape() : {edge_attr.shape}")
# Calculate dynamic class weights
def calculate_dynamic_class_weights(labels):
    class_counts = labels.sum(axis=0)
    class_weights = 1. / (class_counts + 1e-5)  # Avoid division by zero
    class_weights = class_weights / class_weights.mean()  # Normalizing
    return class_weights.to(device)

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def perturb_edges(edge_index, num_nodes, prob=0.1):
    # Randomly remove edges
    remove_mask = torch.rand(edge_index.shape[1]) < prob
    edge_index = edge_index[:, ~remove_mask]  # Remove edges based on the mask
    return edge_index

def feature_dropout(x, drop_prob=0.2):
    mask = torch.rand_like(x) > drop_prob  # Random mask for dropout
    x = x * mask.float()  # Apply the dropout mask
    return x

# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(14, 4).to(device)
model.apply(init_weights)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Prepare data loaders
train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[int(len(dataset) * 0.8):], batch_size=64, shuffle=False)
val_loader = DataLoader(testdaset, batch_size=64, shuffle=False)
test_loader = DataLoader(testdaset, batch_size=64, shuffle=False)
y_train = []
for data in train_loader:
    y_train.append(data.y)

y_train = data.y.cpu().numpy() 
y_train_flat = np.argmax(y_train, axis=1) 
class_weights = compute_class_weight('balanced', classes=[0, 1, 2, 3], y=y_train_flat)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
class_weights = 1. / torch.tensor([1.8620, 0.2787, 0.8974, 0.9619*2], dtype=torch.float32) #[0.01538021 0.09767975 0.03045193 0.02705859]
class_weights = class_weights / class_weights.sum() * 4  # Normalize to keep the same scale
class_weights = class_weights.to(device)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Decay LR every 50 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
# Adaptive Learning Rate Scheduler


# Training loop
#train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        edge_index = perturb_edges(data.edge_index, num_nodes=data.x.size(0), prob=0.1) 
        x = feature_dropout(data.x, drop_prob=0.2)
        out = model(x , edge_index, data.edge_attr)
        loss = criterion(out, data.y) #.float()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #scheduler.step(total_loss / len(train_loader))
    return total_loss / len(train_loader)


# Validation function
def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(val_loader)



def run_gnn():
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement before stopping
    patience_counter = 0  # Counter for tracking the number of epochs without improvement
    
    for epoch in range(50):  # You might not need 500 epochs; adjust as early stopping might kick in
        train_loss = train()
        val_loss = validate()
        scheduler.step(val_loss)
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
        uuii = uuid.uuid4()
        print(f"gnn_model_{uuii}")
        torch.save(model.state_dict(), f'saved_models/gnn_model_{uuii}.pth')

def get_predictions(loader, model):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # To store raw probabilities
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.edge_attr)  # Get raw output
            probs = torch.sigmoid(logits)  # Convert logits to probabilities
            
            all_preds.append((probs > 0.5).float().cpu().numpy())  # Apply default threshold to get binary predictions (default 0.5)
            all_labels.append(data.y.cpu().numpy())  # Collect the true labels
            all_probs.append(probs.cpu().numpy())  # Store raw probabilities for threshold optimization

    return np.vstack(all_preds), np.vstack(all_labels), np.vstack(all_probs)  # Return raw probabilities as well



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
def calculate_optimal_thresholds(predictions, labels):

    optimal_thresholds = []
    for i in range(predictions.shape[1]):  # Loop through each label
        label_preds = predictions[:, i]
        label_true = labels[:, i]
        
        best_threshold = 0
        best_f1 = 0
        # Try different thresholds (you can fine-tune this)
        for threshold in np.linspace(0, 1, 101):  # Test thresholds from 0 to 1
            preds = (label_preds > threshold).astype(float)
            f1 = f1_score(label_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
    
    return np.array(optimal_thresholds)

def apply_thresholds(predictions, thresholds):

    adjusted_preds = (predictions > thresholds).astype(float)
    return adjusted_preds

def evaluate_model(loader, model):
    preds, labels, probabilities = get_predictions(loader, model)

    # Calculate optimal thresholds using raw probabilities
    optimal_thresholds = calculate_optimal_thresholds(probabilities, labels)
    print(f"Optimal thresholds: {optimal_thresholds}")



    # Now calculate metrics with the adjusted predictions
    metrics = calculate_metrics_per_label(probabilities, optimal_thresholds , labels)
    return metrics

def calculate_metrics(preds, labels, threshold= 0.149):
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
        "f1_score": f1,
    }


def optimize_threshold(preds, labels):
    precision, recall, thresholds = precision_recall_curve(labels.ravel(), preds.ravel())
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # avoid division by zero
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

def adjust_threshold_for_precision(predictions, labels, target_label_index):
    precision, recall, thresholds = precision_recall_curve(labels[:, target_label_index], predictions[:, target_label_index])
    # Find the threshold that gives the highest precision while maintaining a reasonable recall
    threshold_index = np.argmax(precision[recall > 0.5])  # assuming you want to keep recall above 50%
    optimal_threshold = thresholds[threshold_index]
    adjusted_predictions = (predictions[:, target_label_index] > optimal_threshold).astype(int)
    return adjusted_predictions

    
def calculate_roc_auc(preds, labels):
    auc_scores = []
    for i in range(preds.shape[1]):
        auc = roc_auc_score(labels[:, i], preds[:, i])
        auc_scores.append(auc)
    return np.mean(auc_scores)

def calculate_metrics_per_label(probabilities, optimal_thresholds , labels):
        # Apply the optimal thresholds to the raw probabilities
    
    preds = apply_thresholds(probabilities, optimal_thresholds)
    metrics = {}
    for i in range(labels.shape[1]):
        
        label_preds = preds[:, i]
        label_true = labels[:, i]
        accuracy = accuracy_score(label_true, label_preds)
        precision = precision_score(label_true, label_preds, zero_division=1)
        recall = recall_score(label_true, label_preds, zero_division=1)
        f1 = f1_score(label_true, label_preds, zero_division=1)
        roc_auc = calculate_roc_auc(preds, labels)#roc_auc_score(label_true, label_preds) if len(np.unique(label_true)) > 1 else "NA"
        metrics[f'Label {i}'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    return metrics




def get_correct_predictions(predictions, labels, optimal_thresholds):
    # Ensure predictions and labels are tensors
    preds_tensor = torch.tensor(predictions).float().to(device)  # Convert predictions to tensor
    labels_tensor = torch.tensor(labels).float().to(device)  # Convert labels to tensor
    optimal_thresholds_tensor = torch.tensor(optimal_thresholds).float().to(device)  # Convert optimal thresholds to tensor

    # Apply optimal thresholds to get predicted binary labels
    predicted_labels = (preds_tensor > optimal_thresholds_tensor).float()

    # Check for correctness: we consider the prediction correct if all labels match
    correct_predictions = (predicted_labels == labels_tensor).all(dim=1)  # Check if all labels are correct for each graph

    return correct_predictions


#"""
run_gnn()
# Evaluate model
#model.load_state_dict(torch.load('saved_models/gnn_model_5bcef763-0f6f-4c33-9527-06bdf8885ab2.pth'))
#preds, labels = get_predictions(val_loader , model)
preds, labels, probabilities = get_predictions(val_loader, model)
print(f"preds lenght : {len(preds)}")
print(f"labels lenght : {len(labels)}")
metrics = calculate_metrics(preds, labels)
print(metrics)
# Load the model


# Use this function to find the best threshold at the end of your training
optimal_threshold = optimize_threshold(preds, labels)
print(f'Optimal Threshold: {optimal_threshold}')
print(f"dataset name :{dataset_name}")
# Assuming you have access to the dataset labels
print("Distribution of Labels Across Dataset:", np.mean(labels, axis=0))  # This will show the average per label
val_metrics = evaluate_model(val_loader, model)
for label, metrics in val_metrics.items():
    print(f"Metrics for {label}: {metrics}")

print("test_dataaaa : ")
preds, labels , probabilities= get_predictions(test_loader , model)
metrics = calculate_metrics(preds, labels)
print(metrics)

# Use this function to find the best threshold at the end of your training
optimal_threshold = optimize_threshold(preds, labels)
print(f'Optimal Threshold: {optimal_threshold}')
print(f"dataset name :{dataset_name}")
# Assuming you have access to the dataset labels
print("Distribution of Labels Across Dataset:", np.mean(labels, axis=0))  # This will show the average per label
test_metrics = evaluate_model(test_loader, model)
for label, metrics in test_metrics.items():
    print(f"Metrics for {label}: {metrics}")#"""

