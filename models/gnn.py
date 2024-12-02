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
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        # Update the in_channels from 10 to 14
        self.conv1 = GCNConv(in_channels=14, out_channels=16)  # Now expects 14 features as input
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
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
model = GNN(10, 4).to(device)
model.apply(init_weights)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Prepare data loaders
train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[int(len(dataset) * 0.8):], batch_size=64, shuffle=False)
test_loader = DataLoader(testdaset, batch_size=64, shuffle=False)
y_train = []
for data in train_loader:
    y_train.append(data.y)

y_train = data.y.cpu().numpy() 
y_train_flat = np.argmax(y_train, axis=1) 
class_weights = compute_class_weight('balanced', classes=[0, 1, 2, 3], y=y_train_flat)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
class_weights = 1. / torch.tensor([1.8620, 0.2787, 0.8974, 0.9619], dtype=torch.float32) #[0.01538021 0.09767975 0.03045193 0.02705859]
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

    # Apply the optimal thresholds to the raw probabilities
    adjusted_preds = apply_thresholds(probabilities, optimal_thresholds)
    
    # Now calculate metrics with the adjusted predictions
    metrics = calculate_metrics_per_label(adjusted_preds, labels)
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

def calculate_roc_auc(preds, labels):
    auc_scores = []
    for i in range(preds.shape[1]):
        auc = roc_auc_score(labels[:, i], preds[:, i])
        auc_scores.append(auc)
    return np.mean(auc_scores)

def calculate_metrics_per_label(preds, labels):
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
#model.load_state_dict(torch.load('saved_models/gnn_model_74ddab15-d608-4033-bb8f-b3eafecf8a57.pth'))
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


"""class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Use edge_attr for convolution
        # If you have multiple edge attributes, you can try combining them (e.g., by averaging or concatenating)
        # Let's just pass them through without any aggregation for now
        edge_attr = edge_attr.mean(dim=1)  # Reduce 3D to 1D (e.g., average the edge features)
        
        # First convolution layer
        x = F.relu(self.conv1(x, edge_index, edge_attr))  # Apply edge_attr
        x = F.dropout(x, training=self.training)
        
        # Second convolution layer
        x = self.conv2(x, edge_index, edge_attr)  # Apply edge_attr again
        return x
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GATConv(in_channels, 16, edge_dim=3)  # Use edge_dim to specify the dimensionality of edge_attr
        self.conv2 = GATConv(16, out_channels, edge_dim=3)  # Same here

    def forward(self, x, edge_index, edge_attr):
        # Apply GATConv which inherently uses edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        
        # Apply dropout for regularization
        x = F.dropout(x, training=self.training)
        
        # Apply second GATConv layer
        x = self.conv2(x, edge_index, edge_attr)
        return x        
        
        class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GATConv(in_channels, 16, edge_dim=3)  # Edge attributes included
        self.conv2 = GATConv(16, out_channels, edge_dim=3)  # Edge attributes included

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))  # Attention mechanism applied to edge_attr
        x = F.dropout(x, training=self.training)  # Dropout for regularization
        x = self.conv2(x, edge_index, edge_attr)
        return x
        """
"""def prepare_data_for_record(dataset, record_index):
    # Extract the data for the specific record (graph)
    graph = dataset[record_index]

    # Prepare the node features (x), edge indices (edge_index), edge attributes (edge_attr), and labels (y)
    x = graph.x  # Node features
    edge_index = graph.edge_index  # Edge indices (shape: [2, num_edges])
    edge_attr = graph.edge_attr if 'edge_attr' in graph else None  # Edge attributes (if available)
    y = graph.y  # Ground truth labels

    # Return the data as a Data object in PyTorch Geometric format
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def filter_and_save_correct_data(test_data_path, model, optimal_thresholds, output_path):
    correct_data = []
    with open(test_data_path, 'r') as infile:
        for idx, line in enumerate(infile):
            # Prepare the data for the current record
            data_for_record = prepare_data_for_record(dataset, idx)

            # Get predictions for this specific record
            model.eval()
            with torch.no_grad():
                logits = model(data_for_record.x, data_for_record.edge_index, data_for_record.batch)
                probs = torch.sigmoid(logits)

            # Apply the optimal thresholds
            pred_labels = (probs > torch.tensor(optimal_thresholds).to(probs.device)).cpu().numpy()

            # Check if all predictions are correct
            if (pred_labels == data_for_record.y.cpu().numpy()).all():
                correct_data.append(line)  # If prediction is correct, save the record

    # Save the correct data to the output file
    with open(output_path, 'w') as outfile:
        outfile.writelines(correct_data)

    print(f"Correct predictions saved to {output_path}")"""

"""def aggregate_predictions(predictions, labels, optimal_thresholds):
    correct_predictions = []

    # Iterate over each graph in the dataset
    idx = 0
    while idx < len(labels):
        # Get the number of nodes for this graph
        num_nodes = len(labels[idx])  # Each graph has a variable number of nodes

        # Get the node-level predictions and labels for the current graph
        graph_predictions = predictions[idx:idx+num_nodes]
        graph_labels = labels[idx:idx+num_nodes]

        # Compare node-level predictions with the labels
        # Check if all nodes in the graph are predicted correctly
        correct = True
        for i in range(graph_labels.shape[0]):
            node_predictions = graph_predictions[i]
            node_labels = graph_labels[i]

            # Compare each node's prediction to the optimal threshold
            for label_idx in range(len(optimal_thresholds)):
                if (node_predictions[label_idx] > optimal_thresholds[label_idx]) != node_labels[label_idx]:
                    correct = False
                    break

            if not correct:
                break
        
        correct_predictions.append(correct)
        
        # Move to the next graph (skip the nodes of the current graph)
        idx += num_nodes
    
    return correct_predictions

def filter_and_save_correct_data(test_data_path, predictions, labels, optimal_thresholds, output_path):
    correct_predictions = []  # This will store correct predictions at the graph level

    # Start iterating over the test data and node-level predictions
    with open(test_data_path, 'r') as infile, open(output_path, 'w') as outfile:
        # Read the test dataset line by line (this is at the graph level)
        for idx, line in enumerate(infile):
            # Get the prediction and label for the current graph (indexed by `idx`)
            graph_predictions = predictions[idx]
            graph_labels = labels[idx]
            
            # Check if the entire graph's predictions are correct (i.e., all nodes are predicted correctly)
            # Apply optimal thresholds to the predictions for this graph
            correct_nodes = (graph_predictions > optimal_thresholds) == graph_labels
            
            # If all nodes are predicted correctly, add the graph to the correct predictions
            if correct_nodes.all():  # Only save if all nodes are correct
                outfile.write(line)  # Write the entire graph record to the output file
                correct_predictions.append(True)  # Mark this graph as correctly predicted
            else:
                correct_predictions.append(False)  # Mark this graph as incorrectly predicted

    print(f"Number of correctly predicted graphs: {len(correct_predictions)}")"""


"""
model.load_state_dict(torch.load('saved_models/gnn_model_74ddab15-d608-4033-bb8f-b3eafecf8a57.pth'))
preds, labels , probabilities= get_predictions(test_loader , model)
test_data_path = 'MicroservicesTestset/handgenerated1.jsonl'  # Path to your test data
output_path = 'correct_data.jsonl'  # Path where you want to save correct data


all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())  # Store predictions
        all_labels.append(data.y.cpu().numpy())  # Store ground truth labels

# Convert the list of predictions and labels to numpy arrays
predictions = np.vstack(all_preds)
labels = np.vstack(all_labels)

optimal_thresholds = calculate_optimal_thresholds(probabilities, labels)

# Filter and save correct data
filter_and_save_correct_data(test_data_path=test_data_path, 
                             predictions=predictions, 
                             labels=labels, 
                             optimal_thresholds=optimal_thresholds, 
                             output_path=output_path)
                             
                    



def filter_and_save_correct_data(test_data_path, predictions, labels, optimal_thresholds, output_path):
    correct_predictions = []

    # Iterate through each test record and check if the prediction is correct
    with open(test_data_path, 'r') as infile, open(output_path, 'w') as outfile:
        for idx, line in enumerate(infile):
            # For each record, check if the prediction matches the ground truth
            is_correct = check_if_correct(predictions[idx], labels[idx], optimal_thresholds)
            if is_correct:
                outfile.write(line)  # Save the correct record to the output file
                correct_predictions.append(True)
            else:
                correct_predictions.append(False)

    print(f"Number of correct predictions saved: {len(correct_predictions)}")

def check_if_correct(prediction, label, optimal_thresholds):
    # Apply thresholds to predictions
    adjusted_prediction = (prediction > optimal_thresholds).astype(int)
    adjusted_label = (label > 0.5).astype(int)  # You can use 0.5 as a default threshold for ground truth labels

    # Compare prediction and label
    return np.array_equal(adjusted_prediction, adjusted_label)

def test_on_correct_data(output_path, model, device):
    # Load the saved correct predictions data
    with open(output_path, 'r') as f:
        correct_data = [json.loads(line.strip()) for line in f]

    # Assuming correct_data is now a list of dictionaries, each representing a record
    all_preds = []
    all_labels = []
    
    # Loop over the correct data records and perform inference
    for record in correct_data:
        # Convert the record to PyTorch tensors
        x = torch.tensor(record['Nodes'], dtype=torch.float32).to(device)
        edge_index = torch.tensor(record['edge_index'], dtype=torch.long).to(device)
        edge_attr = torch.tensor(record['edge_attr'], dtype=torch.long).to(device)
        y = torch.tensor(record['labels'], dtype=torch.float32).to(device)  # Assuming 'y' is the ground truth
        
        # Pass through the model
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index, edge_attr)
            probs = torch.sigmoid(logits)
        
        # Append predictions and labels
        all_preds.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    # Convert the list of predictions and labels to numpy arrays
    predictions = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    # Now you can evaluate the predictions and calculate metrics
    metrics = calculate_metrics(predictions, labels)
    print(f"Metrics for correct data: {metrics}")

model.load_state_dict(torch.load('saved_models/gnn_model_74ddab15-d608-4033-bb8f-b3eafecf8a57.pth'))
preds, labels , probabilities= get_predictions(test_loader , model)
test_data_path = 'MicroservicesTestset/handgenerated1.jsonl'  # Path to your test data
output_path = 'correct_data.jsonl'  # Path where you want to save correct data
model.eval()
"""