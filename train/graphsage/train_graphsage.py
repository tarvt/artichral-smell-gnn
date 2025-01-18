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
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, NNConv , GraphConv
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.nn import NNConv, GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import os

#from ...MicroservicesDataset.dataset import MicroservicesDataset
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../MicroservicesDataset'))
from dataset import MicroservicesDataset

def main():
    dataset = MicroservicesDataset(root='E:\Masters_degree\project code\MicroservicesDataset')
    testdaset = MicroservicesDataset(root='E:\Masters_degree\project code\MicroservicesTestset')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGENet(14, 4).to(device)
    model.apply(init_weights)
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
    class_weights = 1. / torch.tensor([1.8620, 0.2787, 0.8974, 0.9619*2], dtype=torch.float32) 
    class_weights = class_weights / class_weights.sum() * 4  # Normalize to keep the same scale
    class_weights = class_weights.to(device)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    model = start_train(scheduler , model , train_loader ,val_loader, device , optimizer , criterion)

    preds, labels, probabilities = get_predictions(val_loader, model , device)
    print(f"preds lenght : {len(preds)}")
    print(f"labels lenght : {len(labels)}")
    metrics = calculate_metrics(preds, labels)
    print(metrics)

    optimal_threshold = optimize_threshold(preds, labels)
    print(f'Optimal Threshold: {optimal_threshold}')

    # Assuming you have access to the dataset labels
    print("Distribution of Labels Across Dataset:", np.mean(labels, axis=0))  # This will show the average per label
    val_metrics = evaluate_model(val_loader, model , device)
    for label, metrics in val_metrics.items():
        print(f"Metrics for {label}: {metrics}")

    print("test_data set : ")
    preds, labels , probabilities= get_predictions(test_loader , model , device)
    metrics = calculate_metrics(preds, labels)
    print(metrics)
    # Use this function to find the best threshold at the end of your training
    optimal_threshold = optimize_threshold(preds, labels)
    print(f'Optimal Threshold: {optimal_threshold}')
    # Assuming you have access to the dataset labels
    print("Distribution of Labels Across Dataset:", np.mean(labels, axis=0))  # This will show the average per label
    test_metrics = evaluate_model(test_loader, model  , device)
    for label, metrics in test_metrics.items():
        print(f"Metrics for {label}: {metrics}")


def train_model(model, train_loader, val_loader, device, optimizer, criterion, epochs):
    # Dictionaries to store metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()

            # Multi-label classification
            predicted = torch.sigmoid(outputs)  # Apply sigmoid for multi-label
            predicted = (predicted > 0.5).float()  # Binarize the predictions

            # Calculate accuracy
            correct = (predicted == data.y).sum().item()  # Sum correct predictions
            total = data.y.numel()  # Total number of labels (num_labels * batch_size)

            train_loss += loss.item()
            train_correct += correct
            train_total += total

        # Calculate epoch training accuracy
        train_accuracy = train_correct / train_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_accuracy)

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(outputs, data.y)
                val_loss += loss.item()

                # Multi-label classification
                predicted = torch.sigmoid(outputs)
                predicted = (predicted > 0.5).float()

                # Calculate accuracy
                correct = (predicted == data.y).sum().item()
                total = data.y.numel()

                val_correct += correct
                val_total += total

        # Calculate epoch validation accuracy
        val_accuracy = val_correct / val_total
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_accuracy)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

    return history

def calculate_dynamic_class_weights(labels , device):
    class_counts = labels.sum(axis=0)
    class_weights = 1. / (class_counts + 1e-5)  # Avoid division by zero
    class_weights = class_weights / class_weights.mean()  # Normalizing
    return class_weights.to(device)

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

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

def perturb_edges(edge_index, num_nodes, prob=0.1):
    # Randomly remove edges
    remove_mask = torch.rand(edge_index.shape[1]) < prob
    edge_index = edge_index[:, ~remove_mask]  # Remove edges based on the mask
    return edge_index

def feature_dropout(x, drop_prob=0.2):
    mask = torch.rand_like(x) > drop_prob  # Random mask for dropout
    x = x * mask.float()  # Apply the dropout mask
    return x

def plot_metrics(history):
    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train(model , train_loader , device , optimizer , criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #edge_index = perturb_edges(data.edge_index, num_nodes=data.x.size(0), prob=0.1) 
        x = feature_dropout(data.x, drop_prob=0.2)
        out = model(x , data.edge_index, data.edge_attr)
        loss = criterion(out, data.y) #.float()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #scheduler.step(total_loss / len(train_loader))
    return total_loss / len(train_loader)


def validate(model , val_loader , device , criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def start_train(scheduler , model ,  train_loader ,val_loader ,  device , optimizer , criterion):
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement before stopping
    patience_counter = 0  # Counter for tracking the number of epochs without improvement
    
    for epoch in range(50):  # You might not need 500 epochs; adjust as early stopping might kick in
        train_loss = train(model , train_loader , device , optimizer , criterion)
        val_loss = validate(model , val_loader , device , criterion)
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
    
    return model

def get_predictions(loader, model , device):
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

def evaluate_model(loader, model , device):
    preds, labels, probabilities = get_predictions(loader, model , device)

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




def get_correct_predictions(predictions, labels, optimal_thresholds , device):
    # Ensure predictions and labels are tensors
    preds_tensor = torch.tensor(predictions).float().to(device)  # Convert predictions to tensor
    labels_tensor = torch.tensor(labels).float().to(device)  # Convert labels to tensor
    optimal_thresholds_tensor = torch.tensor(optimal_thresholds).float().to(device)  # Convert optimal thresholds to tensor

    # Apply optimal thresholds to get predicted binary labels
    predicted_labels = (preds_tensor > optimal_thresholds_tensor).float()

    # Check for correctness: we consider the prediction correct if all labels match
    correct_predictions = (predicted_labels == labels_tensor).all(dim=1)  # Check if all labels are correct for each graph

    return correct_predictions




if __name__ == "__main__":
    main()