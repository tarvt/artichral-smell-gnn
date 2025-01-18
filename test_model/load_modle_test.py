import torch
import json
from torch_geometric.data import Data
from classes.graphsage import GraphSAGENet  




# Define the binary label descriptions
label_descriptions = {
    0: "ESB Usage – Nodes excessively relying on an Enterprise Service Bus for simple tasks.",
    1: "Cyclic Dependency – Nodes that are part of a cyclic chain of calls.",
    2: "Inappropriate Service Intimacy – Nodes excessively interacting with or relying on another service’s private data.",
    3: "Microservice Greedy – Nodes representing services created for minimal functionalities."
}
def load_model(model_path, device):
    model = GraphSAGENet(in_channels=14, out_channels=4)  # This needs to match the training setup
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

# Optimal thresholds for each label
optimal_thresholds = torch.tensor([0.151, 0.110, 0.018, 0.122])

def prepare_data_for_testing(input_json):
    dummy_featue = [0] * 4  
    nodes_with_dummy_labels = [node + dummy_featue for node in input_json['Nodes']]
    x = torch.tensor(nodes_with_dummy_labels, dtype=torch.float)
    edge_index = torch.tensor(input_json['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(input_json['edge_attr'], dtype=torch.float)
    min_vals = x.min(0, keepdim=True)[0]
    max_vals = x.max(0, keepdim=True)[0]
    x = (x - min_vals) / (max_vals - min_vals + 1e-5)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def predict(data, model, device, thresholds):
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits)
        predictions = (probs > thresholds.to(device)).long()
    return predictions.cpu().numpy()


def interpret_predictions(predictions):
    """
    Interprets the predictions for each node and identifies which anti-patterns have been detected.

    Args:
        predictions (numpy.ndarray): A 2D array where each row corresponds to a node and each column to a binary prediction
            for an anti-pattern.

    Returns:
        results (list of dicts): A list where each element is a dictionary with detailed information about the node and the
            detected anti-patterns.
    """
    results = []
    for node_idx, node_preds in enumerate(predictions):
        node_result = {
            "Node": node_idx,
            "Anti-Patterns": [label_descriptions[i] for i, label in enumerate(node_preds) if label == 1]
        }
        results.append(node_result)
    return results


def main():
    model_path = "trained_model/gnn_model_07497e6d-6173-4108-9573-e6da9f96273a.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("test_model/test_data.json", 'r') as json_file:
        input_data = json.load(json_file)
    model = load_model(model_path, device)
    data = prepare_data_for_testing(input_data)
    predictions = predict(data, model, device, optimal_thresholds)
    results = interpret_predictions(predictions)
    for result in results:
        print(f"Node {result['Node']}:")
        if result['Anti-Patterns']:
            for ap in result['Anti-Patterns']:
                print(f"  - {ap}")
        else:
            print("  - No anti-patterns detected.")

if __name__ == "__main__":
    main()