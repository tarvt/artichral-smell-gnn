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
        print("Raw probabilities:", probs)  # Debugging output
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
    model_path = "trained_models/gnn_model_07497e6d-6173-4108-9573-e6da9f96273a.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = {"Nodes": [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 7954.0, 39.0], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 11.0, 6798.0, 35.0], [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 16.0, 3049.0, 12.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 6.0, 5052.0, 43.0], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 11.0, 6836.0, 23.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 2646.0, 31.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 6420.0, 49.0], [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 7.0, 7628.0, 6.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 19.0, 1851.0, 6.0], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 6.0, 4083.0, 11.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 17.0, 2673.0, 28.0]], "edge_index": [[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 8, 8, 8, 9, 9, 10, 10, 10, 10, 10], [1, 5, 10, 0, 3, 5, 1, 3, 5, 8, 1, 5, 0, 1, 2, 3, 6, 7, 8, 10, 5, 8, 5, 5, 6, 9, 8, 10, 0, 1, 4, 5, 9]], "edge_attr": [[0.35801225900650024, 0.0989205613732338, 0.40707412362098694], [168.0, 5698.0, 3557.0], [0.5873093008995056, 0.6887664794921875, 0.24040108919143677], [0.7759605050086975, 0.6382788419723511, 0.9815276861190796], [0.5187488794326782, 0.5641970634460449, 0.22194598615169525], [70.0, 2987.0, 2284.0], [9.0, 608.0, 992.0], [97.0, 229.0, 445.0], [78.0, 4142.0, 5116.0], [15.0, 838.0, 414.0], [0.8577240705490112, 0.8714771866798401, 0.7761836647987366], [138.0, 9870.0, 4687.0], [168.0, 5698.0, 3557.0], [70.0, 2987.0, 2284.0], [78.0, 4142.0, 5116.0], [138.0, 9870.0, 4687.0], [196.0, 2386.0, 8804.0], [166.0, 4693.0, 9794.0], [143.0, 9924.0, 5927.0], [116.0, 6163.0, 4437.0], [196.0, 2386.0, 8804.0], [0.6714752316474915, 0.9637337923049927, 0.9571280479431152], [166.0, 4693.0, 9794.0], [143.0, 9924.0, 5927.0], [0.3609383702278137, 0.6156208515167236, 0.8704443573951721], [0.8220771551132202, 0.26007628440856934, 0.6826091408729553], [0.9578343629837036, 0.7861849069595337, 0.0539676696062088], [0.7863875031471252, 0.2494673728942871, 0.5866664052009583], [71.0, 466.0, 949.0], [87.0, 181.0, 162.0], [82.0, 230.0, 908.0], [116.0, 6163.0, 4437.0], [0.2129400074481964, 0.39156776666641235, 0.5502252578735352]], "labels": [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]}
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