import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from collections import Counter
import numpy as np
test_path = 'MicroservicesDataset/test.jsonl'

class CyclicDependencyService:
    def __init__(self, edge_index):
        self.edge_index = edge_index
        self.adj_list = self.build_adjacency_list(edge_index)

    def build_adjacency_list(self, edge_index):
        from collections import defaultdict
        adj_list = defaultdict(list)
        for src, dst in zip(edge_index[0], edge_index[1]):
            adj_list[src].append(dst)
        return adj_list

    def is_cyclic_util(self, node, visited, rec_stack):
        if rec_stack[node]:
            return True
        if visited[node]:
            return False

        visited[node] = True
        rec_stack[node] = True

        for neighbor in self.adj_list[node]:
            if self.is_cyclic_util(neighbor, visited, rec_stack):
                return True

        rec_stack[node] = False
        return False

    def is_cyclic(self):
        # Handle the case where there are no edges
        if not self.edge_index[0] or not self.edge_index[1]:  # Check if either list is empty
            return False  # No edges means no cycles possible

        # Find the maximum index in edge_index to ensure all possible nodes are covered
        max_index = max(max(self.edge_index[0]), max(self.edge_index[1])) + 1  # Adding 1 because index starts at 0

        visited = [False] * max_index
        rec_stack = [False] * max_index

        for node in range(max_index):  # Adjust the range to max_index
            if not visited[node]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return True
        return False




def get_esb_context(edge_index):
    if edge_index != [[], []]:
        incoming = Counter(edge_index[1])
        outgoing = Counter(edge_index[0])
        total_connections = {node: incoming.get(node, 0) + outgoing.get(node, 0) for node in set(incoming) | set(outgoing)}

        # Calculate the threshold for being considered an ESB
        # Here, we look for nodes whose total connections are significantly higher than the average
        avg_connections = sum(total_connections.values()) / len(total_connections)
        threshold = avg_connections * 1.5  # Setting a simpler threshold, 50% above the average
        for node, count in total_connections.items():
            if count > threshold:
                return True
        return False
    else :
        return False



def detect_greedy_microservices(nodes):
    # Simulated detection of greedy microservices based on entity and static file counts
    greedy_services = []
    # [5]: Instance Count, [6]: Number of Dependencies
    for index, node in enumerate(nodes):
        # Node Feature extraction for simulation
        instance_count = node[5]
        dependency_count = node[6]
        # Simplified heuristic: consider a microservice greedy if it has few dependencies but many instances
        if instance_count > 5 and dependency_count < 3:
            greedy_services.append(index)
            return True
    
    # Construct output
    #labels = [0]*4 
    #for i in greedy_services:
    #    labels[i] = 1  # Mark the node as greedy
    #    return True
 
    return False

def detect_inappropriate_service_intimacy(nodes, edge_index):
    if edge_index != [[], []]:
        # Simulate entity lists associated with each node
        entities_per_node = {
            0: {"entity1", "entity2", "entity3"},
            1: {"entity2", "entity4"},
            2: {"entity3", "entity5"}
        }

        # Create a context to store results
        intimacy_issues = []

        # Check each edge for shared entities that might indicate intimacy issues
        for src, dst in zip(edge_index[0], edge_index[1]):
            if src != dst:  # ensure we are not comparing the same service
                entities_src = entities_per_node.get(src, set())
                entities_dst = entities_per_node.get(dst, set())

                # Calculate intersection and similarity
                shared_entities = entities_src.intersection(entities_dst)
                if entities_src and entities_dst:  # avoid division by zero
                    similarity = len(shared_entities) / max(len(entities_src), len(entities_dst))
                    if similarity > 0.5:  # threshold for determining intimacy
                        intimacy_issues.append({
                            "from": src,
                            "to": dst,
                            "similarity": similarity
                        })

        return json.dumps(intimacy_issues, indent=4)

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def evaluate_cyclic_dependency(data):
    results = []
    actuals = []
    for graph in data:
        nodes = graph['Nodes']
        edge_index = [graph['edge_index'][0], graph['edge_index'][1]]
        labels = graph['labels']
        
        # Actual label for cyclic dependency
        actual_cyclic = any(label[0] for label in labels)
        actuals.append(actual_cyclic)
        
        # Detection using CyclicDependencyService
        detector = CyclicDependencyService(edge_index)
        detected_cyclic = detector.is_cyclic()
        results.append(detected_cyclic)
    
    return results, actuals

def convert_labels(labels):
    """Ensure labels are integers."""
    return np.array(labels, dtype=int)

def evaluate_service_intimacy(data):
    results = []
    actuals = []

    for graph in data:
        # Aggregate labels for each graph individually
        graph_actual = any(label[2] for label in graph['labels'])  #  index 2 : service_intimacy
        actuals.append(graph_actual)

        # Process each graph for results
        nodes = graph['Nodes']
        edge_index = [graph['edge_index'][0], graph['edge_index'][1]]
        result = detect_inappropriate_service_intimacy(nodes, edge_index)
        results.append(result)
    
    # Calculate metrics
    accuracy = accuracy_score(actuals, results)
    precision = precision_score(actuals, results)
    recall = recall_score(actuals, results)
    f1 = f1_score(actuals, results)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def evaluate_service_ESB(data):
    results = []
    actuals = []

    for graph in data:
        # Aggregate labels for each graph individually
        graph_actual = any(label[0] for label in graph['labels'])  #  index 2 : Microservice Greedy
        actuals.append(graph_actual)

        # Process each graph for results
        nodes = graph['Nodes']
        edge_index = [graph['edge_index'][0], graph['edge_index'][1]]
        result = get_esb_context(edge_index)
        results.append(result)
    
    # Calculate metrics
    accuracy = accuracy_score(actuals, results)
    precision = precision_score(actuals, results)
    recall = recall_score(actuals, results)
    f1 = f1_score(actuals, results)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def evaluate_service_gready(data):
    results = []
    actuals = []

    for graph in data:
        # Aggregate labels for each graph individually
        graph_actual = any(label[3] for label in graph['labels'])  #  index 2 : Microservice Greedy
        actuals.append(graph_actual)

        # Process each graph for results
        nodes = graph['Nodes']
        edge_index = [graph['edge_index'][0], graph['edge_index'][1]]
        result = detect_greedy_microservices(nodes)
        results.append(result)
    
    # Calculate metrics
    accuracy = accuracy_score(actuals, results)
    precision = precision_score(actuals, results)
    recall = recall_score(actuals, results)
    f1 = f1_score(actuals, results)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def aggregate_labels(labels):
    # Assuming `labels` is a list of lists where each sublist corresponds to a node's label set
    aggregated_label = any(any(label) for label in labels)  # `any(label)` checks if any condition is true for a node
    return aggregated_label
def detect_inappropriate_service_intimacy(nodes, edge_index):
    entities_per_node = {
        0: {"entity1", "entity2", "entity3"},
        1: {"entity2", "entity4"},
        2: {"entity3", "entity5"}
    }

    intimacy_detected = False  # Flag to indicate detection of any intimacy issue

    for src, dst in zip(edge_index[0], edge_index[1]):
        if src != dst:  # Avoid comparing the same service
            entities_src = entities_per_node.get(src, set())
            entities_dst = entities_per_node.get(dst, set())
            shared_entities = entities_src.intersection(entities_dst)

            if entities_src and entities_dst:  # Avoid division by zero
                similarity = len(shared_entities) / max(len(entities_src), len(entities_dst))
                if similarity > 0.5:  # Threshold for determining intimacy
                    intimacy_detected = True
                    break  # Exit if any intimacy issue is detected

    return intimacy_detected  # Return a boolean rather than a JSON string

def main():
    data = load_data(test_path)
    
    # Evaluate Cyclic Dependency
    cyclic_results, cyclic_actuals = evaluate_cyclic_dependency(data)
    print('Cyclic Dependency Metrics:')
    print(f"Accuracy: {accuracy_score(cyclic_actuals, cyclic_results)}")
    print(f"Precision: {precision_score(cyclic_actuals, cyclic_results)}")
    print(f"Recall: {recall_score(cyclic_actuals, cyclic_results)}")
    print(f"F1 Score: {f1_score(cyclic_actuals, cyclic_results )}")

    # Evaluate other services
    # Example: evaluating ESB Service
    metrics_intimacy  = evaluate_service_intimacy(data)  
    print('intimacy Service Metrics:')
    print(metrics_intimacy )

    metrics_greedy  = evaluate_service_gready(data)  # Change the function as necessary
    print('Microservice Greedy Metrics:')
    print(metrics_greedy )

    metrics_ESB  = evaluate_service_ESB(data)  
    print('ESB Service Metrics:')
    print(metrics_ESB )


if __name__ == '__main__':
    main()