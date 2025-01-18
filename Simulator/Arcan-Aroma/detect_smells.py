import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_path = 'MicroservicesTestset/handgenerated1.jsonl'

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def arcan_detect_smells(nodes, edge_index, edge_attr):
    # Simulate the cyclic dependency detection (simple DFS-based approach)
    def dfs(node, visited, rec_stack, graph):
        visited[node] = True
        rec_stack[node] = True
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, visited, rec_stack, graph):
                    return True
            elif rec_stack[neighbor]:
                return True
        
        rec_stack[node] = False
        return False
    
    # Build the graph (adjacency list)
    graph = {i: [] for i in range(len(nodes))}
    for i, j in zip(edge_index[0], edge_index[1]):
        graph[i].append(j)
    
    # Perform DFS on all nodes to detect any cycle
    visited = [False] * len(nodes)
    rec_stack = [False] * len(nodes)
    
    for node in range(len(nodes)):
        if not visited[node]:
            if dfs(node, visited, rec_stack, graph):
                return [1]  # Cyclic dependency detected (represented as 1)
    return [0]  # No cyclic dependency detected (represented as 0)


def evaluate_cyclic_dependency(data):
    results = []
    actuals = []
    
    for graph in data:
        nodes = graph['Nodes']
        edge_index = [graph['edge_index'][0], graph['edge_index'][1]]
        labels = graph['labels']
        
        # Actual label for cyclic dependency (1 if cyclic dependency exists, 0 otherwise)
        actual_cyclic = any(label[0] for label in labels)  # Assuming label[0] is the cyclic dependency label
        actuals.append(actual_cyclic)
        
        # Call the arcan_detect_smells function to get the prediction
        predicted_result = arcan_detect_smells(nodes, edge_index, None)
        results.append(predicted_result[0])  # Assuming result is either 0 or 1
    
    return results, actuals

import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_path = 'MicroservicesTestset/handgenerated1.jsonl'

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def aroma_detect_smells(nodes, edge_index):
    # Step 1: Convert edge_index into an adjacency list
    adjacency_list = {i: [] for i in range(len(nodes))}  # Initialize empty adjacency list for each node
    
    # Populate adjacency list from edge_index
    for i, j in zip(edge_index[0], edge_index[1]):
        adjacency_list[i].append(j)
    
    # Step 2: Define a helper function for DFS
    def dfs(node, visited, rec_stack):
        if node not in visited:  # Node not visited
            visited.add(node)
            rec_stack.add(node)

            # Explore all adjacent vertices
            for neighbor in adjacency_list.get(node, []):
                if neighbor not in visited and dfs(neighbor, visited, rec_stack):
                    return 1
                elif neighbor in rec_stack:
                    return 1

        rec_stack.remove(node)
        return 0
    
    # Step 3: Check for cycles using DFS
    visited = set()
    rec_stack = set()

    for node in range(len(nodes)):
        if node not in visited:  # If not visited, start DFS
            if dfs(node, visited, rec_stack):
                return True  # Cycle detected

    return False  # No cycles found


def evaluate_cyclic_dependency(data):
    results = []
    actuals = []
    
    for graph in data:
        nodes = graph['Nodes']
        edge_index = [graph['edge_index'][0], graph['edge_index'][1]]
        labels = graph['labels']
        
        # Actual label for cyclic dependency (1 if cyclic dependency exists, 0 otherwise)
        actual_cyclic = any(label[0] for label in labels)  # Assuming label[0] is the cyclic dependency label
        actuals.append(actual_cyclic)
        
        # Call the arcan_detect_smells function to get the prediction
        predicted_result = aroma_detect_smells(nodes, edge_index)
        results.append(predicted_result)  # Assuming result is either 0 or 1
    
    return results, actuals

def handler():
    data = load_data(test_path)
        # Evaluate Cyclic Dependency
    cyclic_results, cyclic_actuals = evaluate_cyclic_dependency(data)
    print('Cyclic Dependency Metrics:')
    print(f"Accuracy: {accuracy_score(cyclic_actuals, cyclic_results)}")
    print(f"Precision: {precision_score(cyclic_actuals, cyclic_results)}")
    print(f"Recall: {recall_score(cyclic_actuals, cyclic_results)}")
    print(f"F1 Score: {f1_score(cyclic_actuals, cyclic_results )}")
# Example usage
nodes = [
    [1, 0, 0, 0, 1, 5, 3, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 5, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 2, 0, 0, 0]
]
edge_index = [
    [1, 1, 2],
    [0, 2, 1]
]
edge_attr = [
    [10, 1000, 2000],
    [15, 1500, 2500],
    [5, 500, 1000]
]
# Example usage
#print(arcan_detect_smells(nodes, edge_index, edge_attr))
handler()

def handler():
    data = load_data(test_path)
        # Evaluate Cyclic Dependency
    cyclic_results, cyclic_actuals = evaluate_cyclic_dependency(data)
    print('Cyclic Dependency Metrics:')
    print(f"Accuracy: {accuracy_score(cyclic_actuals, cyclic_results)}")
    print(f"Precision: {precision_score(cyclic_actuals, cyclic_results)}")
    print(f"Recall: {recall_score(cyclic_actuals, cyclic_results)}")
    print(f"F1 Score: {f1_score(cyclic_actuals, cyclic_results )}")
# Example usage
nodes = [
    [1, 0, 0, 0, 1, 5, 3, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 5, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 2, 0, 0, 0]
]
edge_index = [
    [1, 1, 2],
    [0, 2, 1]
]
edge_attr = [
    [10, 1000, 2000],
    [15, 1500, 2500],
    [5, 500, 1000]
]
# Example usage
#print(arcan_detect_smells(nodes, edge_index, edge_attr))
handler()