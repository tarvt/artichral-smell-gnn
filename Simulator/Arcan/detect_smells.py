import json

# Arcan uses static code analysis to build a dependency graph and detect architectural smells
def arcan_detect_smells(nodes, edge_index, edge_attr):
    # Placeholder function to simulate Arcan's detection of hard-coded endpoints
    def detect_hard_coded_endpoints(nodes):
        # Example logic to detect hard-coded endpoints
        return [int(node[1] == 1) for node in nodes]

    hardcoded_endpoints = detect_hard_coded_endpoints(nodes)
    
    # Construct output
    labels = [[0]*4 for _ in nodes]
    for i, is_hardcoded in enumerate(hardcoded_endpoints):
        labels[i][0] = is_hardcoded  # ESB Usage placeholder
    
    architecture = {
        "Nodes": nodes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "labels": labels,
        "description": "Arcan detected hard-coded endpoints using static code analysis."
    }
    
    return json.dumps(architecture, indent=4)


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
print(arcan_detect_smells(nodes, edge_index, edge_attr))