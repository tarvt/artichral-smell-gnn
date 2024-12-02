import json

# Aroma uses dynamic trace data to construct a graph and detect smells
def aroma_detect_smells(nodes, edge_index, edge_attr):
    # Simulate detection of cyclic dependencies using a placeholder function
    def detect_cyclic_dependency(graph):
        visited = set()
        stack = set()
        
        def visit(vertex):
            if vertex in visited:
                return False
            visited.add(vertex)
            stack.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor in stack or visit(neighbor):
                    return True
            stack.remove(vertex)
            return False
        
        return any(visit(v) for v in graph)

    # Construct the graph from edge index
    graph = {i: [] for i in range(len(nodes))}
    for src, dst in zip(edge_index[0], edge_index[1]):
        graph[src].append(dst)
    
    # Detect smells
    has_cyclic_dependency = detect_cyclic_dependency(graph)
    
    # Construct output
    labels = [[0]*4 for _ in nodes]  # Initialize labels for each node
    if has_cyclic_dependency:
        for node in graph:
            labels[node][1] = 1  # Mark cyclic dependency
    
    architecture = {
        "Nodes": nodes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "labels": labels,
        "description": "Aroma detected cyclic dependencies using dynamic trace data."
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

print(aroma_detect_smells(nodes, edge_index, edge_attr))