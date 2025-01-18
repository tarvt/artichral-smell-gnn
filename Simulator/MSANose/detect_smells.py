import json
from collections import defaultdict, Counter
import numpy as np

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
        visited = [False] * len(self.adj_list)
        rec_stack = [False] * len(self.adj_list)

        for node in range(len(self.adj_list)):
            if not visited[node]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return True
        return False


class ESBService:
    def __init__(self, edge_index):
        self.edge_index = edge_index

    def get_esb_context(self):
        incoming = Counter(self.edge_index[1])
        outgoing = Counter(self.edge_index[0])
        total_connections = {node: incoming.get(node, 0) + outgoing.get(node, 0) for node in set(incoming) | set(outgoing)}

        # Calculate the threshold for being considered an ESB
        # Here, we look for nodes whose total connections are significantly higher than the average
        avg_connections = sum(total_connections.values()) / len(total_connections)
        threshold = avg_connections * 1.5  # Setting a simpler threshold, 50% above the average

        esb_candidates = [node for node, count in total_connections.items() if count > threshold]

        return {"candidateESBs": esb_candidates}


def detect_greedy_microservices(nodes):
    # Simulated detection of greedy microservices based on entity and static file counts
    greedy_services = []
    
    # Iterate over nodes, assuming node features are defined as mentioned
    # [5]: Instance Count, [6]: Number of Dependencies
    for index, node in enumerate(nodes):
        # Node Feature extraction for simulation
        instance_count = node[5]
        dependency_count = node[6]
        
        # Simplified heuristic: consider a microservice greedy if it has few dependencies but many instances
        if instance_count > 5 and dependency_count < 3:
            greedy_services.append(index)
    
    # Construct output
    labels = [0]*4 
    for i in greedy_services:
        labels[i] = 1  # Mark the node as greedy


    
    return labels

def detect_inappropriate_service_intimacy(nodes, edge_index):
    intimacy_issues = []
    max_index = len(nodes) - 1  # Maximum valid index

    for i in range(len(edge_index[0])):
        src = edge_index[0][i]
        dst = edge_index[1][i]

        # Check if source and destination indices are within the valid range
        if src > max_index or dst > max_index:
            print(f"Error: Node index out of range. src={src}, dst={dst}, max_index={max_index}")
            continue  # Skip this iteration

        # Avoid self-loops and ensure no duplication in checking
        if src != dst:
            src_node = nodes[src]
            dst_node = nodes[dst]

            # Extract node characteristics
            src_core = src_node[0]
            dst_core = dst_node[0]
            src_db = src_node[1]
            dst_db = dst_node[1]
            src_infra = src_node[2]
            dst_infra = dst_node[2]
            src_support = src_node[3]
            dst_support = dst_node[3]

            # Check for excessive dependency based on node features
            # If a core service is heavily dependent on another non-core service, it could be an intimacy issue
            if src_core and not dst_core:
                intimacy_issues.append({
                    "from": src,
                    "to": dst,
                    "reason": "Core service excessively dependent on a non-core service."
                })

            # Check if infrastructural services are improperly relied upon by non-infrastructure services
            if src_infra and not dst_infra:
                intimacy_issues.append({
                    "from": src,
                    "to": dst,
                    "reason": "Infrastructure service overly relied upon by a non-infrastructure service."
                })

    # Return a JSON formatted string with the results
    return json.dumps(intimacy_issues, indent=4)


