x ={
  "Nodes": [
    [1, 0, 0, 0, 1, 3, 2, 0, 0, 0],  
    [0, 1, 0, 0, 0, 2, 3, 0, 0, 0], 
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0], 
    [1, 0, 0, 0, 1, 4, 1, 0, 0, 0],  
    [0, 0, 0, 1, 0, 2, 4, 0, 0, 0],  
    [0, 1, 0, 0, 0, 3, 2, 0, 0, 0],  
    [0, 0, 0, 1, 0, 1, 2, 0, 0, 0]  
  ],
  "edge_index": [
    [0, 0, 1, 2, 3, 3, 4, 5],
    [1, 2, 2, 4, 5, 6, 1, 6]
  ],
  "edge_attr": [
    [20, 1500, 3000], 
    [10, 800, 1600],  
    [5, 500, 1200],   
    [15, 1000, 2000], 
    [30, 2000, 4000], 
    [10, 1200, 1800], 
    [25, 1700, 3300], 
    [5, 300, 600]    
  ],
  "labels": [
    [0, 0, 1, 0],  
    [0, 0, 0, 1],  
    [0, 1, 0, 0],  
    [0, 0, 1, 0],  
    [0, 0, 0, 0],  
    [0, 0, 0, 1],  
    [0, 0, 0, 0]   
  ],
  "description": "This JSON object represents a microservices architecture with 7 microservices. Each service's roles, interactions, and identified design anti-patterns are included, demonstrating interconnections and potential flaws within the system."
}
print(x)