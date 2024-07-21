import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('./ToyotaCorolla.csv')
x = dataset['KM']
y = dataset['Weight']
z = dataset['Price']

plt.tricontourf(x, y, z, levels=20, cmap='jet')
plt.colorbar(label='Price')
plt.xlabel('KM')
plt.ylabel('Weight')
plt.title('Contour Plot')
plt.show()


from queue import PriorityQueue

def astar(graph, start, goal, heuristic):
    visited = set()
    pq = PriorityQueue()
    pq.put((0 + heuristic[start], start))
    parent = {start: None}
    path_cost = {start: 0}
    
    while not pq.empty():
        current_cost, node = pq.get()
        
        if node == goal:
            break
        
        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in graph[node].items():
                new_cost = path_cost[node] + edge_cost
                if neighbor not in path_cost or new_cost < path_cost[neighbor]:
                    path_cost[neighbor] = new_cost
                    total_cost = new_cost + heuristic[neighbor]
                    pq.put((total_cost, neighbor))
                    parent[neighbor] = node
    
    path = []
    node = goal
    total_cost = 0
    while node is not None:
        path.append(node)
        if parent[node] is not None:
            total_cost += graph[parent[node]][node]
        node = parent[node]
    path.reverse()
    
    return path, total_cost

graph = {
    'A': {'B': 2, 'E': 3},
    'B': {'C': 1, 'G': 9},
    'E': {'D': 6},
    'D': {'G': 1},
    'C': {},
    'G': {}  
}

start_node = 'A'
goal_node = 'G'

heuristic_values = {
    'A': 11,
    'B': 6,
    'C': 99,
    'E': 7,
    'D': 1,
    'G': 0
}

path, total_cost = astar(graph, start_node, goal_node, heuristic_values)
print("Best First Search Path:", path)
print("Total Cost:", total_cost)
print("Number of Nodes Visited:", len(path))
