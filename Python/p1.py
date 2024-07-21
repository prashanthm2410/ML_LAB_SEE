import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./ToyotaCorolla.csv')
x = dataset['KM']
y = dataset['Doors']
z = dataset['Price']

ax = plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,cmap="jet")
ax.set_title("3D Surface Plot")

plt.show()


from queue import PriorityQueue

def best_first_search(graph, start, goal, heuristic):
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start))
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
    if goal not in parent:
        return [],float('inf')
    
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    
    return path, path_cost.get(goal, float('inf'))

graph = {
    'S': {'A': 1, 'B': 2},
    'A': {'C': 3, 'D': 4},
    'B': {'E': 5, 'F': 6},
    'C': {},
    'D': {},
    'E': {'H': 1},
    'F': {'I': 2, 'G': 3},
    'H': {},
    'I': {},
    'G': {},
}

start_node = 'S'
goal_node = 'G'

heuristic_values = {
    'S': 13,
    'A': 12,
    'B': 4,
    'C': 7,
    'D': 3,
    'E': 8,
    'F': 2,
    'H': 4,
    'I': 9,
    'G': 0,
}

path, total_cost = best_first_search(graph, start_node, goal_node, heuristic_values)
if not path:
    print("Not Found")
else:
    print("Best First Search Path:", path)
    print("Total Cost:", total_cost)
    print("Number of Nodes Visited : ",len(path))