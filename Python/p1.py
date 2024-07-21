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
    pq.put((heuristic[start], start))
    total = 0

    while not pq.empty():
        h, node = pq.get()
        total += h
        
        if node == goal:
            print(f"Goal reached: {node}")
            print(f"Total heuristic cost: {total}")
            return
        
        if node not in visited:
            print(f"Visiting node: {node} ")
            visited.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    pq.put((heuristic[neighbor], neighbor))
    
    print("Goal not found!")
    print(f"Total heuristic cost: {total}")

graph = {
    'S': ['A', 'B'],
    'A': ['C','D'],
    'B': ['E', 'F'],
    'C':[],
    'D': [],
    'E': ['H'],
    'F': ['I', 'G'],
    'H': [],
    'I': [],
    'G': []
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
    'G':0
}

best_first_search(graph, start_node, goal_node, heuristic_values)
