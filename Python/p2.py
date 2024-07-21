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



import heapq

def a_star_search_heapq(graph, start, goal, heuristic, cost):
    priority_queue = []
    heapq.heappush(priority_queue, (0 + heuristic[start], start))
    visited = set()
    g_cost = {start: 0}
    parent = {start: None}

    while priority_queue:
        current_f_cost, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == goal:
            break

        for neighbor in graph[current_node]:
            new_cost = g_cost[current_node] + cost[(current_node, neighbor)]
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic[neighbor]
                heapq.heappush(priority_queue, (f_cost, neighbor))
                parent[neighbor] = current_node

    path = []
    node = goal
    total_cost = 0
    while node is not None:
        if parent[node] is not None:  
                total_cost += cost[(parent[node], node)]
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, total_cost

graph = {
    'A': ['B', 'E'],
    'B': ['C', 'G'],
    'C': [],
    'D': ['G'],
    'E': ['D'],
    'G':[]
}

heuristic = {
    'A': 11,
    'B': 6,
    'C': 99,
    'D': 1,
    'E': 7,
    'G':0
}

cost = {
    ('A', 'B'): 2,
    ('A', 'E'): 3,
    ('B', 'C'): 1,
    ('B', 'G'): 9,
    ('D', 'G'): 1,
    ('E', 'D'): 6
}

start = 'A'
goal = 'G'

path, total_cost = a_star_search_heapq(graph, start, goal, heuristic, cost)
print("A* Search Path:", path)
print("Total Cost:", total_cost)