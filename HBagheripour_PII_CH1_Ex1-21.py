# Social Network Chapter 1
# Haydeh Bagheripour
# Ex 1-21

# Problems:
# Download the email-Eu-core directed network from the SNAP dataset repository available at http://snap.stanford.edu/data/email-Eu-core.html.
# For this dataset compute the following network parameters:

import urllib.request
import gzip
import networkx as nx
import random
from matplotlib import pyplot as plt

# EX 1 "Number of nodes"
# EX 2 "Number of edges"

file_path = 'email-Eu-core.txt'
N = 1005

def count_node_eadges(file_path):
    nodes = set()
    eadges = 0

    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            nodes.add(sender)
            nodes.add(receiver)
            eadges += 1

    num_nodes = len(nodes)
    return num_nodes, eadges

num_nodes, num_eadges = count_node_eadges(file_path)
print('Result Question 1- 2: ')
print('Number of nodes: ', num_nodes)
print('Number of eadges: ', num_eadges)


'''result:
Number of nodes:  1005
Number of eadges:  25571'''




#  EX 2 "In-degree, out-degree and degree of the first five nodes"

def calculate_degrees(file_path):
    in_degrees = {}
    out_degrees = {}
    degrees = {}

    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            out_degrees[sender] = out_degrees.get(sender, 0) + 1
            in_degrees[receiver] = in_degrees.get(receiver, 0) + 1

    for node in in_degrees:
        degrees[node] = in_degrees[node] + out_degrees.get(node, 0)

    return in_degrees, out_degrees, degrees

in_degrees, out_degrees, degrees = calculate_degrees(file_path)

for node in list(in_degrees.keys())[:5]:
    print('Result Question 3: ')
    print('Node:', node)
    print('In_degree:', in_degrees.get(node, 0))
    print('Out_degree:', out_degrees.get(node, 0))
    print('Degree:', degrees.get(node, 0))
    print()
    

'''result:
Node: 1
In_degree: 51
Out_degree: 1
Degree: 52

Node: 3
In_degree: 62
Out_degree: 56
Degree: 118

Node: 4
In_degree: 74
Out_degree: 8
Degree: 163

Node: 6
In_degree: 93
Out_degree: 109
Degree: 202

Node: 7
In_degree: 49
Out_degree: 67
Degree: 116'''



#  EX 4 "Number of source nodes" &
#  EX 5 "Number of sink nodes" &
#  EX 6 "Number of isolated nodes"

file_path = 'email-Eu-core.txt'

def count_source_sink_isolated_nodes(file_path):
    receivers = set()
    sources = set()
    senders = set()
    sinks = set()

    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            receivers.add(receiver)
            senders.add(sender)
            sources.discard(receiver)
            sources.add(sender)
            sinks.discard(sender)
            sinks.add(receiver)

    num_source_nodes = len(sources)
    num_sink_nodes = len(sinks)
    isolated_nodes = senders.symmetric_difference(receivers)
    num_isolated_nodes = len(isolated_nodes)
    return num_source_nodes, num_sink_nodes, num_isolated_nodes

num_source_nodes, num_sink_nodes, num_isolated_nodes = count_source_sink_isolated_nodes(file_path)
print('Result Question 4-5-6: ')
print('Number of sources nodes:', num_source_nodes)
print('Number of sinks nodes:', num_sink_nodes)
print('Number of isolated nodes:', num_isolated_nodes)


'''result:
Number of sources nodes: 396
Number of sinks nodes: 644   
Number of isolated nodes: 151 '''






#  EX 7 "In-degree distribution" &
#  EX 8 "Out-degree distribution" 

import matplotlib.pyplot as plt
from collections import deque

file_path = 'email-Eu-core.txt'

def calculate_degree_distribution(file_path):
    in_degrees = {}
    out_degrees = {}

    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())

            out_degrees[sender] = out_degrees.get(sender, 0) + 1
            in_degrees[receiver] = in_degrees.get(receiver, 0) + 1

    in_degree_values = list(in_degrees.values())
    out_degree_values = list(out_degrees.values())

    return in_degree_values, out_degree_values

in_degree_values, out_degree_values = calculate_degree_distribution(file_path)

plt.hist(in_degree_values, bins=10, color='black')
plt.title('In_degree Distribution1')
plt.xlabel('In_degree')
plt.ylabel('Count')
plt.show()

plt.hist(out_degree_values, bins=10, color='black')
plt.title('Out_degree Distribution1')
plt.xlabel('Out_degree')
plt.ylabel('Count')
plt.show()



#  EX 9 "Average degree, average in-degree and average out-degree" 

file_path = 'email-Eu-core.txt'

def calculate_average_degrees(file_path):
    num_nodes = 0
    total_degree = 0
    total_in_degree = 0
    total_out_degree = 0

    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            total_out_degree += 1
            total_in_degree += 1

            num_nodes = max(num_nodes, sender, receiver)

    total_degree = total_out_degree + total_in_degree
    num_nodes += 1

    average_degree = total_degree / num_nodes
    average_in_degree = total_in_degree / num_nodes
    average_out_degree = total_out_degree / num_nodes

    return average_degree, average_in_degree, average_out_degree


# Example:
average_degree, average_in_degree, average_out_degree = calculate_average_degrees(file_path)

print("Average Degree:", average_degree)
print("Average In-Degree:", average_in_degree)
print("Average Out-Degree:", average_out_degree)




#  EX 10 "Distance between five pairs of random nodes"

import matplotlib.pyplot as plt
from collections import deque

file_path = 'email-Eu-core.txt'

def calculate_distance(file_path, pairs):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = []
            if receiver not in graph:
                graph[receiver] = []
            graph[sender].append(receiver)
            graph[receiver].append(sender)

    distances = []
    for pair in pairs:
        start, end = pair
        if start not in graph or end not in graph:
            distances.append(-1)  # One or both nodes not found in the graph
        else:
            visited = set()
            queue = deque([(start, 0)])  # (node, distance) tuple

            while queue:
                node, distance = queue.popleft()

                if node == end:
                    distances.append(distance)
                    break

                if node not in visited:
                    visited.add(node)
                    neighbors = graph[node]
                    queue.extend((neighbor, distance + 1) for neighbor in neighbors)

            if node != end:
                distances.append(-1)  # No path found between the nodes

    return distances


# Example:
pairs = [(1, 426), (3, 183), (5, 212), (7, 717), (9, 932)]  # Replace with your pairs of nodes
distances = calculate_distance(file_path, pairs)

for pair, distance in zip(pairs, distances):
    start, end = pair
    print(f"Distance between {start} and {end}: {distance}")



'''result: Distance between 1 and 426: 2
Distance between 3 and 183: 1
Distance between 5 and 212: 2
Distance between 7 and 717: 3
Distance between 9 and 932: 2'''





#  EX 11 "Shortest path length distribution" &
#  EX 12 "Diameter" 

import matplotlib.pyplot as plt
from collections import deque

file_path = 'email-Eu-core.txt'

def calculate_shortest_path_lengths(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = []
            if receiver not in graph:
                graph[receiver] = []
            graph[sender].append(receiver)
            graph[receiver].append(sender)

    diameter = 0

    shortest_path_lengths = []
    num_nodes = len(graph)

    for start_node in graph:
        distances = {}
        visited = set()
        queue = deque([(start_node, 0)])  # (node, distance) tuple

        while queue:
            node, distance = queue.popleft()

            if node not in visited:
                visited.add(node)
                distances[node] = distance
                neighbors = graph[node]
                queue.extend((neighbor, distance + 1) for neighbor in neighbors)

        max_distance = max(distances.values())
        diameter = max(diameter, max_distance)

        for end_node in graph:
            if start_node != end_node:
                shortest_path_lengths.append(distances.get(end_node, -1))

    return shortest_path_lengths, diameter


# Example :
shortest_path_lengths, diameter = calculate_shortest_path_lengths(file_path)

# Plotting the shortest path length distribution
print('Diameter is:', diameter)
plt.hist(shortest_path_lengths, bins=10, edgecolor='black')
plt.title("Shortest Path Length Distribution")
plt.xlabel("Shortest Path Length")
plt.ylabel("Count")
plt.show()




#  EX 13 "s the graph strongly connected? If so, compute the strongly connected component size distribution" 

import matplotlib.pyplot as plt
from collections import deque

file_path = 'email-Eu-core.txt'

def tarjan_strongly_connected_components(graph):
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    result = []

    def strongconnect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)

        for neighbor in graph[node]:
            if neighbor not in index:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in stack:
                lowlink[node] = min(lowlink[node], index[neighbor])

        if lowlink[node] == index[node]:
            component = []
            while True:
                neighbor = stack.pop()
                component.append(neighbor)
                if neighbor == node:
                    break
            result.append(component)

    for node in graph:
        if node not in index:
            strongconnect(node)

    return result

def is_strongly_connected(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = []
            if receiver not in graph:
                graph[receiver] = []
            graph[sender].append(receiver)

    sccs = tarjan_strongly_connected_components(graph)

    scc_sizes = [len(scc) for scc in sccs]
    scc_size_distribution = {size: scc_sizes.count(size) for size in set(scc_sizes)}

    return len(sccs) == 1, scc_size_distribution


# Example :
try:
    is_strongly_connected, scc_size_distribution = is_strongly_connected(file_path)

    print("Strongly Connected Component Size Distribution:")
    for size, count in scc_size_distribution.items():
        print(f"Component Size {size}: {count} component(s)")
    if is_strongly_connected:
        print("The graph is strongly connected.")
    else:
        print("The graph is not strongly connected.")
except KeyError as e:
    print(f"Error: Invalid node ID encountered: {e}")


'''result EX 13
Strongly Connected Component Size Distribution:
Component Size 1: 202 component(s)  
Component Size 803: 1 component(s)  
The graph is not strongly connected.'''





#  EX 14 "s the graph weakly connected? If so, compute the weakly connected component size distribution"

import matplotlib.pyplot as plt
from collections import deque

file_path = 'email-Eu-core.txt'


from collections import defaultdict, deque

def construct_undirected_graph(file_path):
    undirected_graph = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            undirected_graph[sender].append(receiver)
            undirected_graph[receiver].append(sender)
    return undirected_graph

def find_weakly_connected_components(graph):
    visited = set()
    weakly_connected_components = []

    for node in graph:
        if node not in visited:
            component = []
            queue = deque([node])
            while queue:
                current_node = queue.popleft()
                if current_node not in visited:
                    visited.add(current_node)
                    component.append(current_node)
                    queue.extend(graph[current_node])
            weakly_connected_components.append(component)

    return weakly_connected_components

def calculate_weakly_connected_component_size_distribution(file_path):
    undirected_graph = construct_undirected_graph(file_path)
    weakly_connected_components = find_weakly_connected_components(undirected_graph)

    size_distribution = defaultdict(int)
    for component in weakly_connected_components:
        size_distribution[len(component)] += 1

    return len(weakly_connected_components) == 1, size_distribution,


# Example:
is_weakly_connected, weakly_connected_component_size_distribution = calculate_weakly_connected_component_size_distribution(file_path)

print("Weakly Connected Component Size Distribution:")
for size, count in weakly_connected_component_size_distribution.items():
    print(f"Component Size {size}: {count} component(s)")

if is_weakly_connected:
    print("The graph is weakly connected.")
else:
    print("The graph is not weakly connected.")



'''result EX 14:
Weakly Connected Component Size Distribution:
Component Size 986: 1 component(s)
Component Size 1: 19 component(s) 
The graph is not weakly connected.'''




#  EX 15 "Number of bridge edges"

import matplotlib.pyplot as plt
from collections import deque

file_path = 'email-Eu-core.txt'


def find_bridge_edges(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = []
            if receiver not in graph:
                graph[receiver] = []
            graph[sender].append(receiver)
            graph[receiver].append(sender)

    num_bridge_edges = 0
    visited = set()
    lowlink = {}
    discovery = {}
    parent = {}

    def dfs(node):
        nonlocal num_bridge_edges
        visited.add(node)
        lowlink[node] = discovery[node] = len(visited)

        for neighbor in graph[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                dfs(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
                if lowlink[neighbor] > discovery[node]:
                    num_bridge_edges += 1
            else:
                if parent[node] != neighbor:
                    lowlink[node] = min(lowlink[node], discovery[neighbor])

    for node in graph:
        if node not in visited:
            parent[node] = None
            dfs(node)

    return num_bridge_edges


# Example:
num_bridge_edges = find_bridge_edges(file_path)

print("Number of Bridge Edges:", num_bridge_edges)


'''result EX 15 : 
Number of Bridge Edges: 95 '''





#  EX 16 "Number of articulation nodes"

import matplotlib.pyplot as plt
from collections import deque

file_path = 'email-Eu-core.txt'
def find_articulation_nodes(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = []
            if receiver not in graph:
                graph[receiver] = []
            graph[sender].append(receiver)
            graph[receiver].append(sender)

    num_articulation_nodes = 0
    visited = set()
    discovery = {}
    lowlink = {}
    parent = {}
    is_articulation = set()

    def dfs(node):
        nonlocal num_articulation_nodes
        visited.add(node)
        discovery[node] = len(visited)
        lowlink[node] = discovery[node]
        child_count = 0
        is_articulation_node = False

        for neighbor in graph[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                child_count += 1
                dfs(neighbor)

                lowlink[node] = min(lowlink[node], lowlink[neighbor])

                if parent[node] is None and child_count > 1:
                    is_articulation_node = True

                if parent[node] is not None and lowlink[neighbor] >= discovery[node]:
                    is_articulation_node = True
            elif neighbor != parent[node]:
                lowlink[node] = min(lowlink[node], discovery[neighbor])

        if is_articulation_node:
            num_articulation_nodes += 1
            is_articulation.add(node)

    for node in graph:
        if node not in visited:
            parent[node] = None
            dfs(node)

    return num_articulation_nodes, is_articulation


# Example :
num_articulation_nodes, articulation_nodes = find_articulation_nodes(file_path)

print("Number of Articulation Nodes:", num_articulation_nodes)
print("Articulation Nodes:", articulation_nodes)


'''result EX 16 
Number of Articulation Nodes: 73
Articulation Nodes: {641, 129, 258, 516, 5, 6, 263, 2, 777, 521, 393, 12, 269, 137, 271, 145, 405, 21, 408, 411, 
412, 157, 414, 543, 544, 417, 285, 38, 295, 936, 170, 560, 306, 564, 52, 55, 57, 189, 191, 577, 65, 321, 452, 327, 72, 712, 971,
717, 462, 82, 211, 84, 215, 87, 88, 605, 350, 96, 353, 121, 611, 231, 107, 748, 238, 495, 242, 376, 377, 506, 635, 380, 381}'''





#  EX 17 "Number of nodes in In(v) for five random nodes"

import matplotlib.pyplot as plt
from collections import deque
import random

file_path = 'email-Eu-core.txt'
N = 1005

def count_nodes_in_incoming_set(file_path, node_list):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = []
            if receiver not in graph:
                graph[receiver] = []
            graph[sender].append(receiver)

    num_nodes_in_incoming_set = []
    for node in node_list:
        if node in graph:
            incoming_set = set()
            for sender, receivers in graph.items():
                if node in receivers:
                    incoming_set.add(sender)
            num_nodes_in_incoming_set.append(len(incoming_set))
        else:
            num_nodes_in_incoming_set.append(0)

    return num_nodes_in_incoming_set


# Example:
random_nodes = random.sample(range(1, N + 1), 5)  # Replace num_nodes with the actual number of nodes in the graph
num_nodes_in_incoming_set = count_nodes_in_incoming_set(file_path, random_nodes)

for i, node in enumerate(random_nodes):
    print(f"Number of nodes in In({node}): {num_nodes_in_incoming_set[i]}")



'''result  EX 17:
Number of nodes in In(857): 4
Number of nodes in In(729): 8
Number of nodes in In(608): 23
Number of nodes in In(60): 39
Number of nodes in In(978): 3'''





#  EX 18 "Number of nodes in Out(v) for five random nodes"

import matplotlib.pyplot as plt
from collections import deque
import random

file_path = 'email-Eu-core.txt'
N = 1005


def count_nodes_in_outgoing_set(file_path, node_list):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = []
            if receiver not in graph:
                graph[receiver] = []
            graph[sender].append(receiver)

    num_nodes_in_outgoing_set = []
    for node in node_list:
        if node in graph:
            num_nodes_in_outgoing_set.append(len(graph[node]))
        else:
            num_nodes_in_outgoing_set.append(0)

    return num_nodes_in_outgoing_set


# Example:
random_nodes = random.sample(range(1, N + 1), 5)  # Replace num_nodes with the actual number of nodes in the graph
num_nodes_in_outgoing_set = count_nodes_in_outgoing_set(file_path, random_nodes)

for i, node in enumerate(random_nodes):
    print(f"Number of nodes in Out({node}): {num_nodes_in_outgoing_set[i]}")


'''result EX 18:
Number of nodes in Out(390): 44
Number of nodes in Out(469): 12
Number of nodes in Out(411): 95
Number of nodes in Out(445): 20
Number of nodes in Out(813): 24'''







#  EX 19 "Clustering coefficient for five random nodes"

import matplotlib.pyplot as plt
from collections import deque
import random

file_path = 'email-Eu-core.txt'
N = 1005

def calculate_clustering_coefficient(file_path, node_list):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = set()
            if receiver not in graph:
                graph[receiver] = set()
            graph[sender].add(receiver)
            graph[receiver].add(sender)

    clustering_coefficients = []
    for node in node_list:
        neighbors = graph.get(node, set())
        num_edges = len(neighbors)
        num_possible_edges = num_edges * (num_edges - 1) / 2 if num_edges >= 2 else 1
        num_actual_edges = 0

        for neighbor in neighbors:
            for neighbor2 in graph.get(neighbor, set()):
                if neighbor2 in neighbors:
                    num_actual_edges += 1

        clustering_coefficient = num_actual_edges / num_possible_edges if num_possible_edges > 0 else 0
        clustering_coefficients.append(clustering_coefficient)

    return clustering_coefficients


# Example :
random_nodes = random.sample(range(1, N + 1), 5)  # Replace num_nodes with the actual number of nodes in the graph
clustering_coefficients = calculate_clustering_coefficient(file_path, random_nodes)

for i, node in enumerate(random_nodes):
    print(f"Clustering coefficient for Node {node}: {clustering_coefficients[i]}")



'''result EX 19:
Clustering coefficient for Node 140: 0.8935897435897436
Clustering coefficient for Node 362: 0.7634271099744245
Clustering coefficient for Node 728: 1.0916666666666666
Clustering coefficient for Node 422: 0.906060606060606
Clustering coefficient for Node 761: 1.0'''






#  EX 20 "Clustering coefficient distribution"

import matplotlib.pyplot as plt
from collections import deque
import random

file_path = 'email-Eu-core.txt'
N = 1005


def calculate_clustering_coefficient_distribution(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = set()
            if receiver not in graph:
                graph[receiver] = set()
            graph[sender].add(receiver)
            graph[receiver].add(sender)

    clustering_coefficients = []
    for node in graph:
        neighbors = graph[node]
        num_edges = len(neighbors)
        num_possible_edges = num_edges * (num_edges - 1) / 2 if num_edges >= 2 else 1
        num_actual_edges = 0

        for neighbor in neighbors:
            for neighbor2 in graph.get(neighbor, set()):
                if neighbor2 in neighbors:
                    num_actual_edges += 1

        clustering_coefficient = num_actual_edges / num_possible_edges if num_possible_edges > 0 else 0
        clustering_coefficients.append(clustering_coefficient)

    return clustering_coefficients


# Example:
clustering_coefficients = calculate_clustering_coefficient_distribution(file_path)

# Plotting the clustering coefficient distribution
plt.hist(clustering_coefficients, bins=10, edgecolor='black')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Frequency')
plt.title('Clustering Coefficient Distribution')
plt.show()





#  EX 21 "Average clustering coefficient"

import matplotlib.pyplot as plt
from collections import deque
import random

file_path = 'email-Eu-core.txt'
N = 1005

def calculate_average_clustering_coefficient(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            sender, receiver = map(int, line.strip().split())
            if sender not in graph:
                graph[sender] = set()
            if receiver not in graph:
                graph[receiver] = set()
            graph[sender].add(receiver)
            graph[receiver].add(sender)

    total_coefficient = 0
    for node in graph:
        neighbors = graph[node]
        num_edges = len(neighbors)
        num_possible_edges = num_edges * (num_edges - 1) / 2 if num_edges >= 2 else 1
        num_actual_edges = 0

        for neighbor in neighbors:
            for neighbor2 in graph.get(neighbor, set()):
                if neighbor2 in neighbors:
                    num_actual_edges += 1

        clustering_coefficient = num_actual_edges / num_possible_edges if num_possible_edges > 0 else 0
        total_coefficient += clustering_coefficient

    num_nodes = len(graph)
    average_clustering_coefficient = total_coefficient / num_nodes if num_nodes > 0 else 0

    return average_clustering_coefficient


# Example:
average_clustering_coefficient = calculate_average_clustering_coefficient(file_path)

print(f"Average Clustering Coefficient: {average_clustering_coefficient}")

'''Average Clustering Coefficient: 1.188434825246865'''




